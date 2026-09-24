import argparse
import hashlib
import json
import pathlib
import time
import traceback

import torch

from .. import common
from .smooth_fp8 import SmoothFP8Linear

ROOT = (
    pathlib.Path(__file__).resolve().parents[3]
    / ".research"
    / "native-precision-smooth"
)


@torch.inference_mode()
def calibrate(output=None):
    engine = common.BlackwellEngine(common.model_path())
    values = {}
    handles = []
    report = {
        "metadata": common.metadata(),
        "calibration_indices": list(range(10)),
        "source": "Unmodified BF16 baseline activations, before each encoder MLP Wi and Wo linear",
        "scope": "Synthetic teacher calibration only. Indices 34..65 are not used for calibration.",
        "requests": [],
        "modules": {},
    }
    start = time.perf_counter()
    try:
        for i, layer in enumerate(engine.model.net.encoder.layers):
            for field in ("Wi", "Wo"):
                name = f"{i}.{field}"

                def hook(module, inputs, name=name):
                    x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                    maximum = x.abs().amax(dim=0)
                    values[name] = (
                        maximum
                        if name not in values
                        else torch.maximum(values[name], maximum)
                    )

                handles.append(
                    getattr(layer.mlp, field).register_forward_pre_hook(hook)
                )
        for index, request in enumerate(common.validation_requests()[:10]):
            prepared = engine.prepare(**request)
            shape = engine._shape(prepared)
            host = engine._allocate(shape, host=True)
            engine._fill(host, prepared)
            inputs = {key: value.to(engine.device) for key, value in host.items()}
            inputs["global_attention_unmasked"] = engine._graph_key(prepared)[-1]
            engine._forward(inputs)
            report["requests"].append(
                {
                    "index": index,
                    "shape": shape,
                    "input_tokens": prepared.input_tokens,
                    "sha256": hashlib.sha256(
                        json.dumps(request, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
            print("calibrated", index, shape, flush=True)
        torch.cuda.synchronize()
        assert len(values) == 56
        cpu = {key: value.cpu() for key, value in values.items()}
        for key, value in cpu.items():
            assert torch.isfinite(value).all() and (value >= 0).all()
            report["modules"][key] = {
                "channels": value.numel(),
                "minimum": value.min().item(),
                "maximum": value.max().item(),
            }
        torch.save(cpu, ROOT / "activation_amax.pt")
        report["calibration_ms"] = (time.perf_counter() - start) * 1000
        report["status"] = "passed"
        (output or ROOT / "calibration.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    finally:
        for handle in handles:
            handle.remove()
        engine.close()


def subset(validation, indices):
    rows = [row for row in validation["cases"] if row["case"] in indices]
    total = sum(row["decisions"] for row in rows)
    agreement = sum(row["agreement"] for row in rows)
    return {
        "requests": len(rows),
        "decisions": total,
        "agreement": agreement,
        "sdk_agreement": sum(row["sdk_agreement"] for row in rows),
        "unbucketed_sdk_agreement": sum(
            row["unbucketed_sdk_agreement"] for row in rows
        ),
        "max_probability_error": max(row["max_probability_error"] for row in rows),
        "passed_current_engine": agreement == total
        and max(row["max_probability_error"] for row in rows) <= 0.01,
        "failed_case_indices": [
            row["case"] for row in rows if row["agreement"] != row["decisions"]
        ],
        "indices": [row["case"] for row in rows],
    }


@torch.inference_mode()
def quantizer_probe(engine):
    import triton as tr

    from laya_blackwell.quantization import _quantize_rows

    from .smooth_fp8 import quantize_balanced_rows

    records = []
    for field in ("Wi", "Wo"):
        module = getattr(engine.model.net.encoder.layers[0].mlp, field)
        k = module.in_features
        rows = 64
        x = (
            torch.linspace(-16, 16, rows * k, device=engine.device)
            .reshape(rows, k)
            .bfloat16()
        )
        expected_input = x.float() * module.inverse_channel_scale[None, :]
        actual = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        expected = torch.empty_like(actual)
        actual_scale = torch.empty((rows, 1), device=engine.device)
        expected_scale = torch.empty_like(actual_scale)
        block = tr.next_power_of_2(k)
        warps = 4 if block <= 2048 else 8
        quantize_balanced_rows[(rows,)](
            x,
            actual,
            actual_scale,
            module.inverse_channel_scale,
            k,
            block,
            num_warps=warps,
        )
        _quantize_rows[(rows,)](
            expected_input, expected, expected_scale, k, block, num_warps=warps
        )
        mismatches = (
            (actual.view(torch.uint8) != expected.view(torch.uint8)).sum().item()
        )
        scale_error = (actual_scale - expected_scale).abs().max().item()
        record = {
            "module": field,
            "elements": actual.numel(),
            "fp8_byte_mismatches": mismatches,
            "max_scale_error": scale_error,
            "reference": "Existing row quantizer applied to an explicit FP32 X/S intermediate; no BF16 rescale rounding",
        }
        records.append(record)
        assert mismatches == 0 and scale_error == 0, record
    return records


def run(alpha, output=None):
    report = {
        "variant": f"smooth-fp8-mlp-{alpha}",
        "alpha": alpha,
        "metadata": common.metadata(),
        "method": "FP32 channel balancing, static FP8 output-row weight scales and fused dynamic FP8 activation-row quantization; all 56 encoder MLP linears only.",
        "calibration_indices": list(range(10)),
        "heldout_indices": list(range(34, 66)),
        "scope": "Teacher numerical parity, not labeled accuracy. 32 calibration-heldout requests are reused across the three preselected alphas.",
        "modules": {},
    }
    engine = None
    path = output or ROOT / f"alpha-{alpha}.json"
    try:
        engine = common.BlackwellEngine(common.model_path())
        calibration = torch.load(
            ROOT / "activation_amax.pt", map_location="cpu", weights_only=True
        )
        start = time.perf_counter()
        for i, layer in enumerate(engine.model.net.encoder.layers):
            for field in ("Wi", "Wo"):
                name = f"{i}.{field}"
                replacement, info = SmoothFP8Linear.from_linear(
                    getattr(layer.mlp, field), calibration[name], alpha
                )
                setattr(layer.mlp, field, replacement)
                report["modules"][name] = info
        torch.cuda.synchronize()
        report["quantization_setup_ms"] = (time.perf_counter() - start) * 1000
        report["rows"] = common.benchmark(
            engine, cases=[(1, "short"), (16, "long")], repeats=50, warmups=10
        )
        path.write_text(json.dumps(report, indent=2) + "\n")
        report["validation"] = common.validate(engine)
        report["calibration_parity"] = subset(report["validation"], range(10))
        report["heldout_parity"] = subset(report["validation"], range(34, 66))
        report["other_uncalibrated_parity"] = subset(
            report["validation"], range(10, 34)
        )
        if alpha == 0.75:
            report["quantizer_probe"] = quantizer_probe(engine)
        report["status"] = "complete"
    except (RuntimeError, ValueError, AssertionError, OSError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        if engine is not None:
            engine.close()
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("variant", "status", "heldout_parity")
                    if key in report
                }
            ),
            flush=True,
        )


def verify_quantizer(alpha, output=None):
    engine = common.BlackwellEngine(common.model_path())
    try:
        calibration = torch.load(
            ROOT / "activation_amax.pt", map_location="cpu", weights_only=True
        )
        for field in ("Wi", "Wo"):
            replacement, _ = SmoothFP8Linear.from_linear(
                getattr(engine.model.net.encoder.layers[0].mlp, field),
                calibration[f"0.{field}"],
                alpha,
            )
            setattr(engine.model.net.encoder.layers[0].mlp, field, replacement)
        record = {"status": "passed", "alpha": alpha, "checks": quantizer_probe(engine)}
        (output or ROOT / "quantizer-probe.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record), flush=True)
    finally:
        engine.close()


def main():
    global ROOT
    parser = argparse.ArgumentParser(
        description="Calibrated SmoothQuant-style FP8 MLP experiment; synthetic teacher parity only."
    )
    parser.add_argument(
        "--task", choices=["calibrate", "run", "verify-quantizer"], default="run"
    )
    parser.add_argument("--alpha", type=float, choices=[0.25, 0.5, 0.75])
    parser.add_argument("--work-dir", type=pathlib.Path, default=ROOT)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    ROOT = args.work_dir.expanduser().resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.task == "calibrate":
        calibrate(args.output)
    else:
        if args.alpha is None:
            parser.error("--alpha is required for quantized execution")
        if args.task == "verify-quantizer":
            verify_quantizer(args.alpha, args.output)
        else:
            run(args.alpha, args.output)


if __name__ == "__main__":
    main()
