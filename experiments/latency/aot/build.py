"""Export one fixed request shape and build a deployable AOTInductor package."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch

from experiments.native.compiler import install_padded_rope_window, pin_libdevice
from experiments.native.compiler.preserve_ops import install as preserve_ops
from experiments.native.kernels import install as install_kernels
from laya_blackwell.engine import KEYS, MODEL_ID, BlackwellEngine
from laya_blackwell.workloads import workload


class FixedShapeModel(torch.nn.Module):
    def __init__(self, model, unmasked):
        super().__init__()
        self.model = model
        self.unmasked = unmasked

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_pos=marker_pos,
            marker_mask=marker_mask,
            qtype=qtype,
            global_attention_unmasked=self.unmasked,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="1-short")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--packed-inputs", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    args.package.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    from torch._inductor import config

    config.compile_threads = 2
    report = {
        "case": args.case,
        "package": str(args.package.resolve()),
        "torch": torch.__version__,
        "libdevice": pin_libdevice(),
        "status": "initializing",
        "strict_export": args.strict,
        "packed_inputs": args.packed_inputs,
    }
    engine = None
    started = time.perf_counter()

    def checkpoint():
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        engine = BlackwellEngine(max_graphs=1)
        install_kernels(engine, "cuda_vector_norm_triton_geglu_corrected")
        install_padded_rope_window(engine)
        preserve_ops(engine)
        batch, length = args.case.split("-")
        prepared = engine.prepare(**workload(int(batch), length))
        shape = engine._shape(prepared)
        host = engine._allocate(shape, host=True)
        engine._fill(host, prepared)
        if args.packed_inputs:
            from experiments.native.host.adapter import packed_allocate

            _storage, views = packed_allocate(
                shape, engine.agent.tok.pad_token_id, engine.device
            )
            for key in KEYS:
                views[key].copy_(host[key])
            inputs = tuple(views[key] for key in KEYS)
        else:
            inputs = tuple(host[key].to(engine.device) for key in KEYS)
        model = FixedShapeModel(engine.model, engine._graph_key(prepared)[-1])
        report.update(
            shape=list(shape), unmasked=model.unmasked, hardware=engine.hardware
        )
        report["input_alignment_mod16"] = [x.data_ptr() % 16 for x in inputs]
        runtime = args.package.parent / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "rl_agent_config.json").write_text(
            json.dumps(engine.agent.cfg, indent=2) + "\n"
        )
        engine.agent.tok.save_pretrained(runtime / "tokenizer")
        args.package.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "shape": list(shape),
                    "unmasked": model.unmasked,
                    "torch": torch.__version__,
                    "libdevice": report["libdevice"],
                    "sequence_buckets": list(engine.sequence_buckets),
                    "runtime_directory": "runtime",
                    "case": args.case,
                    "model_id": MODEL_ID,
                    "model_revision": engine.revision,
                },
                indent=2,
            )
            + "\n"
        )
        with torch.inference_mode():
            expected = model(*inputs)
            torch.cuda.synchronize()
            torch.save(
                {
                    "inputs": tuple(x.cpu() for x in inputs),
                    "expected": tuple(x.cpu() for x in expected),
                },
                args.package.with_suffix(".inputs.pt"),
            )
            report["model_setup_ms"] = (time.perf_counter() - started) * 1000
            report["status"] = "exporting"
            checkpoint()
            export_start = time.perf_counter()
            exported = torch.export.export(model, inputs, strict=args.strict)
            report["export_ms"] = (time.perf_counter() - export_start) * 1000
            report["export_nodes"] = len(list(exported.graph.nodes))
            report["status"] = "compiling"
            checkpoint()
            compile_start = time.perf_counter()
            torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(args.package.resolve()),
                inductor_configs={
                    "triton.cudagraphs": False,
                    "emulate_precision_casts": True,
                    "compile_threads": 2,
                },
            )
            report["compile_package_ms"] = (time.perf_counter() - compile_start) * 1000
            report["package_bytes"] = args.package.stat().st_size
            report["package_sha256"] = hashlib.file_digest(
                args.package.open("rb"), "sha256"
            ).hexdigest()
            report["status"] = "loading"
            checkpoint()
            load_start = time.perf_counter()
            loaded = torch._inductor.aoti_load_package(str(args.package.resolve()))
            report["same_process_load_ms"] = (time.perf_counter() - load_start) * 1000
            actual = loaded(*inputs)
            torch.cuda.synchronize()
            report["same_process_exact"] = [
                torch.equal(a, b) for a, b in zip(actual, expected)
            ]
            report["same_process_max_error"] = [
                (a.float() - b.float()).abs().max().item()
                for a, b in zip(actual, expected)
            ]
            report["status"] = "complete"
    except Exception as error:  # noqa: BLE001 - preserve the exact experimental failure.
        report["failed_stage"] = report["status"]
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        if engine is not None:
            engine.close()
        checkpoint()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
