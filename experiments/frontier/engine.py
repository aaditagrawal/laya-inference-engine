"""Fixed small-shape matrix choices; all other shapes retain the native path."""

import json
import types
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine

from .matmul import matmul

ROOT = Path(__file__).resolve().parents[2]


class _ShortShapeCompiler(torch.nn.Module):
    def __init__(self, original, compiled):
        super().__init__()
        self.original = original
        self.compiled = compiled

    def forward(self, input_ids, *args, **kwargs):
        model = self.compiled if input_ids.numel() == 64 else self.original
        return model(input_ids, *args, **kwargs)


class FrontierEngine(V2Engine):
    def __init__(
        self,
        *,
        policy="bf16-exact",
        attention=None,
        fuse_reduce_norm=False,
        token_tables=False,
        prefetch=None,
        prefetch_chunk=16384,
        packed_qkv=False,
        fuse_mlp_geglu=False,
        mlp_geglu_unpacked=False,
        head_kernels=False,
        host_prepare=None,
        attention_special=False,
        global_attention=False,
        host_runtime=False,
        native_format=False,
        **kwargs,
    ):
        if global_attention and not attention_special:
            raise ValueError(
                "Global attention specialization requires attention_special"
            )
        if attention_special and (attention != "native" or not token_tables):
            raise ValueError(
                "Attention specialization requires native attention and token tables"
            )
        if attention not in {None, "triton", "cudnn", "native"}:
            raise ValueError(attention)
        if mlp_geglu_unpacked and not fuse_mlp_geglu:
            raise ValueError("Unpacked MLP weights require MLP/GEGLU fusion")
        if host_prepare not in {None, "single", "batch", "template"}:
            raise ValueError(host_prepare)
        compile_model = policy.endswith("-compiled")
        policy = policy.removesuffix("-compiled")
        short_compile = policy.endswith("-short") and compile_model
        if short_compile:
            policy = policy.removesuffix("-short")
        if policy not in {
            "bf16-exact",
            "bf16-tma-exact",
            "bf16-pipeline-exact",
            "bf16-splitk-exact",
            "bf16-fast",
            "lt-exact",
            "lt-fast",
            "mx-all",
            "mx-mlp",
            "wfp8-all",
            "nvfp4-all",
        }:
            raise ValueError(policy)
        super().__init__(optimization="native", **kwargs)
        self.policy, self.plans, self.selection = policy, {}, {}
        try:
            filename = (
                "matmul-nvfp4.json"
                if policy.startswith("nvfp4")
                else "matmul-weight-fp8.json"
                if policy.startswith("wfp8")
                else "matmul-mxfp8.json"
                if policy.startswith("mx")
                else "matmul-cublaslt.json"
                if policy.startswith("lt")
                else "matmul-bf16.json"
            )
            report = json.loads((ROOT / "results/frontier" / filename).read_text())
            if policy in {"bf16-tma-exact", "bf16-pipeline-exact", "bf16-splitk-exact"}:
                tma = json.loads(
                    (ROOT / "results/frontier/matmul-tma.json").read_text()
                )
                report["rows"].extend(tma["rows"])
            if policy in {"bf16-pipeline-exact", "bf16-splitk-exact"}:
                pipeline = json.loads(
                    (ROOT / "results/frontier/matmul-pipeline.json").read_text()
                )
                report["rows"].extend(pipeline["rows"])
            if policy == "bf16-splitk-exact":
                splitk = json.loads(
                    (ROOT / "results/frontier/matmul-splitk.json").read_text()
                )
                report["rows"].extend(splitk["rows"])
            for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]:
                if policy == "mx-mlp" and not field.startswith("mlp"):
                    continue
                rows = [
                    row
                    for row in report["rows"]
                    if row["field"] == field and row.get("speedup", 0) > 1.02
                ]
                if policy.endswith("exact"):
                    rows = [row for row in rows if row.get("mismatches") == 0]
                if not rows:
                    continue
                selected = min(rows, key=lambda row: row["ms"])
                self.selection[field] = selected
                group, attr = field.split(".")
                for layer in self.base.model.net.encoder.layers:
                    module = getattr(getattr(layer, group), attr)
                    if module.bias is not None:
                        raise ValueError(
                            "This matrix experiment only supports bias-free encoder projections"
                        )
                    original = module.forward
                    if selected["variant"] == "tma":
                        from .tma import compiled_matmul as tma_matmul

                        config = tuple(selected["config"])

                        def forward(module, x, original=original, config=config):
                            if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                                return tma_matmul(
                                    x, module.weight, [int(v) for v in config]
                                )
                            return original(x)
                    elif policy.startswith("nvfp4"):
                        from .nvfp4 import matmul as fp4_matmul
                        from .nvfp4 import quantize

                        q, scales, outer = quantize(module.weight)
                        module.register_buffer("frontier_quantized", q)
                        module.register_buffer("frontier_scales", scales)
                        module.register_buffer("frontier_outer", outer)
                        config = tuple(selected["config"])

                        def forward(module, x, original=original, config=config):
                            if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                                return fp4_matmul(
                                    x,
                                    module.frontier_quantized,
                                    module.frontier_scales,
                                    module.frontier_outer,
                                    config,
                                )
                            return original(x)
                    elif policy.startswith("wfp8"):
                        from .matmul import quantize_fp8_weight

                        q, scales = quantize_fp8_weight(module.weight)
                        module.register_buffer("frontier_quantized", q)
                        module.register_buffer("frontier_scales", scales)
                        config = tuple(selected["config"])

                        def forward(module, x, original=original, config=config):
                            if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                                return matmul(
                                    x,
                                    module.frontier_quantized,
                                    config,
                                    module.frontier_scales,
                                    quant_mode=2,
                                )
                            return original(x)
                    elif policy.startswith("mx"):
                        from .mxfp8 import matmul as mx_matmul
                        from .mxfp8 import quantize

                        q, scales = quantize(module.weight)
                        module.register_buffer("frontier_quantized", q)
                        module.register_buffer("frontier_scales", scales)
                        config = tuple(selected["config"])

                        def forward(module, x, original=original, config=config):
                            if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                                return mx_matmul(
                                    x,
                                    module.frontier_quantized,
                                    module.frontier_scales,
                                    config,
                                )
                            return original(x)
                    elif policy.startswith("lt"):
                        from .lt import Plan

                        n, k = module.weight.shape
                        if field not in self.plans:
                            self.plans[field] = Plan(64, n, k)
                            if (
                                dict(self.plans[field].native.info(selected["index"]))
                                != selected["algorithm"]
                            ):
                                raise RuntimeError(
                                    "cuBLASLt algorithm enumeration changed; retune on this stack"
                                )
                        plan = self.plans[field]

                        def forward(
                            module,
                            x,
                            original=original,
                            plan=plan,
                            index=selected["index"],
                        ):
                            if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                                return plan(x, module.weight, index)
                            return original(x)
                    else:
                        config = tuple(selected["config"])

                        def forward(
                            module,
                            x,
                            original=original,
                            config=config,
                            partial_bf16=selected["variant"] == "bf16-partial",
                        ):
                            if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                                return matmul(
                                    x, module.weight, config, partial_bf16=partial_bf16
                                )
                            return original(x)

                    module.forward = types.MethodType(forward, module)
            original_model = self.base.model
            if packed_qkv:
                from .packed_tma import install_qkv

                self.selection["packed_qkv"] = install_qkv(original_model)
            if token_tables:
                from .token_tables import install as install_token_tables

                self.selection["token_tables"] = install_token_tables(original_model)
            if prefetch is not None:
                if not token_tables:
                    raise ValueError(
                        "Prefetch probe currently requires the short token-table forward"
                    )
                from .prefetch import install as install_prefetch

                install_prefetch(original_model, prefetch, prefetch_chunk)
                self.selection["prefetch"] = {
                    "mode": prefetch,
                    "chunk_bytes": prefetch_chunk,
                }
            if fuse_reduce_norm:
                from .reduce_norm import install as install_reduce_norm

                install_reduce_norm(original_model, self.selection["mlp.Wo"])
                self.selection["reduce_norm"] = {"partials": 4, "encoder_layers": 27}
            if fuse_mlp_geglu:
                if not token_tables or prefetch in {"geglu", "both"}:
                    raise ValueError(
                        "MLP fusion probe requires token tables and an unmodified GEGLU call"
                    )
                from .mlp_geglu import install as install_mlp_geglu

                self.selection["mlp_geglu"] = install_mlp_geglu(
                    original_model, unpacked=mlp_geglu_unpacked
                )
            if head_kernels:
                from .head_install import install as install_head

                self.selection["head_kernels"] = install_head(
                    original_model, ROOT / "results/frontier/head-gemm.json"
                )
            if compile_model:
                from experiments.native.compiler.policy import install_compiler

                original = self.base.model
                install_compiler(self.base)
                if short_compile:
                    self.base.model = _ShortShapeCompiler(
                        original, self.base.model
                    ).eval()
                self.policy += "-short-compiled" if short_compile else "-compiled"
            if attention is not None:
                from .attention import install

                # Compiler preservation copies the functional namespace. Install
                # after that copy and before the first trace/capture.
                install(original_model, attention)
                self.policy += "-attn-" + attention
                self.selection["attention"] = {
                    "kind": attention,
                    "shape": [1, 16, 64, 64],
                    "config": [16, False, 1, 4] if attention == "triton" else None,
                    "expected_exact": attention == "native",
                }
            if fuse_reduce_norm:
                self.policy += "-reduce-norm"
            if token_tables:
                self.policy += "-token-tables"
            if prefetch is not None:
                self.policy += "-prefetch-" + prefetch + "-" + str(prefetch_chunk)
            if packed_qkv:
                self.policy += "-packed-qkv"
            if fuse_mlp_geglu:
                self.policy += "-mlp-geglu"
                if mlp_geglu_unpacked:
                    self.policy += "-unpacked"
            if head_kernels:
                self.policy += "-head-kernels"
            if host_prepare is not None:
                from .host_prepare import install as install_host_prepare

                install_host_prepare(self, host_prepare)
                self.selection["host_prepare"] = {"mode": host_prepare}
                self.policy += "-host-" + host_prepare
            if attention_special:
                from .attention_special_adapter import (
                    install as install_attention_special,
                )

                self.selection["attention_special"] = install_attention_special(
                    self, (32, True, True), include_padding=True
                )
                self.policy += "-attention-special"
            if global_attention:
                from .global_attention_adapter import (
                    install as install_global_attention,
                )

                self.selection["global_attention"] = install_global_attention(
                    self, include_padding=True
                )
                self.policy += "-global-attention"
            if host_runtime:
                from .host_runtime import install as install_host_runtime

                install_host_runtime(self)
                self.selection["host_runtime"] = {
                    "mode": "owned-numpy-views-and-direct-cache-hit"
                }
                self.policy += "-host-runtime"
            if native_format:
                from .native_format import install as install_native_format

                install_native_format(self)
                self.selection["native_format"] = {
                    "mode": "cpp-numpy-fp32-loops-with-reference-fallback",
                    "numpy_version": "2.5.3",
                }
                self.policy += "-native-format"
        except BaseException:
            self.close()
            raise

    def close(self):
        super().close()
        if hasattr(self, "plans"):
            self.plans.clear()
