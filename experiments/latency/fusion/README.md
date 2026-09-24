Projection fusion on the local RTX 5070 Ti

`best.json` keeps the round-one engine for up to 512 token rows. Above 512 rows it fuses the Wi projection and BF16-rounded GEGLU. Above 2,048 rows it also fuses QKV projection, FP32 RoPE and the masked attention tail. Both fusions fall back above 8,192 rows, the largest tuned shape. Row count is padded batch size times padded sequence length.

The Wi weights alternate activation/gate channels so one Tensor Core GEMM can produce the gated output directly. A complete 65,536-entry BF16 GELU table preserves PyTorch's intermediate rounding. Original weights stay resident for the fallback, so packing adds 300,941,312 bytes. The QKV kernel uses the original weight layout. Generated PTX targets `sm_120a` and contains BF16 `mma.sync`, `ldmatrix` and `cp.async`; representative PTX and register/spill metadata are saved with the results. These are software fusion gains measured on Blackwell, not evidence of gains exclusive to Blackwell.

The final gate passed 66 requests and 208 decisions with identical logits/actions. Twelve additional probes cover all timing workloads, changed input in a retained graph, and replay of the original input. Their 102 decisions/actions were also identical. These are synthetic implementation-parity checks, not labeled model-quality measurements.

The final matched experiment reduced 16-long request p50 from 88.1435 to 81.2315 ms and 16-short from 10.7907 to 10.4983 ms. Small-shape microkernel wins did not give stable full-request gains; the final selector retains existing projections there. `results/latency-optimizations/fusion/summary.json` derives the aggregates from saved raw samples. Measurements include tokenization, copies, inference and formatting; they exclude startup and HTTP.

Reproduce the full numerical gate and interleaved comparison:

```bash
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.fusion.benchmark --config experiments/latency/fusion/best.json --output results/latency-optimizations/fusion/reproduced.json --repeats 30 --rounds 3
```

Reproduce bounded tile and exact-epilogue screens:

```bash
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.fusion.microbench --batch 1 --length 64
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.fusion.microbench --batch 16 --length 64
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.fusion.microbench --batch 16 --length 512 --shortlist
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.fusion.epilogues
```

Integration uses `install(engine, **json.loads(Path("experiments/latency/fusion/best.json").read_text()))` before any graph capture. The engine must use the round-one `native-window` mode. New benchmark runs install the v2 `StableHostAdapter` for each compared engine, following its separate 120-eviction test. Saved fusion timings used the round-one adapter and passed every numerical probe; the adapter change does not change the model kernels.

This forward adapter derives from the repository's Apache-2.0 model computation. Retain the repository NOTICE and upstream notices. No round-one source or results were changed by this experiment.
