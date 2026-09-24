# CPU, GPU and hybrid comparison on RTX A6000

On 24 September 2026 IST, this repository and the CPU, GPU and hybrid implementations from the comparison project were benchmarked on **the RTX A6000 system**. The compiled GPU variants had the lowest measured warm latency. This engine beat the uncompiled GPU and hybrid variants, but did not beat their compiled versions.

## Warm latency

Each line lists median milliseconds in this order: **one short question / sixteen short questions / one long question / sixteen long questions**. Actual input-token counts were 63 / 993 / 512 / 8,192. Long requests use 512 tokens per question.

- This engine, BF16 with FP32 residuals: **3.57 / 13.03 / 8.69 / 99.52 ms**.
- Comparison GPU, FP16: **4.47 / 17.45 / 10.75 / 145.97 ms**.
- Comparison compiled GPU, FP16: **2.57 / 11.05 / 8.14 / 88.78 ms**.
- Comparison GPU, BF16 with FP32 residuals: **4.48 / 17.35 / 11.14 / 140.86 ms**.
- Comparison compiled GPU, BF16 with FP32 residuals: **2.63 / 10.92 / 7.97 / 87.40 ms**.
- Comparison hybrid, FP16 GPU and FP32 CPU scorer: **4.60 / 18.33 / 11.01 / 147.35 ms**.
- Comparison compiled hybrid, FP16 GPU and FP32 CPU scorer: **2.78 / 12.17 / 8.27 / 90.17 ms**.
- Comparison CPU, BF16 MLPs, 32 threads: **58.81 / 290.20 / 206.39 / 2942.21 ms**.
- Comparison CPU pool, 12 workers with 8 threads each: **75.41 / 153.21 / 379.23 / 994.49 ms**.

For one short question, compiled FP16 GPU used 2.57 ms versus 3.57 ms for this engine, about 28% lower latency. For sixteen long questions, compiled mixed BF16 used 87.40 ms versus 99.52 ms, about 12% lower latency. Small differences between the compiled precision variants are not statistically established by this run.

With BF16 matrix weights and FP32 residuals, the comparison GPU implementation went from 4.48 ms without compilation to 2.63 ms with compilation for the short single-question case. Its sixteen-long-question result went from 140.86 to 87.40 ms. These measurements show a software optimization benefit on Ampere. They do not quantify a Blackwell hardware benefit.

Full per-case p50, p95, mean, decisions/s, first-call times and numerical checks are in [summary.csv](../results/rtx-a6000/summary.csv) and [summary.json](../results/rtx-a6000/summary.json). Individual profile JSON files retain all 20 timing samples and responses. Throughput uses question count divided by mean wall time. It is not generated-token TPS or a measurement of maximum concurrent serving capacity.

## Method and environment

- RTX A6000, 48 GB, compute capability 8.6, driver 580.173.02; Threadripper PRO 7995WX, 96 physical cores. An existing idle Laya GPU server remained resident throughout the experiment.
- Python 3.12.10, PyTorch 2.14.0 with CUDA 13.0, Transformers 5.17.0, Triton 3.8.0, Laya 0.3.9. All profiles used the same runtime. The comparison project originally used Laya 0.3.7; an isolated 0.3.9 overlay normalized the SDK version without changing its environment. This is a controlled source-snapshot comparison, not a replay of the untouched original package environment.
- Same English checkpoint, revision `5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`. Weight SHA-256: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
- Frozen source snapshots and identical [request JSON](../results/rtx-a6000/requests.json). Remote/local SHA-256 comparisons verified all 12 engine modules, 32 comparison source modules, requests and measurement script.
- One process per profile, sequential execution, five warmups and twenty serial timed requests per shape. Each timed synchronous `predict` includes tokenization, input copies, model execution, output copies and response formatting. No answer cache.
- Model loading, first use, warmup, HTTP, wire JSON serialization and SSH transport are excluded equally. First-call and model-load costs are recorded separately. These are not clean-machine cold starts; filesystem and compiler caches can be reused.
- This engine uses a **benchmark-only override of the hardware guard** to run its existing BF16/Triton path on Ampere. Published runtime modules are unchanged and still require Blackwell. The comparison uses CUDA 13.0 rather than the published Blackwell CUDA 13.2 environment.
- Precision, host threads and padding follow each recorded profile. GPU uses four host threads; compiled hybrid uses one, ordinary hybrid four; this engine inherits 96 PyTorch threads. CPU pool workers use eight threads each. This engine buckets the short sequence to 64 positions, while the comparison implementation uses its exact shape.
- CPU uses BF16 MLPs and FP32 attention/residuals. CPU pool shards questions within a request across up to twelve workers; one question uses one worker. The serial hybrid benchmark does not exercise its `predict_many` pipeline. GPU graph caches held all four tested shapes, with no capacity fallback.

First use matters: this engine took 395 ms for its first short request after loading; compiled FP16 GPU took 36.7 seconds and compiled hybrid 28.9 seconds. Those compilation/capture costs are separate from the warm results above.

## Response checks

Every profile matched 34 of 34 top options from the CPU FP32 reference, with matching request hashes, input-token counts, response schemas and categorical answer values. Maximum absolute differences below use SDK-rounded option probabilities and the scalar noul probability.

- This engine, BF16 with FP32 residuals: probability difference 0.0029; largest other numeric difference 0.0048.
- Comparison GPU, FP16: probability difference 0.0008; largest other numeric difference 0.0013.
- Comparison compiled GPU, FP16: probability difference 0.0010; largest other numeric difference 0.0016.
- Comparison GPU, BF16 with FP32 residuals: probability difference 0.0020; largest other numeric difference 0.0033.
- Comparison compiled GPU, BF16 with FP32 residuals: probability difference 0.0056; largest other numeric difference 0.0106.
- Comparison hybrid, FP16 GPU and FP32 CPU scorer: probability difference 0.0009; largest other numeric difference 0.0014.
- Comparison compiled hybrid, FP16 GPU and FP32 CPU scorer: probability difference 0.0009; largest other numeric difference 0.0013.
- Comparison CPU, BF16 MLPs, 32 threads: probability difference 0.0010; largest other numeric difference 0.0020.
- Comparison CPU pool, 12 workers with 8 threads each: probability difference 0.0010; largest other numeric difference 0.0013.

Other numeric values include score, confidence and action probability. Top options use argmax of rounded choice/score probabilities, breaking ties by response order; noul uses a 0.5 threshold. Four synthetic requests with repeated questions provide a limited regression check, not labeled task accuracy or calibration.

## Native implementation opportunities

C++/CUDA or Rust with CUDA bindings can implement this engine. The useful next targets are matrix-kernel selection and fusion, windowed attention, and input/output buffer handling. Most arithmetic already runs in CUDA or Triton, and warm requests already replay a CUDA Graph. A language rewrite alone has no demonstrated speedup.

This engine measured 3.57 ms end to end for one short question, while a separate CUDA-event measurement of device graph replay measured 3.23 ms. Those different timing methods do not define a strict overhead bound, but they show that GPU work dominates this case. [CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

Concrete experiments are tuned cuBLASLt/CUTLASS GEMMs; fusing residual addition, LayerNorm and BF16 conversion while preserving required rounding; attention that uses window bounds instead of a dense local mask; fused GELU/gating epilogues; and projecting final-head queries only for selected CLS/marker positions. Pack reusable pinned input/output buffers before replacing the host runtime. Test compiler fusion of the optimized model before committing to a full native rewrite.

A6000 kernels target SM86 and BF16/FP16. RTX Blackwell SM120 and data-center Blackwell SM100 require different kernel families; SM100 tensor-memory instructions are not a universal Blackwell path. FP8/FP4 experiments need their own output checks. [CUTLASS architecture guidance](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

PTX is an intermediate instruction set translated to target machine instructions. Hand-written PTX is useful for a demonstrated compiler limitation, not an automatic source of speed. [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## Reproduction

The measurements used an isolated `.comparison-20260924-blackwell` directory on the RTX A6000 system. The comparison project's source files, virtual environment and existing server were preserved. The measurement script changes only its own in-process hardware-check function when `--allow-non-blackwell` is provided.

From that directory, use the existing interpreter and isolated SDK overlay:

```bash
export PYTHONPATH="$PWD/vendor:$PWD/src:$PWD/source"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor"
export TRITON_CACHE_DIR="$PWD/cache/triton"
~/.local/bin/uv run --no-project --python ../.venv/bin/python python compare_implementations.py \
  --source source --model ../models/laya --requests requests.json \
  --profile blackwell --allow-non-blackwell --output results/blackwell.json
```

Other profiles are `reference`, `cpu`, `cpu-pool`, `gpu`, `gpu-mixed-bf16`, `gpu-compiled`, `gpu-compiled-mixed-bf16`, `hybrid`, and `hybrid-compiled`. `reference` records one FP32 CPU answer per case without timed repetitions.

In this repository, regenerate the checked aggregate with:

```bash
uv run --no-project python scripts/summarize_comparison.py
```
