# Balanced and fast modes

`balanced` is the default public mode. `fast` packages the retained BF16
optimizations for the pinned Laya checkpoint on SM120. Both expose the same
request and response API.

| | Balanced | Fast |
| --- | --- | --- |
| Python | `BlackwellEngine()` or `create_engine(mode="balanced")` | `FastEngine()` or `create_engine(mode="fast")` |
| CLI | Default, or `--mode balanced` | `--mode fast` |
| Native build | No separate build | Run `laya-blackwell build-fast` |
| Additional tables | No fast token tables | 491.9 MiB of token-local projections |
| Short request execution | BF16 with CUDA Graphs | Tuned BF16 kernels, native attention and host code, short-shape compilation |
| Longer requests and batches | Existing BF16 path | Native path; the short-shape compiler is not applied |
| Model support | Existing engine options | Pinned root English checkpoint only |

Fast uses extra GPU memory and preparation to lower warm latency. FLOPs, power
and energy per completed request have not been measured; "much more compute"
is not an established tradeoff. Neither mode caches answers or prepared requests.

## Setup and use

Balanced mode needs the locked Python environment and a compatible Blackwell
GPU. Fast mode additionally needs Linux x86_64, the `fast` extra, a C++ compiler
and a CUDA 13.x toolkit at least as new as 13.1 with `nvcc`. CUDA 13.1 is the
validated local toolkit; the toolkit major must match the pinned Torch runtime.

```bash
uv sync --extra fast --locked
uv run laya-blackwell build-fast
uv run laya-blackwell predict --mode fast --request examples/request.json
uv run laya-blackwell serve --mode fast
```

The builder accepts `--cuda-home`, `--cutlass`, `--flash-attention`,
`--cache-dir` and `--offline`. It fetches the pinned CUTLASS and FlashAttention
header checkouts when they are absent. `--offline` requires cached or supplied
clean checkouts at the exact pinned revisions. A verified existing build can
be reused without a compiler, network access or those checkouts. Import and
inference do not download headers or compile native extensions.

The default build cache is `$XDG_CACHE_HOME/laya-blackwell/fast`, using
`~/.cache` when `XDG_CACHE_HOME` is unset. Use `LAYA_FAST_CACHE` during inference
if the build uses a custom cache directory:

```bash
uv run laya-blackwell build-fast --cache-dir /path/to/laya-fast-cache
LAYA_FAST_CACHE=/path/to/laya-fast-cache uv run laya-blackwell serve --mode fast
```

Fast mode checks the model, revision, device and validated software stack.
It supports `convaiinnovations/laya` at revision
`5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`, without a subfolder or local-model
override, with the `fused` BF16 backend. It does not expose FP8 or FP4 policies.
The pinned environment uses Python 3.12, Torch 2.14.0+cu132, Triton 3.8.0,
Transformers 5.17.0 and NumPy 2.5.3. The native response formatter depends on
NumPy's private FP32 loop interface, so the NumPy pin is part of correctness.
The build manifest also identifies the exact Torch build and Python ABI.
Changing these dependencies requires rebuilding and revalidation.

```python
from laya_blackwell import create_engine

with create_engine(mode="fast", device="cuda:0") as engine:
    engine.warmup(state, questions)
    response = engine.predict(state=state, questions=questions)
```

Both modes support `predict`, `system_one`, `warmup`, `close` and context
management. They accept the same device and graph-cache limits. An engine
protects reusable buffers by serializing requests; callers receive fresh output
copies. Neither mode adds a concurrent GPU scheduler.

Building native extensions does not eliminate first-use setup. The engine
prepares token tables and GELU correction data, compiles the short 64-row shape,
and captures CUDA Graphs. Warm the shapes used by your application before
measuring steady-state latency. New or evicted graph configurations need setup
again. Keep the process alive to reuse the work.

## What the latency numbers compare

The packaged modes were measured together over nine randomized paired rounds,
with 100 requests per mode per round. Full warm p50 results were:

| Workload | Balanced | Fast | Latency reduction |
| --- | ---: | ---: | ---: |
| One short question | 2.804 ms | 1.610 ms | 42.6% |
| One long question | 8.616 ms | 6.968 ms | 19.1% |
| Sixteen short questions | 12.591 ms | 10.553 ms | 16.2% |

These include tokenization, transfers, inference and formatting and exclude
loading, compilation, first capture and HTTP. Cycling 128 short fixtures over
nine rounds measured 1.636 ms fast versus 2.784 ms balanced. Sub-millisecond
full requests were not achieved. [Paired package results](../results/fast-package.json)

The same run measured 1.612 ms for the retained research implementation beside
fast mode's 1.610 ms. Packaging preserved its performance; the tiny difference
does not establish a new speedup. Prepared inputs, raw choice/action logits and
public responses, excluding runtime metrics, matched on 199 requests with 354
decisions. Another 52 concurrent calls, graph eviction, output ownership and
close behavior passed. This is implementation-parity evidence, not labeled
task-accuracy evidence.

An earlier comparison measured the retained implementation at 1.624 ms against
an optimized native baseline at 2.197 ms. That baseline was not balanced mode.
Its 26.1% reduction remains a separate pre-extraction measurement.
[Earlier comparison](../results/frontier/summary.json),
[measurement guide](performance.md)

Upstream `Agent(fast=True)` is the Laya SDK's optional acceleration mode. It is
separate from this repository's `FastEngine` and CLI `--mode fast`.

## Separate research results

The following measurements retain their original experimental APIs and scopes.
They are not additional flags in the public fast mode.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/experimental-gains-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/experimental-gains-light.png">
  <img alt="Separate RTX 5070 Ti experiments: projection fusion reduces sixteen-long-question p50 from 88.14 to 81.23 ms; four CUDA streams raise four-caller short-request throughput from 433 to 818 requests per second. Gains measured separately without HTTP or startup." src="assets/experimental-gains-light.png" width="1000" loading="lazy">
</picture>

Projection fusion reduced sixteen-long latency from 88.14 to 81.23 ms with
about 301 MB of extra packed weights and a 128 KiB GELU table. It did not
materially improve a single short request. The retained fast MLP kernel uses
the original weight layout, so that packed copy is not part of fast mode.
[Fusion experiment](../experiments/latency/fusion/README.md),
[measurements](../results/latency-optimizations/fusion/best.json)

Four CUDA stream lanes reached 818 requests/s versus 433 for serialized GPU
execution with four short-request callers. p95 fell from 9.43 to 4.96 ms.
Weights were shared, while each lane needed its own graph buffers and streams.
One caller became slightly slower. This was a local service test without HTTP.
The streams and fusion speedups were measured separately and must not be
multiplied. [Serving experiment](../experiments/latency/serving/README.md),
[measurements](../results/latency-optimizations/serving/confirmation-summary.json)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/startup-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/startup-light.png">
  <img alt="Separate RTX 5070 Ti startup experiment from module entry through first response: fresh torch.compile 20.64 seconds, cached torch.compile 8.67 seconds, native 6.33 seconds, prebuilt AOT 3.73 seconds. Median of three processes with warm OS caches. These are not FastEngine startup measurements." src="assets/startup-light.png" width="1000" loading="lazy">
</picture>

Prebuilt AOT deployment measured 3.73 seconds from module entry to the first
response, versus 6.33 seconds native and 20.64 seconds with fresh compilation.
The three-process medians include imports and model loading, use warm OS file
caches, and exclude artifact build, download and HTTP. The AOT path changes
loader and tokenizer initialization, supports two fixed batch-one shapes, and
needs about 958 MB per artifact. These are separate deployment modes, not
startup measurements of the current FastEngine.
[AOT scope and reproduction](../experiments/latency/aot/README.md),
[startup matrix](../results/latency-optimizations/aot/final-offline/matrix-summary.json)
