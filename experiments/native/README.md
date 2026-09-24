# RTX 5070 Ti experiments

Opt-in implementation for the English Laya checkpoint on the RTX 5070 Ti,
SM120. The installed `laya_blackwell` engine and server retain their existing
defaults. Run these experiments from a repository checkout.

This combines native CUDA residual/LayerNorm fusion, a corrected Triton GEGLU,
windowed attention, C++ input packing and graph replay, and direct use of the
existing Rust Tokenizers backend. Python still loads the model, handles the
request protocol and manages the engine. This is not a full C++ or Rust rewrite.

The [experiment report](../../docs/native-optimizations.md) records which changes
helped, which failed numerical validation, and how the measurements were made.

## Build and use

Use the repository's locked environment. The tested stack is PyTorch
2.14.0+cu132, Triton 3.8.0, Transformers 5.17.0 and Laya 0.3.9. Native extensions
were compiled with CUDA Toolkit 13.1 and the installed C++ compiler. Build
artifacts go under the ignored `.research/` directory.

```bash
export CUDA_HOME=/usr/local/cuda-13.1
flock -s /tmp/laya-gpu-experiments.lock \
  uv run --no-sync python experiments/native/host/build.py
flock -s /tmp/laya-gpu-experiments.lock \
  uv run --no-sync python experiments/native/kernels/build.py --variant all
```

Run from the repository root:

```python
from experiments.native.engine import ExperimentalEngine

with ExperimentalEngine(mode="native-window") as engine:
    result = engine.predict(
        state="The customer was billed twice.",
        questions={"urgent": {"type": "noul", "instructions": "Is action urgent?"}},
    )
```

`native` enables the native kernels and host path. `native-window` also uses
the exact bidirectional window implementation. `compiled` adds full-graph
Inductor compilation with explicit precision preservation. New compiled shapes
have substantial first-use costs; select representative warmup requests for
your deployment.

`autotuned` tunes shared Linear kernels with Inductor's max-autotune mode,
including TMA candidates. It is numerically changed and stays opt-in. The
66-request screening matched all 208 selected decisions, with maximum
probability drift 0.00432 and action-logit drift 32. Action probabilities stayed
unchanged on the saturated fixture. Its first sixteen-long tuning pass took
89.82 seconds. The recorded winning kernels were ordinary Triton or ATen;
enabling TMA candidates does not establish a TMA speedup.

The compiled path pins the CUDA `libdevice` math library before deriving sparse
BF16 GELU corrections. It preserves intermediate rounding and uses opaque
operations for the remaining LayerNorm and small GELU modules. Changing the
math library after initialization invalidates those corrections. The native
window path uses a private ATen operator; both compiler and attention adapters
are restricted to the tested PyTorch 2.14 series.

## Reproduce the comparisons

The fixture contains 66 synthetic requests and 208 decisions. Reference logits
are stored in `results/native-optimizations/reference/`; they are implementation
parity evidence, not a labeled accuracy dataset. Regenerate after changing the
checkpoint, dependencies, production engine or fixture:

```bash
flock /tmp/laya-gpu-experiments.lock \
  uv run --no-sync python -m experiments.native.common
flock /tmp/laya-gpu-experiments.lock \
  uv run --no-sync python -m experiments.native.benchmark \
  --mode native-window --validate --output results/native-optimizations/rerun.json
flock /tmp/laya-gpu-experiments.lock \
  uv run --no-sync python -m experiments.native.benchmark \
  --mode compiled --validate --output results/native-optimizations/rerun-compiled.json
flock /tmp/laya-gpu-experiments.lock \
  uv run --no-sync python -m experiments.native.benchmark \
  --mode autotuned --validate --output results/native-optimizations/rerun-autotuned.json
flock /tmp/laya-gpu-experiments.lock \
  uv run --no-sync python -m experiments.native.compare_modes \
  --output results/native-optimizations/compiler/rerun-direct-modes.json
```

The runner keeps baseline and candidate resident, checks raw logits and action
outputs, and randomizes timed blocks over five rounds. Each block has five
warmups and 30 measured requests. Timed `predict` includes preparation,
transfers, inference and formatting. Model load, first-shape setup and HTTP are
excluded equally. The first-shape observations are reported separately and may
reuse disk caches. Throughput means completed decisions per second.

Native and full-model compiled comparisons require exact logits and actions.
The explicitly nonexact `autotuned` comparison instead requires identical
selected decisions and action argmax, plus at most 0.01 absolute drift in both
choice and action probabilities. Raw differences are always recorded; the
packaged run measured 0.00432 choice drift and zero action-probability drift.

For process startup:

```bash
flock /tmp/laya-gpu-experiments.lock \
  uv run --no-sync python -m experiments.native.startup \
  --mode compiled --kernel cuda_norm_triton_geglu_corrected \
  --cache-state reused --output results/native-optimizations/startup-rerun.json
```

Fresh cache measurements additionally require `--cache-state fresh` and explicit,
empty `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` directories. This reports
imports, model/adapter setup, first prediction and total process startup
separately. Model download and native extension builds are excluded.

Use `--cases 5-medium 16-medium 32-short 32-long --max-graphs 1` for larger and
irregular batches. The maximum batch allowed by the request protocol is 64;
available VRAM may impose a lower limit when several models and graphs are
resident. GPU timings and CPU-heavy extension builds share the advisory lock
shown above so they cannot overlap during these experiments.

The model and derived model code retain Apache-2.0 notices. Native normalization
also retains the PyTorch BSD license and attribution in `kernels/NOTICE` and
`kernels/pytorch-LICENSE`. Original contributions retain the repository's MIT
license.
