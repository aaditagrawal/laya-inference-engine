# AOTInductor deployment experiment

This builds an actual `.pt2` artifact containing compiled code, CUDA binaries,
and model weights. The request runtime loads tokenizer/config files and the
artifact; it does not construct `Agent` or load the original checkpoint weights.
Tokenizer initialization reads the serialized Rust tokenizer directly. Its
prepared requests matched AutoTokenizer on all 66 reference requests.
Retain the repository's licenses and upstream notices when using these experiments.

The tested runtime is PyTorch 2.14.0+cu132, SM120, Linux x86_64 with AVX2/FMA/F16C.
Each artifact supports one batch/sequence/options shape and one global-attention
mask specialization. Unsupported requests raise an error; there is no silent
fallback. Package creation and deployment startup are separate measurements.

## Build

Run with the existing native extensions already built:

```bash
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python \
  -m experiments.latency.aot.build --strict --packed-inputs \
  --case 1-short \
  --package .research/latency-optimizations/aot/one-short-packed.pt2 \
  --output results/latency-optimizations/aot/build-one-short-packed.json
```

Use `--case 1-long` and a different filename to build the 512-token artifact.
Strict export is required for the existing raw Triton calls. Exporting against
the actual packed input layout avoids a deployment-time alignment copy.
The package, its `.manifest.json`, and the adjacent `runtime/` directory belong
together. Each tested package is about 958 MB; weights are currently duplicated
between shape artifacts.

## Load a supported request

```python
from experiments.latency.aot.engine import PackageEngine
from laya_blackwell.workloads import workload

with PackageEngine(
    ".research/latency-optimizations/aot/one-short-packed.pt2",
    loader="checked-cpp",
) as engine:
    result = engine.predict(**workload(1, "short"))
```

`checked-cpp` calls the private C++ package loader after explicit checks of Torch
version, platform, CPU instructions, GPU architecture, and the pinned math
library. It avoids the high-level loader's C++ compiler feature probe. This API
is version pinned and must be reviewed when upgrading Torch. The `standard`
loader remains available for comparison.

This is not a Python-free executable. The runtime still registers the existing
Python LayerNorm/GELU custom operators and loads the prebuilt native norm
extension. CUDA graphs use the v2 adapter with separately owned capture streams.

## Reproduce measurements

`run_matrix.py` launches three fresh processes per mode and separately locks
each GPU phase. It records completed requests, including tokenization, copies,
inference, and formatting; HTTP and artifact creation are excluded. Use new
cache directories for each fresh run. The final controls resolve the original
checkpoint locally with Hub networking disabled. Use
cache directories without spaces so PyTorch's compiler feature probe can build:

```bash
uv run --no-sync python -m experiments.latency.aot.run_matrix \
  --package .research/latency-optimizations/aot/one-short-packed.pt2 \
  --output results/latency-optimizations/aot/my-matrix \
  --cache-root /tmp/laya-aot-my-matrix --runs 3 \
  --modes aot-fast native compile
```

`deploy.py` is an additional model-only diagnostic. Its warm timings exclude
request processing and must not be compared directly with full `predict`.
`validate.py` checks deterministic token and question-type perturbations within
the exported specialization. These checks establish implementation parity for
the tested tensors, not labeled model quality or arbitrary-shape support.
