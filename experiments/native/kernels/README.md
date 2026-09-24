# Native kernel experiments

These adapters leave the production engine unchanged. The default research target
is RTX 5070 Ti's RTX5070Ti, SM120. FP32 residuals and the existing BF16 rounding steps
are preserved. See the local experiment report for measured alternatives and
rejected numerical changes.

Build outside timed measurements, using the shared lock to avoid disturbing
another agent's GPU measurements:

```bash
flock -s /tmp/laya-gpu-experiments.lock \
  uv run --no-sync python experiments/native/kernels/build.py --variant vector
```

The build uses the installed CUDA toolkit and PyTorch headers directly. It does
not require Ninja, alter the Python environment, or assume paths have no spaces.
Earlier controls and their raw measurements remain in the local research folder.

Install an adapter before warming or capturing graphs:

```python
from laya_blackwell.engine import BlackwellEngine, model_path
from experiments.native.kernels import install

engine = BlackwellEngine(model_path())
install(engine)
# engine.predict(state=..., questions=...)
engine.close()
```

The native norm supports contiguous FP32 rows of width1024, FP32 norm parameters,
and optional contiguous FP32/BF16 residuals. Its fixed reduction tree matches
PyTorch's Welford order. The final variant replaces inter-warp bookkeeping with
one barrier and uses aligned vector loads/stores.

The GEGLU adapter preserves BF16 GELU rounding before multiplying the gate. It
checks all65,536 BF16 encodings at setup and corrects the few differing erf
rounding boundaries, or uses the complete128KiB activation lookup table for
shapes where that is faster. These are activation values, not inference answers.
The default uses sparse corrections for every shape. Select
`cuda_vector_norm_adaptive_geglu` to try the measured lookup-table alternative.
The selected Triton libdevice is pinned before calibration; if configuring
Inductor to use a different library, select it before calling `install`.

Each adapter owns its forward-function globals. Baseline and experimental models
can coexist for randomized comparisons. The CUDA ops have fake registrations for
`torch.compile`. A compiler may still change other model arithmetic; validate the
complete compiled model before treating it as equivalent.

PyTorch's BSD license is retained in `pytorch-LICENSE`; see `NOTICE`.
