These are second-round experiments for Laya on the RTX 5070 Ti,
SM120. They use the locked environment and native extensions from
[round one](../native/README.md). Run from a repository checkout; the installed
engine and server retain their existing defaults.

The reusable engine preserves the existing batch/sequence padding, gives every
captured graph its own CUDA capture stream, and enables the measured projection
fusions for larger shapes. It retains the existing projections for small inputs
and for token-row counts above 8,192.
Packed projection weights add about 301 MB of GPU storage. For a workload made
entirely of small requests, `optimization="native"` avoids that allocation.

```python
from experiments.latency.engine import V2Engine

with V2Engine() as engine:
    response = engine.predict(
        state="The production service has been unavailable for three hours.",
        questions={
            "urgent": {"type": "noul", "instructions": "Is an urgent response needed?"}
        },
    )
```

Use `optimization="native"` to retain the round-one model kernels while keeping
the graph lifetime fix. `V2Engine` uses the same prepare, predict, run_prepared,
warmup, and close interface as the previous experimental engine. Extensions,
model download, graph capture, and Triton compilation can contribute to first
use. Warm measurements exclude those costs explicitly.

For concurrent requests, `StreamService` owns separate streams and buffers while
sharing immutable model weights. Close the service before its engine:

```python
from experiments.latency.serving.service import StreamService

with V2Engine() as engine:
    service = StreamService(engine.base, lanes=2)
    try:
        # Call service.predict from the application's request threads.
        response = service.predict(
            state="An invoice was duplicated.",
            questions={"refund": {"type": "noul", "instructions": "Is a refund needed?"}},
        )
    finally:
        service.close()
```

The example does not add an HTTP server. Queue-inclusive throughput tests use
local service calls. The service serializes capture against active replays;
warming the shapes used by a deployment avoids capture latency during traffic.
Independent services sharing a model must not capture concurrently.

[AOT deployment](aot/README.md) provides a separate fixed-shape package engine.
Its compiled artifacts embed model weights and currently duplicate those
weights between shapes. The checked C++ package loader still requires Python
and registered custom operators. It is not a standalone C++ executable.

The `padding/` and `MicrobatchService` implementations remain research options.
Changing matrix geometry can change close decisions, even when an initial
fixture passes. They are deliberately separate from `V2Engine`.

Reproduce GPU experiments under the shared lock:

```bash
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python \
  -m experiments.latency.fusion.benchmark \
  --config experiments/latency/fusion/best.json \
  --rounds 3 --repeats 30 --output results/latency-optimizations/fusion/rerun.json

flock /tmp/laya-gpu-experiments.lock uv run --no-sync python \
  -m experiments.latency.serving.graph_churn \
  --cycles 12 --output results/latency-optimizations/serving/churn-rerun.json
```

The CUDA math library, Torch 2.14 custom operators, SM120 target and BF16 rounding
rules are pinned. Updating them requires fresh numerical validation. These are
synthetic implementation-parity tests, not labeled task-accuracy measurements.
Retain the repository's MIT/Apache notices and the round-one PyTorch attribution.
