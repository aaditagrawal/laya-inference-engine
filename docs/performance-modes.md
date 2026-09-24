# Balanced and fast configurations

The measured implementations support a balanced configuration and a fast
configuration. The useful tradeoff is GPU memory and setup cost versus latency
for a specified workload. We have not measured FLOPs or joules per request, so
the results do not justify describing fast as using "much more compute."

These are recipes for the repository's experimental API. They are not new flags in
the published CLI, and they do not change its default backend.

## Balanced

Use the exact native kernels and window attention, with one direct request path.
This avoids the additional packed projection weights and per-lane graph buffers.
It is the sensible choice for a single caller, small requests, or limited VRAM.

```python
from experiments.latency.engine import V2Engine

with V2Engine(optimization="native") as engine:
    response = engine.predict(state=state, questions=questions)
```

One short warm request is approximately 2.2 ms on the tested RTX 5070 Ti.
"Balanced" here means the smaller resource footprint of the tested choices;
it is not a measured energy-efficiency or power-cap mode.

## Fast

Enable the measured projection fusions for larger requests:

```python
from experiments.latency.engine import V2Engine

with V2Engine(optimization="fused") as engine:
    response = engine.predict(state=state, questions=questions)
```

Packed weights use an additional 300,941,312 bytes, about 301 MB of GPU memory,
plus a shared 128 KiB GELU table. In the paired comparison, sixteen long
questions took 81.23 ms instead of 88.14 ms. The selector retains the existing
projection path where fusion did not help. It did not materially improve a
single short request. [Fusion results](../results/latency-optimizations/fusion/best.json).

For concurrent callers, independent stream lanes are another option:

```python
from experiments.latency.engine import V2Engine
from experiments.latency.serving.service import StreamService

with V2Engine(optimization="native") as engine:
    service = StreamService(engine.base, lanes=4)
    try:
        # Call service.predict from the application's concurrent request threads.
        response = service.predict(state=state, questions=questions)
    finally:
        service.close()
```

The model weights are shared, while each lane owns graph buffers and streams.
Additional GPU memory depends on the warmed shapes and graph-cache size.
Four callers with short requests reached 818 requests/s versus 433 for the
serialized GPU path; p95 latency fell from 9.43 to 4.96 ms. The scheduler made
a lone short request slightly slower. These are local service measurements,
without HTTP. [Serving results](../results/latency-optimizations/serving/confirmation-summary.json).

Fusion and concurrent streams passed correctness checks together, but their
combined performance has not been measured. Their independent gains must not
be multiplied. Additional lanes keep more work in flight; that does not imply
more arithmetic or energy for each completed request.

## Compilation and startup

Full-model compilation is a separate workload-specific choice. A direct paired
test reduced one short request from 2.189 to 2.102 ms, about 4%, but increased
sixteen-long latency from 88.036 to 91.701 ms. It adds compilation and
specialization costs, so it is not a universal fast setting.
[Direct comparison](../results/native-optimizations/compiler/direct-native-vs-compiled.json).

Precompiled packages move compilation into the build step. The final repeated
startup experiment measured 3.73 seconds to the first completed request,
including imports and setup, versus 6.33 seconds native and 20.64 seconds with
fresh compilation. Warm short requests took 2.12 ms. The two tested artifacts
support fixed batch-one shapes and each occupies approximately 958 MB on disk.
These packages improve deployment setup rather than providing a new large
single-request latency reduction. [Deployment details](../experiments/latency/aot/README.md).

A suitable product description is: "Fast uses additional GPU memory and may
require more preparation to reduce latency for larger or concurrent workloads."
Keep power use, GPU-seconds per request and peak VRAM separate in future resource
measurements. The current results do not establish a mode that spends far more
compute to halve the roughly 2 ms short-request latency.
