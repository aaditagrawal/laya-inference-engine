These experiments measure concurrent local inference calls on the existing
`native-window` model. Python still owns the API, SDK request construction,
formatting and scheduling. The existing C++ extension packs pinned buffers and
launches CUDA graphs; the existing Rust tokenizer backend prepares tokens. This
is not a full C++ or Rust server rewrite and does not measure HTTP transport.

`StreamService` keeps immutable model weights shared and gives each lane its own
graph, staging buffers, output arrays and CUDA replay stream. Caller threads
already overlap CPU request preparation with GPU execution in the baseline.
Multiple lanes additionally allow independent graphs to execute concurrently.
FIFO admission separates queueing from compute time. The optional `small_lanes`
partition gives one-question requests separate high-priority lanes; large
requests use the remaining lanes. That policy intentionally trades queueing
between request classes and is evaluated on mixed traffic.

`MicrobatchService` has one GPU worker, a bounded queue of 64 pending requests,
a configurable formation wait, at most eight requests and 64 question rows per
GPU batch. It groups the original graph key, preserves unmasked attention's
power-of-two row geometry, and slices copied outputs back to each caller's
original IDs. The formation wait is not a deadline: a busy GPU can cause longer
queueing. Changing batch geometry can change BF16 outputs, so microbatch results
have a separate numerical gate and must not be described as bit-exact.

The caller owns the model and must close every service before closing its base
engine. `close()` drains already accepted requests; new calls then fail. Result
arrays belong to the caller and survive later requests. A service's graph
capture gate drains its other lanes before a new shape is captured. Independent
services sharing a device are not coordinated by this gate and must not run
concurrently while either can capture a new graph.

`graph_adapter.py` fixes a lifetime issue exposed by extended cache churn. The
tested PyTorch build clears cached cuBLAS workspaces associated with a capture
stream when a graph is destroyed. Ordinary PyTorch streams come from a finite
reused pool, so separately live graphs can share that workspace identity. The
new adapter owns one raw CUDA capture stream per graph and destroys it only after
resetting the graph and releasing its slot buffers. Replay arithmetic and the
native launcher are unchanged. `replace_adapter(engine)` installs it before any
capture. Both serving candidates and the matched baseline use this fix.
The relevant lifetime cleanup is in the
[installed PyTorch revision's CUDA graph implementation](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CUDAGraph.cpp).

Run from the repository root with the existing uv environment and built round-one
extensions:

```sh
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.serving.graph_churn --cycles 12 --output results/latency-optimizations/serving/graph-churn-fixed.json
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.serving.run --task validate --variants streams-2 --output results/latency-optimizations/serving/streams-validation-fixed.json
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.serving.run --task validate --variants microbatch-0.25 --output results/latency-optimizations/serving/microbatch-validation-expanded.json
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.serving.run --task benchmark --rounds 1 --requests 48 --output results/latency-optimizations/serving/screen-fixed.json
flock -s /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.latency.serving.cpu_checks
```

The benchmark uses closed-loop clients at concurrency 1, 2, 4 and 8. Every
request's wall time includes tokenization, waiting, GPU execution and formatting;
throughput uses the complete measured client interval. First-use graph capture
and equal warmup traffic are excluded. Four visible input variants prevent an
unchanged-input replay check from being mistaken for input isolation. The mixed
profile contains equal numbers of 1/16-question short/long requests. These are
implementation-parity fixtures, not a labeled model accuracy or public network
service benchmark. Throughput gains are distinct from latency at concurrency 1.

The final three-round confirmation is in
`results/latency-optimizations/serving/confirmation-summary.json`, with individual requests
in `confirmation.json`. At concurrency four, four streams delivered 817.9
one-short requests/s versus 433.2 for serialized replay, and reduced queue-inclusive
p95 from 9.43 to 4.96 ms. At concurrency one, p50 increased from 2.25 to 2.30 ms.
The one-stream FIFO policy and both microbatch waits were also measured in the
initial fixed screen. Microbatching is rejected as a default: the expanded
near-tie gate retained only 231 of 245 unique decisions. Priority isolation is a
separate tradeoff, substantially reducing one-question mixed-traffic latency
while increasing bulk-request waiting.
