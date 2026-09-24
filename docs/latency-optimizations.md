Laya's second experiment round on RTX 5070 Ti found useful gains from projection
fusion, concurrent GPU streams and precompiled deployment. The reusable implementation is
[`V2Engine`](../experiments/latency/engine.py), with
[usage instructions](../experiments/latency/README.md). The code and measurements
are included in this repository as opt-in experiments. Production defaults are
unchanged.

These comparisons use the RTX 5070 Ti, 16 GB, SM120, with PyTorch 2.14.0+cu132,
Triton 3.8 and Transformers 5.17. The baseline is the previous optimized
`native-window` experimental engine, not upstream Laya. The checkpoint revision
is `5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`. Every GPU experiment acquired the
same exclusive lock. Native builds and heavy compilation did not overlap timed
inference. Results measure completed decisions, not generated-token TPS.

Projection fusion reduced median full-request time for 16 long questions from
88.1435 to 81.2315 ms, a 7.84% latency reduction. For 16 short questions it reduced
10.7907 to 10.4983 ms, 2.71%. Small-shape kernel improvements did not translate
into consistent request-level wins, so the final selector retains the previous
projections for small inputs. The results use three randomized blocks and
90 measured calls per implementation and workload, with both models resident.
Preparation, transfers, inference and response formatting are included; model
loading, first-shape setup and HTTP are excluded equally.
[Raw comparison](../results/latency-optimizations/fusion/best.json).

The implementation pairs Wi projection with the GEGLU epilogue and pairs QKV
projection with RoPE where measurements support doing so. It rounds projection
outputs to BF16 before the nonlinear operations. A 128 KiB lookup table preserves
the installed Torch GELU results. Packed Wi weights add 300,941,312 bytes of GPU
storage. The selector retains existing kernels above 8,192 token rows; the
32-question, 512-token fallback was checked separately. The kernels/configuration
are in [`fusion/`](../experiments/latency/fusion/).

Four CUDA streams improve concurrent request throughput while sharing model
weights and keeping graph buffers private. At four concurrent callers, one
short question reached 817.87 requests/s versus 433.23 for the previous serialized
GPU path, 1.888x throughput. Median request latency fell from 9.196 to 4.829 ms;
p95 fell from 9.433 to 4.958 ms. At eight callers, throughput was 766.45 versus
427.73 requests/s, with p95 12.670 versus 19.591 ms. For one long question at eight
callers, throughput rose from 130.77 to 202.16 requests/s and p95 fell from 65.211
to 41.323 ms. A single short request through the scheduler was slightly slower,
2.304 versus 2.249 ms. Use the direct engine for a lone caller.
[Confirmed serving results](../results/latency-optimizations/serving/confirmation-summary.json).

Serving measurements use fixed-count closed-loop clients at concurrency
1, 2, 4 and 8. Each configuration has three randomized blocks of 64 requests.
Latency includes preparation, queueing, transfers, GPU execution and formatting.
Throughput uses total client wall time. All 12,288 measured requests preserved
their formatted responses and incurred zero graph misses after warmup. The
original baseline already overlaps host preparation with serialized GPU replay.
These measurements use local Python service calls, without HTTP on either side.
Large batches gain much less from overlap because one request already uses more
of the GPU. Priority lanes improve small-request latency in mixed traffic while
increasing bulk-request tail latency, so they remain an explicit policy choice.

Ahead-of-time compilation reduced median time from module entry to the first
completed short request to 3.730 seconds, versus 6.326 seconds for native,
20.638 seconds for fresh `torch.compile`, and 8.668 seconds with reused compiler
caches. Each condition ran in three fresh processes, fully offline, with warm
OS file caches and existing model files and native extensions. Imports, loading,
graph capture, preparation, inference and formatting are included; artifact
building and HTTP are excluded. Warm request medians were 2.116 ms for AOT,
2.223 ms for native and 2.110 ms for fresh-compiled inference.
[Final startup matrix](../results/latency-optimizations/aot/final-offline/matrix-summary.json).

The deployment bundle uses strict `torch.export` and AOTInductor, the C++ package
loader with explicit compatibility checks, and direct initialization of the
existing Rust tokenizer. The measured gain includes these loader and tokenizer
changes; it is not an isolated AOT compiler effect. Both compiler caches remained
empty after deployment. Model math matched exactly across 48 tensor-value cases,
24 at each supported shape, and tokenizer preparation matched all 66 reference
requests. Short and long full-request checks also passed. These are separate
validation scopes, not a claim that AOT ran the entire 66-request inference suite.
[AOT evidence and provenance](../results/latency-optimizations/aot/index.json).

Only two batch-one, four-option shapes are built: 64 and 512 tokens, both with
unmasked global attention. Unsupported shapes or mask modes are rejected.
Each artifact is about 958 MB and embeds its own copy of the model weights.
Observed export took 5–6 seconds and package compilation took 24–26 seconds,
using existing development caches. The runtime still needs Python Torch,
registered custom operations, prebuilt native extensions and pinned private
Torch APIs. It is not a standalone C++ executable.
[AOT implementation and usage](../experiments/latency/aot/README.md).

Reducing padding produced large speedups but did not pass the expanded numerical
gate. Exact batch sizing cut 17 medium questions from 87.290 to 45.150 ms and
17 long questions from 180.087 to 94.072 ms. It initially matched all 208 fixture
decisions. Expanding every qualifying four-question reference request into
5 and 17 questions exposed 12 changes across 946 comparisons with the padded
baseline, with maximum probability drift 0.034469. Preserving the baseline's
global-attention mask mode reduced this to five changes and maximum drift
0.014596, still outside the gate. Sequence buckets spaced 32 tokens apart also
failed the 0.01 probability bound, reaching 0.013702 despite matching all
208 selected decisions. These policies remain research options.
[Padding results](../results/latency-optimizations/padding/).

For context, the naive exact-batch result matched the cached original SDK
four-question decisions in all 946 repeated comparisons. That cached reference
was not a fresh SDK run at each expanded batch size. Neither result establishes
which variant is more accurate on labeled tasks. The finding is that batch
geometry and attention dispatch can alter close decisions, making it unsafe to
describe these changes as numerically transparent.

Microbatching also passed the first 208-decision check but changed 14 of
245 decisions in the expanded, deduplicated comparison. Maximum probability
drift was 0.019893. Action probabilities stayed unchanged on this highly
saturated fixture. It is excluded from the recommended engine despite higher
throughput in the initial screen.
[Expanded microbatch evidence](../results/latency-optimizations/serving/microbatch-summary.json).

Denser shapes require more graph-cache entries. Cycling through 1 to 10 short
questions with an eight-entry cache caused zero misses for the original buckets
and 60 misses in 60 measured calls for exact batches. Median latency rose from
6.181 to 52.826 ms once recapture was included. The warm speedup assumes a shape
working set that fits the cache.
[Cache experiment](../results/latency-optimizations/padding/cache-churn-fixed.json).

The stress test also found and fixed a CUDA graph lifetime failure. In the
installed Torch revision, destroying a graph clears cuBLAS workspaces for its
capture stream. Ordinary Torch streams come from a reused pool, so different
live graphs could share that stream identity. Destroying one graph could then
invalidate another graph's workspace. The v2 adapter gives each graph a uniquely
owned raw CUDA capture stream and destroys resources in graph, buffers, stream
order. This explanation follows the pinned
[Torch graph cleanup implementation](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CUDAGraph.cpp#L348).
The fix passed 120 recaptures/evictions plus 120 retained-graph replay checks,
with exact outputs and zero model/graph allocations after cleanup.
[Stress test](../results/latency-optimizations/serving/graph-churn-fixed.json).

The final combined engine passed exact choice and action logits for 66 requests
and 208 decisions, 102 benchmark/replay probe decisions, 32 fallback decisions,
and 39 concurrent decisions across four actual lanes. Outputs remained owned
after replay and close; closing a service left its caller-owned model usable.
Only the intentional 128 KiB process-level GELU table remained after model and
graph cleanup. These are synthetic implementation-parity tests, not a labeled
quality benchmark or a guarantee for every possible input.
[Integrated verification](../results/latency-optimizations/integration-four-lanes.json).

These results isolate software changes on this GPU. They do not measure a
Blackwell-versus-earlier-generation hardware contribution or certify performance
on other Blackwell GPUs. The accepted kernels use BF16 Tensor Core operations
and preserve existing rounding; reduced-precision variants from the first round
remain excluded.

Fusion and serving gains were benchmarked separately. The combined engine passed
correctness checks, but multiplying the independent speedups would not establish
its combined performance. The consolidated [summary](../results/latency-optimizations/summary.json)
and [SHA256 manifest](../results/latency-optimizations/manifest.json) retain the final evidence.
