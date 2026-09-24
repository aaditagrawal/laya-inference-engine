This archive records the attempt to reach less than 1 ms for a complete warm,
single-question request on the RTX 5070 Ti, SM120. The target was not reached.
The best retained
implementation takes **1.624 ms**, compared with **2.197 ms** for its paired
native baseline, a 26.1% latency reduction.

The accepted configuration is now packaged as
[`FastEngine`](../../src/laya_blackwell/fast/engine.py). Use the public
[setup and mode guide](../../docs/performance-modes.md) to run it. This
directory preserves the original probes and rejected alternatives; it is not
a runtime dependency of the installed package.

The measurement includes tokenization, host packing, transfers, inference,
synchronization and response formatting. It excludes loading, compilation,
first graph capture and HTTP. The original short workload has 64 input tokens,
one question and four options. No answers or prepared requests are cached.
Weights and executable CUDA Graphs stay resident.

| Experiment | Full request median | Result |
| --- | ---: | --- |
| Paired native baseline for latest implementation | 2.197 ms | Reference |
| Tuned BF16 projections | 2.061 ms | Exact on both parity suites |
| Tuned BF16 plus compilation for all shapes | 1.971 ms | Exact on both suites; long input slowed about 2% |
| Tuned BF16 plus compilation for the short shape | 1.973 ms | Previous best; other shapes use native execution |
| Smaller native attention query tile | 1.953 ms | Exact on both parity suites |
| Native attention plus selected TMA projections | 1.900 ms | Exact on both parity suites |
| Native attention, TMA QKV and deeper output-projection pipeline | 1.890 ms | Exact on both parity suites |
| Above plus cuBLAS-compatible BF16 partial-sum rounding | 1.860 ms | Exact on both parity suites |
| Above plus reduction/normalization fusion | 1.851 ms | Exact on both parity suites |
| Above plus precomputed token-local projection | 1.829 ms | Exact on both parity suites; adds 492 MiB of GPU tables |
| Above plus fused MLP input projection and GEGLU | 1.782 ms | Exact on both parity suites; packed prototype adds 287 MiB |
| Direct-weight MLP fusion plus selected head kernels | 1.795 ms | Exact on both parity suites; avoids the 287 MiB copy |
| Above plus batched offset-free tokenization | 1.779 ms | Exact on both parity suites |
| Above plus fixed-shape local attention | 1.731 ms | Exact on both parity suites |
| Above plus global attention and host replay changes | 1.655 ms | Exact on both parity suites |
| Above plus native C++ response formatting | 1.624 ms | Retained; exact logits and public responses on both suites |
| Custom Triton attention | 1.847 ms | Rejected; 207/208 original decisions agreed |
| More aggressive BF16 split reductions | 2.018 ms | Rejected after an unseen request changed decision |
| Explicit cuBLASLt algorithm selection | 2.109 ms | Passed original tolerance checks; slower than exact alternative |
| Native Blackwell MXFP8 for all encoder projections | 1.772 ms | Rejected; 207/208 original decisions agreed |
| Native Blackwell MXFP8 for MLP projections | 1.895 ms | Rejected; 207/208 original decisions agreed |
| Weight-only FP8 with BF16 activations | 1.849 ms | Rejected; 207/208 original decisions agreed |
| Native FP4 encoder projections | 1.394 ms | Rejected; 206/208 original and 119/128 additional decisions agreed |

Each full-request result pools all 250 measurements from five randomized blocks.
Each candidate has its own resident, interleaved baseline in the raw report.
Values above are separate comparisons and should not be interpreted as one
simultaneous ranking to microsecond precision.

The retained implementation and the uncompiled exact variant produced identical
raw choice and action logits on 66 original requests with 208 decisions and on
128 additional single-question requests. All 128 additional requests exercise
the optimized 64-row shape. The previous compiled implementation measured
1.941 ms versus its paired 2.175 ms baseline when cycling those changing
requests; the TMA/native-attention variant measured 1.877 ms versus 2.182 ms.
The deeper pipeline variant measured 1.863 ms versus 2.178 ms on that input set.
Adding compatible MLP partial-sum rounding measured 1.820 ms versus 2.177 ms.
Fusing reduction with normalization measured 1.800 ms versus 2.170 ms.
Precomputing the token-local projection measured 1.788 ms versus 2.175 ms.
The MLP/head/host combination measured 1.761 ms versus 2.191 ms.
Adding fixed-shape local attention measured 1.726 ms versus 2.173 ms.
Global attention plus the host replay change measured 1.677 ms versus 2.193 ms.
Native formatting measured 1.647 ms versus 2.191 ms on changing inputs.
This is a different input set from the main table. These are synthetic implementation
parity checks, not labeled task-accuracy evidence or a guarantee for every input.

The broader check matters. The 2.018 ms variant passed the original suite but
changed one of the additional 128 decisions, with a maximum probability error
of 0.03474. The retained version had zero probability error in both suites.

The C++ formatter uses the installed NumPy FP32 arithmetic loops and reductions
and Python rounding. Direct C library math changed results and was rejected.
All 194 original and additional requests produced exactly equal public response
dictionaries, excluding runtime metrics, and unchanged raw logits. A separate
12,103-case CPU check covers rounding boundaries, error behavior and underflow
warnings; five callback checks cover underflow combined with fallback. Ordinary
finite contiguous FP32 inputs use the native path, with conservative guards
that delegate unsupported cases to the original formatter. The implementation
depends on a private NumPy loop interface and is pinned to NumPy 2.5.3.

Formatting alone measured 28.3 versus 9.0 microseconds. In a separate nine-round
paired comparison using the same resident model and graphs, full short requests
fell from 1.642 to 1.609 ms, winning all nine rounds. Changing-input requests
fell from 1.673 to 1.641 ms, winning eight rounds. These paired results isolate
the formatter's contribution; the standard comparison in the table reports
1.624 ms. See `native-format.json`, `native-format-full.json` and
`native-format-integration.json` for validation, raw samples and source hashes.

A longer comparison confirmed the native attention change separately: 1.937 ms
versus 1.988 ms for the previous best, with 900 measurements per variant. It won
all nine randomized rounds on the fixed request and all nine on changing
inputs. The corresponding changing-input medians were 1.929 and 1.970 ms, with
1,152 measurements per variant. See `paired-native-attention.json`; this is a
separate measurement series from the main table.

The subsequent output-projection pipeline change had a smaller, less consistent
effect: 1.894 versus 1.901 ms in a direct comparison, winning six of nine fixed
request rounds. It won eight of nine changing-input rounds (1.870 versus
1.887 ms). `paired-pipeline.json` records the configurations and all samples.
The approximately 1.9 ms result is more representative than a microsecond-level
ranking of these two variants.

The native attention kernel instantiates PyTorch's own CUTLASS implementation
with a 32-row query tile. It preserves the baseline softmax and accumulation
order while launching more work for the small input. The faster custom Triton
version changed a close decision and was rejected. Further softmax-order
experiments, cuDNN attention and the current FlashAttention CuTe SM120 path
also failed elementwise attention parity. Their isolated-kernel timings are
not full-request claims. FlashAttention revision
[`eed1971`](https://github.com/Dao-AILab/flash-attention/tree/eed1971f5132630dc296fe37601e834d4b57a248)
was tested against the same Torch CUDA 13.2 environment as the baseline.

Splitting attention into 16-row or 8-row query blocks increased the number of
CTAs but slowed the measured attention banks. A 32-row control kept the same
arithmetic and CTA count while simplifying indexing. It saved about 2–3
microseconds in isolation, but complete requests were effectively unchanged:
1.621630 versus 1.621594 ms on the fixed input, faster in six of eleven rounds;
changing inputs measured 1.639445 versus 1.638995 ms, faster in seven rounds.
All 197 requests and 354 decisions matched exactly. The raw attention gate
also matched 2,628 local and 1,440 padded-global calls across 146 eligible
requests. The control was not retained. See `query-shard.json` and
`query-shard-full.json`; these are separate comparisons from the main table.

The matrix follow-up screened 312 TMA configurations and another 180 BF16
pipeline configurations. TMA improved the QKV and attention output projections;
a five-stage ordinary BF16 kernel then improved the output projection further.
The selected MLP input kernel stayed unchanged. Generated PTX confirms
`cp.async.bulk.tensor.2d` for the TMA candidates. These mechanisms and the smaller
attention tile are not exclusive to Blackwell; the measured choices are tuned
for this SM120 device. The TMA wrapper keeps host descriptor construction
outside Torch compilation and captures its GPU operations in the request graph.

A hardware check distinguished block clusters from TMA multicast. This GPU
successfully launches clusters of 1, 2, 4 and 8 blocks and passes cross-block
shared-memory reads. However, its SM120 compiler lowers multicast TMA to an
indirect CUDA syscall; ordinary TMA emits a native `UTMALDG` instruction.
[NVIDIA's GeForce CUTLASS example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
explicitly excludes TMA multicast on this architecture. Therefore the proposed
hardware-multicast GEMM was not implemented or timed. See
`cluster-gemm-capability.json` for runtime checks, compiler output and hashes.

The dedicated Blackwell Decompression Engine is also unavailable here. Both
the CUDA decompression algorithm mask and maximum-length attributes return
zero on this RTX 5070 Ti, with successful driver queries. NVIDIA's
[nvCOMP hardware FAQ](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html)
lists B200, B300, GB200 and GB300 support. `decompress-capability.json` records
the local query; this does not rule out software decompression on the SMs.

The MLP output projection needed a different numerical treatment. Profiling
showed that cuBLAS split its reduction across four blocks. Its BF16 partial
sums round differently from the FP32 partials in the initial Triton experiment.
Matching the partition and intermediate rounding produced exact output in
12 of 48 screened configurations. The selected kernel accelerated this
projection by about 14% in isolation; the full model then passed both parity
suites with zero choice/action-logit error. This matches the installed
baseline's [reduced-precision reduction behavior](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms),
and must be revalidated if its CUDA libraries or precision settings change.
The direct follow-up comparison measured 1.852 versus 1.890 ms on the fixed
request and 1.840 versus 1.883 ms on changing inputs. The new reduction won
all nine rounds in each case (`paired-splitk.json`).

The retained variant then fuses that four-part reduction with the following
FP32 residual addition and exact Welford normalization. It preserves the BF16
rounding between reduction and residual addition. The final encoder layer
keeps its ordinary reduction because its next consumer is different. Both
parity suites remained bitwise exact. A direct comparison measured 1.835 versus
1.849 ms, winning all nine fixed-input rounds; changing-input medians were
1.809 versus 1.832 ms, winning eight of nine rounds (`paired-reduce-norm.json`).
This is a small incremental gain, not a route to halving latency by itself.

A later TMA screen preserved the same 704/704/704/512 split boundaries and
BF16 partial rounding while changing load schedules and tile sizes. Of 96
configurations, 94 produced exact partials across all 28 MLP output matrices;
two exceeded shared-memory limits. The best isolated result improved from
0.2521 to 0.2494 ms. Full-request validation passed 197 requests exactly, but
fixed requests slowed from 1.648 to 1.660 ms and changing inputs from 1.682
to 1.685 ms. It was not retained. See `split-tma-summary.json`; the isolated
projection result excludes the following normalization in both variants.

Removing the three zero-filled iterations at the end of the fourth split
also failed to help. Nine exact variants changed the loop bound or CTA order;
the retained 28-matrix bank took 0.2522 ms, the untrimmed clone 0.2531 ms,
and the best trimmed variant 0.2533 ms. A separately compiled tail branch
doubled shared-memory allocation and took about 0.289 ms. All 252 matrix
comparisons and 36 signed-zero/sparse/small-value cases matched BF16 output
bits. No full-request variant was run. See `split-tail-summary.json`.

The next exact change precomputes the normalized embedding and first QKV
projection for every vocabulary token. This work depends on the frozen weights
and token ID, before positional rotation or attention mixes information across
tokens. It does not cache requests or answers. Tables are generated with the
same 64-row matrix geometry and arithmetic as the request path; other shapes
keep their original execution. The 50,368-row tables require 515,768,320 bytes
(491.9 MiB) and took 0.175 seconds to construct in the recorded run, separate
from loading and compilation. All 336 decisions remained bitwise exact.
Every vocabulary row also passed exact embedding/QKV comparison under four
randomized token-position permutations (`token-table-parity.json`).

The direct comparison against reduction/normalization fusion measured 1.828
versus 1.837 ms on the fixed request, winning seven of nine rounds. Changing
inputs measured 1.800 versus 1.811 ms, winning eight of nine rounds
(`paired-token-tables.json`). This gain is small for the extra memory;
At that stage, `token_tables=False` retains the previous implementation. The
later MLP-fusion adapter requires the token-table forward to be enabled.

The next retained kernel fuses the MLP input projection with GEGLU, preserving
the projection's BF16 rounding, exact GELU correction and the rounded product.
All 108 packed-weight configurations matched the retained matrix outputs across
28 distinct layer weights and independent random activations. The best isolated
configuration saved about 26 microseconds across those 28 projection/activation
pairs. The full packed-weight prototype passed both parity suites. In a direct
comparison, it measured 1.791 versus 1.819 ms for the fixed request and 1.782
versus 1.805 ms for changing inputs, winning all nine rounds in each case.
See `mlp-geglu.json` and `paired-mlp-geglu.json`.

A direct-weight variant calculates the paired row addresses without copying
the weights. All 12 screened configurations were exact. Its complete requests
were effectively tied with packed weights: 1.7833 versus 1.7832 ms on the fixed
request, with no consistent changing-input difference. The retained constructor
uses `mlp_geglu_unpacked=True` to avoid the packed prototype's additional
300,941,312 bytes, or 287 MiB. See `mlp-geglu-unpacked.json` and
`paired-mlp-geglu-unpacked.json`. Neither layout caches requests or answers.

A broader direct-weight screen tested 192 exact configurations, including
separate activation/gate accumulators, two or four warps, reduction tiles of
32/64/128 and weight-cache hints. The retained 32x64x64 tile remained fastest
at about 0.446 ms across the 28 projection/GEGLU pairs. Separate accumulators
were slower; cache hints did not improve the retained tile.
`mlp-schedule.json` contains this negative result. These are isolated kernel
timings and do not change the engine configuration.

Changing only weight-row pitch also failed to improve those kernels. Seven
paddings from 0 to 256 BF16 elements were checked across all four projection
families and 28 weights per family. All 784 matrix comparisons were exact.
No nonzero padding beat both the original kernel and its stride-aware
zero-padding control. Some smaller paddings made performance considerably
worse. `weight_pitch-result.json` records the separate controls and results.

Relocating all 136 BF16 parameters into execution-ordered banks also failed to
improve complete requests. A contiguous PyTorch bank took 1.6136 ms versus its
1.6110 ms resident baseline; ordinary CUDA VMM took 1.6149 versus 1.6106 ms;
aligning each parameter to 2 MiB in VMM took 1.6167 versus 1.6154 ms. None showed
a useful changing-input improvement. All stored weight bits and all 197
requests matched exactly for each layout. The aligned allocation required
849 MB versus 740 MB for ordinary VMM. Both reported VMM allocation
granularities were 2 MiB; this does not establish physical page size or TLB
behavior. Compiled method references outlived model close, so dedicated worker
processes retained each VMM mapping until process exit. No explicit unmap ran
while references remained. `weight_layout-result.json` links raw timings,
parameter offsets, and the preserved measured-source provenance.

A separate streaming diagnostic traced all 123 dense projections executed by
the short request, including fused MLP input and packed head projections.
Their matrix weights total 732,959,744 bytes. Reading each matrix once and
writing integer checksums took 0.9881 ms with original allocations and 0.9863 ms
with contiguous slices, using 123 kernels in execution order. One kernel over
the same contiguous bytes took 0.8629 ms. All block checksums matched independent
CPU references before and after nine randomized rounds. The bank is 14.56 times
larger than L2. These controls exclude inference arithmetic and activation
traffic; they are not full-request results or rigorous latency floors. The
123.4 microsecond paired difference also includes grid scheduling and cannot
be attributed entirely to launch overhead. See `weight-stream-summary.json`
and `weight-stream.json`.

An audit of ten existing profiler replays found one 1,072-byte input transfer
and two output transfers of 16 and 8 bytes per short request. Their summed
instrumented durations had a median of 1.648 microseconds. These durations
exclude surrounding delays and do not predict the gain from replacing copies
with mapped memory. `io-trace-audit.json` preserves individual events and the
source trace hash. This audit did not execute or time another inference run.

Replacing the two output copies with one kernel that writes mapped pinned
memory saved 0.580 microseconds in an isolated short-request I/O control,
winning all eleven rounds. Mapping the input transfer too did not improve
the short control. Tests poisoned buffers before every mode and compared with
independent CPU expectations, preventing an earlier replay from hiding stale
results. Fifteen close/recreate checks also passed. An initial teardown failure
and the strengthened validation are documented in `mapped-io-development.json`.

The output-only adapter then passed all 197 full-model requests and 354
decisions exactly, plus owned-output, eviction, close/recreate and 44 concurrent
call checks. Complete fixed requests were effectively tied: 1.608218 ms retained
versus 1.608399 ms mapped, faster in six of eleven rounds. Changing inputs
improved from 1.641779 to 1.637164 ms, faster in ten rounds, with a median paired
saving of 3.040 microseconds. Long requests won only five rounds and batches
won seven. This small changing-input benefit did not improve the fixed target;
the adapter remains a separate experiment. These paired measurements do not
replace the canonical comparison at the top of this file. See
`mapped-io-poison.json`, `mapped-io-lifetime-poison.json`, `mapped-io-full.json`
and `mapped-io-review.json` for the checks, raw samples and source hashes.

An activation audit checked 426,358,016 valid-token GEGLU values from 146 short
requests. Only 52,561 values were exactly zero, or 0.0123%. None of the checked
16x64, 32x64 or 64x64 tiles were entirely zero, including the padded rows
actually computed by the kernels. Only six of 106,589,504 consecutive four-value
groups contained at least two zeros. These inputs offer no useful exact sparse
path for the MLP output projection. No thresholds or approximate pruning were
used. `activation-sparsity.json` records the per-layer and per-request counts.

A PTX compile/disassembly audit confirmed native 256-bit global loads and
stores on SM120, but the assembler rejected 256-bit shared vectors and
32-byte per-thread asynchronous copies. The retained MLP kernel already uses
six 128-bit asynchronous global-to-shared copies per K64 iteration. The
explicit wider-load probe emits one 256-bit register load and two 128-bit
shared stores, replacing two copy instructions with three instructions and
adding register dependencies. This static evidence did not justify a new
matrix implementation; it does not measure a runtime bottleneck.
See `wide-load-audit.json` for all eight compile cases and SASS hashes.

A compact GELU table preserves every BF16 encoding, including subnormals,
signed zeros and all Inf/NaN cases, using 5,788 bytes. Exact middle ranges
use lookup; proven tiny-value and finite-tail ranges use half, identity or
signed zero. Despite that exhaustive equality, the fused 28-matrix bank took
0.4567 ms, versus 0.4455 ms for retained erf plus corrections and 0.4458 ms
for the full lookup table. It lost all nine paired rounds and was not
integrated. See `compact_gelu-domain.json` and `compact_gelu-result.json`.

An explicit packed BF16 product emits `HMUL2.BF16_V2` instead of expanding the
final GELU and gate operands to FP32. It matched 2,097,152 operand pairs and
84 real layer outputs bit for bit. The complete pair domain was not exhausted.
The 28-layer bank measured 0.44540 ms retained versus 0.44518 ms packed, only
0.225 microseconds apart. Both kernels use 60 registers and 24 KiB of shared
memory. This is effectively tied and remains experimental; see
`packed_bf16-result.json` for all nine timing rounds and machine-code evidence.

Native CUTLASS multistage BF16 kernels with a fused GEGLU epilogue also lost to
the retained Triton kernel. All 12 native tile/stage configurations matched
raw and fused outputs across all 28 weights. A matched-output comparison of
the best native tile measured 0.4536 ms with the exact GELU table and 0.4767 ms
with corrected native erf, versus 0.4447 ms retained. Both variants lost all
nine paired rounds. The corrected erf expression matched all 65,536 BF16
encodings, and both native epilogues matched three changing activation seeds
for every weight. These measurements exclude packing/setup and are isolated
weight-bank timings. No full-request variant was run. See
`cutlass-geglu-summary.json`; adapted CUTLASS notices remain alongside the code.

A further native screen tried irregular output widths to reduce partial waves
of CTAs. The default CUTLASS epilogue failed coverage assertions for all five
irregular widths, so a separately checked accumulator-pair epilogue writes
each output directly. Nineteen legal configurations matched raw projection and
GEGLU bits on 84 real matrix inputs each, with no register spills. The best
irregular tile, 32 by 96 with three stages, took 0.4770 ms across 28 weights,
versus 0.4458 ms for retained Triton. It lost all nine timing rounds. Its gain
over a three-stage native control does not establish a gain over the earlier
best two-stage native kernel. No full-request variant was run. See
`irregular_geglu-bank.json` for the exact checks, resources and samples.

The head-kernel screen tested 1,155 configurations across 11 shapes with real
activations and four random inputs per weight. Exact winners replace the two
head QKV projections, final-head output and first FFN projections, and scorer
hidden projection. These kernels retain the original weights. Head-only
results were mixed on the fixed request, 1.821 versus 1.823 ms and five of nine
rounds faster, but improved changing inputs to 1.792 from 1.805 ms in eight of
nine rounds. Both parity suites remained exact. `head-gemm.json` records the
screen and `head-full.json` the complete requests. Kernels with different raw
matrix outputs were excluded even when faster.

Host preparation now optionally batches each question's strings through the
Rust tokenizer's `encode_batch_fast`, omitting unused offset calculations.
It preserves sequence assembly, truncation, option order and validation. Token
memoization is discarded after each request. The isolated host comparison
used the same engine and CUDA Graph while swapping preparation methods. With
direct-weight MLP fusion already enabled, the batch mode reduced short requests
from 1.801 to 1.782 ms, winning nine of nine rounds, and changing inputs from
1.782 to 1.770 ms, winning eight of nine. It preserved prepared inputs, raw
logits and formatted responses across 194 requests and 336 decisions.
Another 610 CPU cases checked preparation and exception behavior, including
Unicode, malformed text, truncation and combined invalid inputs. CPU timings
in `host-prepare.json` overlapped other work and are labeled preliminary;
the promotion evidence is the exclusive-lock run in `host-full.json`.
Immutable template-token reuse was exact but slower than batch mode.

A later special-token lookup shortcut saved about four microseconds in an
initial CPU prototype, but review found cases where changed token definitions
or overridden tokenizer behavior bypassed the expected fallback. After adding
guards, it passed all 610 established cases and 224 targeted mutation, logging
and fallback checks. Fixed preparation then measured 101.291 microseconds
retained versus 101.651 for the candidate, which was faster in only three of
nine rounds. Changing-input preparation improved by just 0.365 microseconds
in pooled medians. It remains experimental; no full-request comparison was run.
`header-prepare-initial.json` preserves the rejected prototype timings, while
`header-prepare.json` and `header-prepare-extra.json` describe the final guarded
candidate. The retained preparation code was not changed.

The optional host runtime adapter then keeps NumPy views of the owned pinned
output buffers and uses the existing graph-cache entry directly on a hit.
Graph capture retains the original inference-mode guards. Each request still
serializes native packing/replay/synchronization and copies its output before
releasing the lock. The same-graph comparison reduced fixed requests from
1.724 to 1.697 ms and changing inputs from 1.744 to 1.709 ms, winning all nine
rounds for both. All 194 requests and 336 decisions preserved raw outputs and
formatted responses. Concurrent calls, output ownership, graph misses,
eviction and calls after close passed separate checks. See `host-runtime.json`.

The final four-variant comparison ran after the agents' other CPU and GPU
benchmarks had finished. With all variants resident, direct-weight MLP fusion
alone measured 1.796 ms, adding head kernels measured 1.790 ms, adding batch
tokenization measured 1.781 ms, and combining both measured 1.771 ms. The
combined variant won eight of nine fixed-input rounds and all nine
changing-input rounds against MLP fusion alone. Changing-input medians were
1.774 and 1.755 ms respectively. Head kernels added to the host improvement
won six of nine fixed-input rounds and seven of nine changing-input rounds;
that incremental gain is small. `stack-comparison.json` records the complete
samples and exact output checks. Its separate native comparison in
the table measured 1.779 ms.

A later selected-Q experiment keeps full K/V projections in the final head
but computes Q only for CLS and option markers. Both 16-row and 32-row tiles
matched all intermediate head outputs on 154 eligible requests. The 32-row
version reduced its isolated head chain from 45.08 to 41.02 microseconds.
Full validation then matched 205 requests and 362 decisions exactly, but the
request-level gain was inconsistent. Fixed short inputs measured 1.624 versus
1.621 ms, winning only eight of 15 paired rounds; changing holdouts worsened
from 1.643 to 1.647 ms and won six rounds. Eight extra fully occupied inputs
improved from 1.612 to 1.605 ms in 13 rounds. The mixed result is not retained.
See `head_q_select-result.json` and `head_q_select-full.json`.

The next attention kernel specializes the existing CUTLASS arithmetic for
one 64-token sequence with 16 heads of dimension 64. Dimensions, scale,
window and dropout become compile-time constants. Fully occupied sequences
omit the redundant all-zero bias; padded sequences retain their real bias.
Other shapes and attention windows keep the prior implementation. An isolated
18-call graph with bias improved from 116.0 to 93.6 microseconds, excluding
bias construction from both variants. In a direct full-request comparison,
the new kernel measured 1.763 versus 1.800 ms and won eight of nine rounds.
Changing padded inputs measured 1.741 versus 1.768 ms and won all nine rounds.
All 205 requests and 362 decisions matched raw choice/action logits exactly,
including both standard suites, eight extra inputs and the three benchmark
workloads. See `attention-special-full.json`. The canonical integrated
comparison measured 1.731 ms against native execution at 2.199 ms, with both
standard suites exact.

Global attention then specializes the FlashAttention source revision pinned
by the installed Torch build. A 64x64 tile with four warps preserves the
checked arithmetic while reducing ten unmasked encoder calls from 53.3 to
22.6 microseconds. Padded global attention uses the existing constant-parameter
CUTLASS implementation with a real broadcast bias; ten calls fall from 80.2
to 51.8 microseconds. The padded branch matches every raw attention element
across all 144 eligible original and holdout requests. Other shapes retain
the prior path.

The combined global-attention adapter passes 205 requests and 362 decisions
with identical raw choice/action logits. Its direct full-request comparison
measures 1.686 versus 1.739 ms, winning all nine fixed-input rounds. Changing
padded inputs measure 1.711 versus 1.740 ms, winning eight of nine rounds.
See `global-attention-full.json`, with the unmasked-only ablation preserved
separately. Combining global attention with the host replay change then
measures 1.655 ms versus native execution at 2.196 ms and passes both standard
suites exactly. Sub-millisecond inference is still unproven.
`global-runtime-integration.json` records the final measurements and source
audit. A post-run hash check detected an unused configuration tuple added
during measurement; removing that metadata yields an identical module AST.
No executed code changed, so the completed timings were retained.

Three programmatic dependent-launch variants were tested against the previous
local-attention implementation. Late and early launch hints increased fixed
request latency from 1.722 to 1.750 ms and from 1.733 to 1.758 ms. Preloading
independent weights before waiting increased it from 1.726 to 2.192 ms.
All three match raw logits and actions on 128 changing requests, but none
improves the paired changing-input median. Each has 290 actual programmatic
graph edges, verified wait/launch PTX instructions, and nine paired timing
rounds. Baseline graphs contain no PDL kernels or programmatic edges.
The preload implementation increases register use and changes the software
pipeline; this result does not rule out other PDL implementations. None was
installed. See `pdl-summary.json`, `pdl.json` and `pdl-preload.json`.

RoPE fused into attention was exact but slower. Both rotation in shared memory
and direct rotated staging retained the pinned CUTLASS/FlashAttention math.
The strongest version matched 282,591,232 attention-output elements across
154 requests and 4,312 layer inputs. However, the isolated 28-layer segment
took 0.246 versus 0.127 ms for a fully occupied sequence, and 0.363 versus
0.221 ms with padding. The standalone RoPE kernel launches 384 blocks; fusion
uses only 16–32 attention blocks and repeats key rotation for each query tile.
This is a likely cause from source inspection, not a hardware-counter finding.
The fused kernels were not installed. See `rope_attention-result.json`.

Two memory experiments did not improve the retained path. The RTX 5070 Ti
reported 48 MiB of L2 and a maximum 30 MiB persisting reservation. The cache
probe verified that CUDA capture transferred its access-policy window to the
graph's kernel nodes, then tested 8/16/30 MiB reservations for each head layer
and the final encoder layer. Fixed-input gains were inconsistent with
changing-input results, so no policy was installed. The original context
limit was restored after the experiment. See `l2-policy.json` and NVIDIA's
[L2 policy documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html).

Per-kernel L1/shared-memory preferences also failed to improve the retained
projections. Hints of 75% or 100% shared memory tied the driver default.
Preferring more L1 slowed QKV from 0.283 to 0.347 ms and MLP input from
0.447 to 0.520–0.587 ms across 28 matrices. All outputs stayed exact, all
requested hints were read back, and the original function preferences were
restored. These are driver hints, not measurements of physical cache
partitions. `carveout.json` records seven randomized rounds per preference.

Separately, a two-stream decoder reconstructed lossless 13-bit weights into
two reusable BF16 buffers while the previous matrix multiplication ran.
All outputs matched the retained arithmetic, but the best overlap variants
achieved only 0.90x, 0.97x, 0.81x and 0.83x its speed for QKV, attention output,
MLP input and MLP output respectively. Serial decoding was slower still.
`matmul-decode-pipeline.json` records all 24 decoder configurations, compared
over 28 distinct layer weights. Eager execution and repeated graph replay both
matched exactly. This probe was not integrated into the model.

Fusing the QKV projection with RoPE screened another 84 configurations. The
epilogue preserves the projection's BF16 rounding and the separately rounded
FP32 rotary products. An alternate weight layout places rotary pairs adjacent
to reduce register exchanges. All 80 successfully compiled configurations
matched the original output; four exceeded compiler/resource constraints.
The best ordinary layout measured 0.303 ms versus 0.304 ms for 28 separate
projection/rotary pairs, while the reordered layout measured 0.306 ms.
This is effectively a tie in the isolated probe, so it was not installed in
the engine (`qkv-rope.json`, `qkv-rope-interleaved.json`).

A later weight-layout screen tested 432 BF16 configurations using ordinary
loads and two- or three-dimensional TMA descriptors. It arranges each weight
matrix in the tiles consumed by the kernel, with either orientation inside
each tile. All screened outputs matched the retained kernels. QKV improved
about 2.5% in isolation; the other projections did not show a useful gain.
The full packed-QKV engine measured 1.837 versus 1.844 ms on the fixed request,
winning eight of nine rounds. It lost eight of nine changing-input rounds,
measuring 1.812 versus 1.806 ms. Its 128-request check was exact, but the
inconsistent performance excludes it from the recommendation. The optional
`packed_qkv=True` flag reproduces this probe. See `matmul-packed-*.json`,
`paired-packed-qkv.json` and `packed-qkv-parity.json`.

The fixed-exponent sparse-escape format reduces the common weight encoding to
12 bits and reads rare outliers from the retained original BF16 array. It keeps
that entire original array in VRAM, so it does not reduce allocated memory.
The GPU decoder reconstructed all 112 encoder matrices bit for bit. However,
its 128 fused GEMM configurations did not preserve all output bits, and their
best speeds were only 0.71x/0.47x/0.81x/0.70x the retained QKV/attention-output/
MLP-input/MLP-output kernels. Exact storage alone does not establish exact
matrix arithmetic. These variants were screened out (`matmul-float12-escape.json`,
`float12-decode-check.json`).

PTX bulk-prefetch hints were also added inside the existing RoPE and GEGLU
kernels to fetch the next projection's weights, without adding a GPU launch.
Six combinations of insertion point and 16/64 KiB chunk size preserved every
raw output on the 128-request check. All six slowed complete requests,
by about 23–97 microseconds in their paired measurements. Neither prefetch nor
cache reservation is enabled in the recommended constructor. The optional
`prefetch="rope"`, `"geglu"` or `"both"` flag, with `prefetch_chunk` in bytes,
reproduces the tests in `prefetch.json`. These experiments use the
[PTX bulk-prefetch hint](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch),
whose cache effect is not guaranteed by the instruction.

An independent native CUDA experiment packs the 13-bit lossless format directly
in tensor-core register order. It adds a two-stage `cp.async` shared-memory
pipeline and swizzles the activation tiles to avoid shared-memory bank
conflicts. Both an independent decoder and the native CUDA decoder reproduce
every weight bit across all 112 encoder matrices. All 96 direct-load and
pipelined configurations match the retained matrix outputs, including the
MLP output's four BF16 partial sums. Despite this, the best compressed pipeline
achieves only 0.553x/0.452x/0.563x/0.622x retained speed for QKV/attention output/
MLP input/MLP output. The uncompressed register-layout controls also lose,
at 0.615x to 0.935x retained speed. The explicit pipeline helps, but it does
not offset decoding and scheduling costs. See `native-lossless.json` and
`native-lossless-pipeline.json`; neither implementation is installed in the
engine. These remain isolated matrix measurements, not full-request timings.

The FP4 experiment uses native SM120 E2M1 tensor-core arithmetic, E4M3 scales
for groups of 16 values, and FP32 outer scales per row. The per-row outer
scales differ from the common per-tensor NVFP4 recipe. Packing and native MMA
were checked against an independently decoded FP32 matrix product before
model integration. The complete request fell to 1.394 ms, but confidence and
decisions changed: maximum probability errors were 0.1700 on the original
suite and 0.2802 on the additional inputs. It is not an accepted optimization.
The 144 ordinary-load and 128 TMA configurations all emitted this instruction:

```text
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
```

TMA improved FP4 attention projections slightly in isolation, but did not
improve both MLP projections. The vendor `scaled_mm` FP4 path was slower than
our custom kernel on all four shapes. No TMA or vendor FP4 full-request speedup
is claimed; those remain matrix-level screens.

Other matrix-level experiments were screened out before full-model integration.
Group-64 weight-only INT8 and lossless BF16 packing spent too much time on
unpacking. A group-32 INT8 Marlin build was slower on three projection types;
its small MLP-input win did not beat the exact BF16 kernel. PyTorch's vendor
MXFP8 implementation beat its BF16 baseline for MLP input but was slower than
our Triton MXFP8 implementation. None of these microbenchmarks is a full-request
latency claim. Each matrix test replays 28 distinct layer weights to exceed L2.

A 13-bit format preserved every BF16 weight bit by storing sign, mantissa and
the checkpoint's restricted exponent range. It rejected unsupported values
before packing and verified every decoded weight. Storage fell by 18.75%, but
the first decoder was much slower than cuBLAS. Loading packed tiles once and
gathering locally improved it substantially; it still failed to beat the
retained BF16 kernels. `matmul-float13*.json` includes all 128 configurations,
including resource-limit failures and numerical differences from alternative
split-K partitions. Lossless storage alone does not guarantee faster GEMM.

A revised tile-based lossless encoding also lost to plain BF16. CUDA's
[transparent memory compression](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory)
was supported and explicitly granted on this GPU, verified per allocation.
All weights were populated through GPU stores and checked for exact values.
Nevertheless, compressible BF16 allocations performed approximately the same
as ordinary allocations across all four projection types (0.99–1.00×).
Representing the same values in FP32 to expose zero low bits was slower
(0.48–0.82× BF16 speed), even in compressible allocations. This experiment
does not measure a physical compression ratio or claim lower VRAM use.
Positive controls confirmed that compression does work: streaming a 256 MiB
zero or constant allocation was 3.77× faster, and a 75%-zero pattern was
1.99× faster. Random data, random 16-bit halves and the tested 50%-zero pattern
showed no gain. These are synthetic checksum-kernel tests, not inference
results or measured physical compression ratios (`compression-patterns.json`).

A follow-up rearranges exact BF16 bits before storing them in compressible
allocations. All six formats round-trip every weight in the 28 MLP input
matrices. Bitplanes improve streaming reads from 0.357 to 0.281 ms, a 1.268x
speedup with unchanged logical size. Widened 32-bit codes appear nearly twice
as fast with compression as without, but require twice the logical bytes and
remain about as slow as ordinary BF16. See `compression-transforms.json`.
The bitplane gain does not yet translate to matrix multiplication. The first
Gluon kernel uses PTX warp shuffles to reconstruct weights before BF16 MMA.
All 12 matrix configurations match exactly, but the best compressed version
takes 1.062 ms for 28 MLP input projections versus 0.445 ms for retained BF16.
Ordinary bitplane allocation takes 1.065 ms. A synchronous BF16 control takes
0.471 ms, isolating most of the regression to the packed representation and
decoder. See `bitplane-gemm.json`; these are isolated projection timings.

A native CUDA follow-up permutes bitplanes into the MMA register order before
inference. Coalesced loads, XOR-swizzled shared memory and two or three
`cp.async` stages overlap reads with work. Paired warp transposes reconstruct
two BF16 values together and improve the matching native configuration by
1.77x. All 32 configurations match across the 28 matrices, but the best still
takes 1.520 ms versus retained BF16 at 0.444 ms. Compressible memory changes
that kernel's time by only 1.001x. The decoder remains too expensive to keep;
`bitplane-pipeline.json` records the full screen. These isolated GEMM
measurements include decoding inside the matrix kernel.

The native lossless follow-up adapts
[Turbo-Lossless revision 50b72e5](https://github.com/cenconq25/Turbo-Lossless/tree/50b72e5520f93cc855f0bd3a0114a7eb97338588).
Its upstream engine handles escape-value correction separately from the v3
GEMM kernel. This adaptation reconstructs those values before MMA so the
retained accumulation and BF16 partial-sum rounding can stay exact. It uses
SM120 TMA loads and graph-safe descriptors. All 65,536 BF16 bit patterns decode
exactly, and all 24 screened matrix configurations match across 112 checkpoint
matrices. The best adapted kernels reach only 0.36x, 0.20x, 0.45x and 0.30x the
retained speed for QKV, attention output, MLP input and MLP output. Even unsafe
controls that skip escape correction are slower. Nothing from this screen was
installed into the engine. See `turbo-lossless.json`, `turbo_LICENSE` and
`turbo_NOTICE` for results, pinned source and Apache-2.0 attribution.

A separate native kernel dedicates producer warps to lossless decoding while
consumer warps perform BF16 MMA. Asynchronous packed copies and shared-memory
phase barriers overlap the work. All 24 configurations match projection and
GEGLU outputs across all 28 MLP input matrices, and all 7,938 supported BF16
patterns round-trip exactly. Compute Sanitizer's two synchronization checks
report no races. Nevertheless, the best projection plus GEGLU takes 0.768 ms
versus retained fusion at 0.444 ms. Its raw projection takes 0.733 ms versus
0.511 ms for the otherwise matching unpacked native control. Decoding still
costs more than the 18.75% storage reduction saves. It was not installed;
`ws-lossless-summary.json` records the screen and source/binary hashes.

The accepted gain comes from better BF16 matrix tiles, compiler fusion,
attention scheduling and selected TMA loads. These techniques also apply to
earlier GPUs. The MXFP8 experiment
explicitly emits Blackwell block-scaled instructions:

```text
mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
```

Generated PTX was checked for every one of the 160 successful MXFP8 configurations.
The original detector expected a different qualifier order. `verify_mma.py`
corrects those false negatives and records PTX hashes without changing the
timing or numerical measurements. Using the native instruction did not make
the resulting quantization acceptable under our parity gate.

Profiling the native baseline put the GPU graph and transfers at about 1.92 ms
and request preparation plus response formatting at about 0.14 ms. A language
rewrite cannot remove the dominant matrix work. The linear weights total
726.7 MB in BF16. Streaming that volume once at the GPU's specified
[896 GB/s](https://blogs.nvidia.com/blog/studio-ai-geforce-rtx-5070-ti-gpu-dlss/)
would take 0.81 ms. This is an arithmetic estimate, not a measured latency floor:
caches, memory access patterns, compression and overlap affect actual traffic.
The attempt to measure DRAM counters with Nsight Compute failed with
`ERR_NVGPUCTRPERM`; no driver settings were changed. Sub-millisecond execution
would need substantially better data movement and kernel scheduling, or reduced
precision that survives broader quality testing.

Before the attention and TMA follow-up, profiling the compiled variant measured
about 1.71 ms for the GPU graph and transfers, with 372 kernel launches per
replay. Matrix kernels dominated. `profile-compiled.json` records the stage
timings and the instrumented kernel breakdown; profiler kernel times should
not be read as an additive full-request latency measurement.
The pipeline profile before the MLP rounding change reduced this to 336 kernel launches and about
1.62 ms for the GPU graph and transfers (`profile-pipeline.json`). Request
preparation and formatting still measured about 0.14 ms in isolation.
Reduction/normalization fusion reduced the count to 309 kernels and the GPU
graph plus transfers to about 1.56 ms (`profile-reduce-norm.json`).
Token tables reduced this to 307 kernels and about 1.55 ms
(`profile-token-tables.json`), still above the full-request target by themselves.
The MLP/head/host combination uses 278 kernels and measured 1.521 ms
for the GPU graph and transfers. Preparation measured 0.102 ms and response
formatting 0.028 ms in isolation. These stage measurements are diagnostic and
are not summed to produce the full-request result. See
`profile-mlp-head-host.json`. Fixed-shape attention reduces the count to
270 kernels and measures 1.484 ms for the GPU graph and transfers. Its profile
is the `profile-...-attention-special.json` report. Adding global attention
keeps 270 kernels on the fixed workload and reduces the graph and transfers
to 1.431 ms. With the host replay change, preparation measures 0.104 ms and
formatting 0.032 ms in isolation. See the
`profile-...-global-attention-host-runtime.json` report. The GPU graph alone
still exceeds 1 ms.

After native formatting, `profile-native-format.json` measures 1.428 ms for
the same 270-kernel GPU graph and transfers, and 0.010 ms for formatting alone.
This CPU change leaves the GPU execution unchanged.

A backward dependency audit found no additional live-token rows to remove from
the fixed workload. Both attention-head layers use every live encoder row,
the final encoder layer is global, and the local attention radius covers all
64 positions. The existing final-head pruning already selects only CLS and the
four real option markers. The fixed input has no sequence padding. Padding
could permit a separate optimization for some changing inputs, but only 9 of
128 holdout requests have an entirely dead 32-row tile. That route would not
improve the fixed target workload. See `dependency-audit.json` for the dependency
closure, fixture counts and source hashes; no padding kernel was implemented.

A read-only audit of that trace found 10.40 microseconds of explicit gaps at
the targeted projection/normalization boundaries per replay. The 55 residual
normalization kernels perform another 79.19 microseconds of work. Their exact
Welford reduction needs every feature of a row, distributed across projection
CTAs, so simply combining launches does not remove that work or the need to
communicate between blocks. This small observed gap does not justify a
cooperative rewrite aimed only at boundary removal. It is an instrumented
observation, not a general limit on kernel fusion. `cooperative-audit.json`
records the trace digest and stage pairs; reproduce it with
`cooperative_audit.py --trace <exported-trace.json> --replays 10`.

A separate normalization-prologue experiment tested recomputing exact row
statistics inside each MLP input tile. It matched the retained FP32 residual,
Welford statistics, BF16 normalization and downstream GEGLU on all 28 layers
for three real requests. However, statistics alone took 0.191–0.320 ms across
the layer bank, versus 0.035 ms for the existing complete normalization stages.
The full retained normalization-plus-GEGLU bank took 0.492 ms. The candidate
timings omit normalization application and GEMM, so they are evidence against
the duplicated-work mapping, not measurements or rigorous lower bounds for a
fully fused kernel. That fused kernel was not built. See
`norm-geglu-feasibility.json` and `norm-geglu-prologue.json`.

Use the retained implementation from a checkout with the
[native experiment prerequisites](../native/README.md) installed:

```python
from experiments.frontier.engine import FrontierEngine

with FrontierEngine(
    policy="bf16-splitk-exact-short-compiled",
    attention="native",
    fuse_reduce_norm=True,
    token_tables=True,
    fuse_mlp_geglu=True,
    mlp_geglu_unpacked=True,
    head_kernels=True,
    host_prepare="batch",
    attention_special=True,
    global_attention=True,
    host_runtime=True,
    native_format=True,
) as engine:
    response = engine.predict(
        state="The production API is unavailable.",
        questions={
            "urgent": {"type": "noul", "instructions": "Is an urgent response needed?"}
        },
    )
```

Use `policy="bf16-exact"` for the uncompiled variant, or
`policy="bf16-exact-short-compiled"` without `attention` for the previous
compiled implementation without the new native attention build. The compiled policy adds
first-use setup and uses the existing Torch 2.14 precision-preserving compiler
configuration. Warm the intended shapes before measuring. Timings in this
report do not describe startup. Matrix configurations are selected from the
saved reports under `results/frontier`; retuning them changes the experimental
implementation and requires fresh validation. Only the RTX 5070 Ti and the
recorded package versions have been tested. Installed engine and server
defaults are unchanged. The native attention extension requires the exact
Torch revision checked by `build_attention.py` and its matching CUTLASS headers:

```bash
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cutlass.git .research/frontier-torch-cutlass
git -C .research/frontier-torch-cutlass sparse-checkout set include
git -C .research/frontier-torch-cutlass checkout e05f953a5b3d38adc240df2ff928e0421c2abba3
flock -s /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.frontier.build_attention
flock -s /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.frontier.build_reduce_norm
flock -s /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.frontier.attention_special_build
git clone --filter=blob:none --no-checkout https://github.com/Dao-AILab/flash-attention.git .research/frontier-torch-flash
git -C .research/frontier-torch-flash sparse-checkout set csrc/flash_attn/src
git -C .research/frontier-torch-flash checkout 14c377950125c70b7a9dabf9c561fca53715ac7d
flock -s /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.frontier.global_attention_build
flock -s /tmp/laya-gpu-experiments.lock uv run --no-sync python -m experiments.frontier.native_format_build
```

The measured original attention extension build took about 30 seconds, the
reduction fusion extension about 24 seconds, and the attention specialization
about 37 seconds. The global-attention build took about 71 seconds. These
builds are separate from model
loading and Torch compilation. This experiment targets warm inference and
does not preserve the earlier uncompiled engine's first-use setup time.
See `NOTICE` for retained PyTorch and CUTLASS attribution.

Reproduce the full comparison and the additional input check:

```bash
flock /tmp/laya-gpu-experiments.lock uv run --no-sync python \
  -m experiments.frontier.compare --policy bf16-splitk-exact-short-compiled \
  --attention native --fuse-reduce-norm --token-tables --fuse-mlp-geglu \
  --mlp-geglu-unpacked --head-kernels --host-prepare batch --attention-special \
  --global-attention --host-runtime --native-format --validate

flock /tmp/laya-gpu-experiments.lock uv run --no-sync python \
  -m experiments.frontier.holdout --policies bf16-splitk-exact-short-compiled \
  --attention native --fuse-reduce-norm --token-tables --fuse-mlp-geglu \
  --mlp-geglu-unpacked --head-kernels --host-prepare batch --attention-special \
  --global-attention --host-runtime --native-format \
  --output results/frontier/holdout-native-format.json

flock /tmp/laya-gpu-experiments.lock uv run --no-sync python \
  -m experiments.frontier.check_token_tables

uv run --no-sync python -m experiments.frontier.summarize
```

`tune.py` screens BF16, `--quant`, `--lossless`, `--mxfp8` and `--weight-fp8`
variants. Its `--pipeline` option expands small-tile pipeline and warp choices.
Pass a separate `--output` for each representation. `tune_tma.py` screens
tensor-descriptor loads and warp specialization; `probe_compression.py` checks
VMM allocation modes; `probe_splitk.py` screens intermediate rounding and
reduction partitions. `probe_attention.py`, `probe_flash.py` and the raw
`attention*.json` files record attention variants. `tune_lt.py` needs
`build_lt.py` first. `profile.py` records native-stage timings and a Torch trace.
`tune_nvfp4.py` and `--tma` screen the numerical FP4 variants, and
`nvfp4_vendor.py` checks the vendor implementation. `probe_float13.py` and
`--tiled` screen exact weight compression. `compression_patterns.py` provides
positive and negative hardware-compression controls.
`probe_decode_pipeline.py` tests overlapping lossless decoding, and `probe_l2.py`
tests persisting-cache reservations. `probe_qkv_rope.py` tests projection/rotary
fusion, with `--interleaved` selecting weights reordered to place rotary pairs
next to each other. These are isolated probes, not engine defaults.
`tune_packed_tma.py --access 0/1/2` selects ordinary loads, 2D TMA or 3D TMA;
use separate output paths for each. `check_packed_qkv.py` checks the integrated
variant. `probe_float12_escape.py --check-only` verifies decoding; omit that
flag to screen GEMMs. `probe_prefetch.py` measures complete requests, and its
`--instructions-only` mode checks the generated prefetch instructions.
For the register-layout decoder, build `experiments.frontier.native_lossless_build`
under the shared build lock, then run `experiments.frontier.native_lossless_probe`
under the exclusive GPU lock. Use
`--pipeline --output results/frontier/native-lossless-pipeline.json` for the
asynchronous shared-memory version and a separate report. Build and result files record the source and binary
hashes; generated binaries remain under ignored `.research/`.
GPU runs use the exclusive lock above; native builds use `flock -s` on the same
lock. Compilation and GPU timing should not compete with each other.

The optional Marlin probe builds Apache-2.0 sources from
[vLLM revision 00b7847](https://github.com/vllm-project/vllm/tree/00b7847c8036b667742b4efb21aab1de51fd4721).
Clone that revision under `.research/frontier-vllm` before running
`build_marlin.py` and `probe_marlin.py`. The source checkout retains its original
notices; generated sources and binaries remain under ignored `.research/`.
The build contains only BF16 activation and biased UINT8 kernels with group
sizes 32 and 64, targeted at SM120. It does not install vLLM or change the
inference environment.

`probe_mlp_geglu.py` screens packed MLP fusion; pass `--unpacked` and
`--output results/frontier/mlp-geglu-unpacked.json` to screen direct weights.
`head_probe.py` screens the head kernels and `head_compare.py` checks their
independent contribution. `head_q_select_probe.py` screens compact final-head
Q projections; `head_q_select_full.py` checks the selected configuration in
complete requests. Both require the exclusive experiment lock.
`host_probe.py` checks preparation and error parity,
and `host_compare.py` measures preparation modes on the same engine.
`stack_compare.py` compares the final four combinations in one process.
`attention_special_probe.py` screens constant-parameter attention and
`attention_special_padding.py` checks its padding-aware arithmetic.
`attention_special_full.py` runs the paired comparison against the previous
MLP/head/host combination. The integrated `attention_special=True` option
requires `attention="native"` and `token_tables=True` and must be set before
graph capture. Its extension uses the same pinned Torch and CUTLASS sources.
Build `query_shard_build.py` under the shared lock, then run
`query_shard_probe.py` under the exclusive lock for the query-block screen.
`query_shard_full.py` checks the 32-row indexing control in complete requests
under the exclusive lock. None of these variants changes engine defaults.
`global_attention_probe.py` checks FlashAttention tile variants and
`global_attention_padding.py` checks the broadcast-mask branch.
`global_attention_full.py` measures their combined adapter against the prior
local-attention implementation. `global_attention=True` requires
`attention_special=True`. `host_runtime_probe.py` compares output-view reuse
and graph-cache dispatch around one identical resident model and checks
ownership, concurrent calls, eviction and close behavior.
`native_format_probe.py` checks NumPy arithmetic, public rounding and fallback
behavior under the exclusive lock. `native_format_compare.py` compares the
original and native formatter around one model. That script reads the
recommended constructor and must use `native_format=False` when measuring a
reference-versus-native comparison. It removes that candidate option before
creating the baseline; its saved report records the constructor.
`mlp_schedule_probe.py` runs the larger direct-weight GEGLU tile/accumulator
screen. `cluster_gemm_capability.py` builds and checks generic cluster support
and multicast code generation; it does not benchmark a multicast GEMM.
`decompress_capability.py` queries dedicated hardware decompression attributes
without creating a context or allocating GPU memory.
`split_tma_probe.py` screens fixed-boundary BF16 partial projections, and
`split_tma_full.py` checks the best configuration in complete requests.
`split_tail_probe.py` checks trimming zero-filled split iterations and CTA
ordering. `weight_pitch_probe.py` checks row padding with zero-padding controls.
`activation_sparsity.py` audits exact GEGLU zeros without measuring latency.
All three use the exclusive experiment lock. `wide_load_audit.py` performs
its compile/disassembly checks under the shared build lock without running
any kernels.
`compact_gelu_probe.py` runs the exhaustive function-domain check and bank
comparison. `packed_bf16_probe.py` checks the packed-product instruction and
its bank timing. Both use the exclusive lock. For native CUTLASS, first run
`cutlass_geglu_build.py` under the shared lock, then `cutlass_geglu_probe.py`
under the exclusive lock. The separate `cutlass_geglu_erf_*` scripts build,
derive corrections, and measure the matched native erf control; they preserve
the initial lookup-based build and reports.
For irregular native tiles, run `irregular_geglu_build.py`, then rerun it with
`--stage-two`, under the shared lock. Run `irregular_geglu_probe.py --smoke`
and then the probe without that flag under the exclusive lock.
Build `norm_geglu_build.py` under the shared lock and run `norm_geglu_probe.py`
under the exclusive lock for the normalization-prologue feasibility control.
`prepare_cpu_profile.py` profiles retained CPU preparation under the exclusive
lock without loading the model. Its instrumented timings diagnose overhead;
they are not full-request latency measurements.
`header_prepare_extra.py` checks special-token mutations and fallback behavior;
`header_prepare_probe.py` checks and times the guarded lookup candidate. Both
use the exclusive lock. `header_prepare_full.py` provides a full-request
comparison harness for future work; it has not been run.
`weight_stream_probe.py` traces the executed dense projections and measures
three checksum controls under its internally acquired exclusive lock.
`weight_layout_probe.py` launches one dedicated worker per weight placement;
each worker acquires the exclusive lock internally. Run the dispatcher without
an outer flock so its workers can acquire that lock. Banks stay owned until
worker exit because compiled methods can retain parameter references.
`io_trace_audit.py --trace PATH` reads an existing Torch profiler trace without
running GPU work.
Run `mapped_io_probe.py` and `mapped_io_lifetime.py` under the exclusive lock
before `mapped_io_full.py`, also under the exclusive lock. The full comparison
requires the strengthened poisoned-buffer reports and matching source hashes.
`mapped_io_adapter.install(engine)` installs the optional output-only capture
path before any graphs exist; it does not change model arithmetic or defaults.
`carveout_probe.py` measures per-function L1/shared-memory preferences and
restores the original settings. Build `rope_attention_build.py --direct`
under the shared lock and run `rope_attention_probe.py` under the exclusive
lock to check direct rotary staging inside attention; omit `--direct` from
the build for the shared-memory rotation variant. Use separate output paths
when comparing builds. These kernels are isolated experiments.
`pdl_probe.py` tests exact-arithmetic kernel clones with `--modes late early
preload`. It reads the currently recommended constructor from `summary.json`;
each report records the constructor used. Its PTX checks run before the first
kernel launch, and its graph checks reject a contaminated baseline.
`compression_transforms.py` measures transformed storage reads;
`probe_bitplane_gemm.py` adds exact warp-shuffle reconstruction and BF16 MMA.
Build `bitplane_pipeline_build.py` under the shared lock and run
`bitplane_pipeline_probe.py` under the exclusive lock for the native
asynchronous version. Its `--smoke` option checks one matrix; omit it for
the complete 28-matrix comparison.
Build `turbo_build.py` under the shared lock, then run `turbo_probe.py` under
the exclusive lock for the separate native lossless adaptation.
Build `ws_lossless_build.py` under the shared lock, then run
`ws_lossless_probe.py --shared-copy` under the exclusive lock for the producer/consumer
lossless pipeline. `ws_lossless_check.py` provides the synchronization probe
used with Compute Sanitizer.
All of these timing runs, including CPU-only preparation benchmarks, require
the exclusive experiment lock. Run no simultaneous CPU or GPU benchmark.

The vendor nvCOMPDx ANS screen tested 16 configurations against the entire
300.94 MB MLP input-weight bank. Every configuration reconstructed all bytes
exactly. Its fastest decode-to-shared-memory plus checksum took 2.801 ms,
compared with 0.357 ms for a matched uncompressed read. The best representation
for speed used 212.85 MB of compressed payload, with allocation padding and
metadata counted separately. Compression, setup and reverse rearrangement were
excluded from this timing, so it is optimistic for a fused matrix multiply.
The decoder lost every paired round. It was not integrated into inference;
these are weight-bank measurements, not full-request timings. See
`nvcompdx-summary.json` and `nvcompdx-ans.json`.

This optional probe uses the CUDA 13 NVIDIA MathDx 26.06.1 SDK under
`.research/frontier-nvcompdx/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06`.
The downloaded archive SHA-256 is
`59a9233db34b75568acbcc5284e6cefe6fad5577ee644f85044971d62eeea353`.
With that SDK present, build `nvcompdx_build.py` under the shared lock and run
`nvcompdx_probe.py` under the exclusive lock. The SDK is not included here;
the adapted example source retains its Apache-2.0 license and notices.

Raw samples and numerical details are in
[`results/frontier`](../../results/frontier/), with computed comparisons in
[`summary.json`](../../results/frontier/summary.json). The final local code and
result hashes are recorded in
[`manifest.json`](../../results/frontier/manifest.json).
