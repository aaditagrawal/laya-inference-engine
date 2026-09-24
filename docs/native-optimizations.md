# RTX 5070 Ti native and compiled experiments

These experiments ran locally on the RTX 5070 Ti, 16 GB, SM120, with
three agents implementing host, native-kernel and compiler experiments. The
recommended direction is exact BF16 kernels, windowed attention and a smaller
host execution path. Quantization changed decisions on the regression fixture.
The implementation and results are included as opt-in experiments.

## Measurement boundaries

The machine has a Ryzen 5 5600G CPU. Runs used four PyTorch CPU threads,
PyTorch 2.14.0+cu132, Triton 3.8.0, Transformers 5.17.0, Laya 0.3.9 and driver
595.84. CUDA extensions target `sm_120` and were built with CUDA Toolkit 13.1.
The checkpoint revision is `5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`, with
weight SHA256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.

Every timed GPU experiment acquired the same exclusive advisory lock.
During the measurement phases, CPU-heavy builds acquired a shared lock that
excluded inference measurements. The initial host-extension setup build
preceded that coordination rule.
An existing idle GPU process remained resident. No clocks, power settings or
unrelated services were changed.

Warm request timings include tokenization, packing, host/device transfers,
inference and response formatting. Model loading, shape compilation/capture,
warmup and HTTP are excluded on both sides. The main combined comparisons keep
both engines resident and use five rounds with randomized engine order, five
warmups and 30 measured requests per block. The raw reports retain all samples
and block order. These are serial request measurements, not concurrent serving
capacity. Laya returns decisions, so generated-token TPS and streaming TTFT do
not apply.

The numerical fixture has 66 requests and 208 decisions, including empty state,
literal markers, left truncation, padding, changed inputs, irregular batches,
single-option requests and questions with up to 33 options. We compare raw
logits, unrounded probabilities, action logits and selected decisions. An
independent upstream model reference uses both matched padded and original
unbucketed shapes. The native and full-model compiled modes match the existing
engine's raw logits and actions exactly on this fixture and match all 208
upstream selected decisions. The separate matrix-autotuned mode has numerical
differences, described below. This is synthetic implementation parity, not
labeled task accuracy.

## Combined confirmation

The packaged default uses the final vectorized CUDA normalization kernel,
corrected Triton GEGLU, exact window attention and native host path. In its
five-round confirmation, median latency changed as follows:

- One short question: 2.794 to 2.188 ms, 21.7% lower.
- Sixteen short questions: 12.451 to 10.700 ms, 14.1% lower.
- One long question: 8.669 to 7.367 ms, 15.0% lower.
- Sixteen long questions: 101.665 to 88.261 ms, 13.2% lower.

Throughput was 457 decisions/s for one short question, 1,491 decisions/s for
sixteen short questions, 135 decisions/s for one long question and 181
decisions/s for sixteen long questions. All 208 decisions, raw logits and
action outputs matched exactly in the packaged implementation. This is the
recommended experimental `native-window` mode.
[Final packaged comparison](../results/native-optimizations/final-native.json)

The initial native combination used exact CUDA normalization, corrected Triton
GEGLU, C++ packing/replay and the direct Rust tokenizer interface. Median warm
request latency fell from 2.811 to 2.207 ms for one short question, 12.532 to
10.782 ms for sixteen short questions, 8.712 to 7.905 ms for one long question,
and 102.300 to 93.069 ms for sixteen long questions.
[Raw randomized comparison](../results/native-optimizations/combined-native.json)

Adding exact window attention improved long inputs further. In a fresh paired
run, medians were 2.816 to 2.221 ms for one short question, 12.507 to 10.802 ms
for sixteen short questions, 8.752 to 7.474 ms for one long question and 101.967
to 88.457 ms for sixteen long questions. That is 21.1%, 13.6%, 14.6% and 13.3%
lower latency respectively. Both engines passed the full fixture in the same
process, including exact outputs for the baseline after candidate installation.
[Raw randomized comparison](../results/native-optimizations/combined-window.json)

The packaged default also passed exact raw-output checks on larger and irregular
batches. Three randomized rounds of 20 timed requests per engine gave these
median latencies:

- Five medium questions: 24.589 to 22.000 ms.
- Sixteen medium questions: 48.585 to 43.277 ms.
- Thirty-two short questions: 24.657 to 21.693 ms.
- Sixty-four short questions: 48.707 to 42.452 ms.
- Thirty-two long questions: 207.399 to 174.611 ms.

This phase limited each engine to one resident graph. All five requests had
bitwise-identical decision and action logits. Sixty-four long questions were
not tested. [Extended batch comparison](../results/native-optimizations/extended-native.json)

## What the language work achieved

The C++ extension packs request data into pinned memory and performs graph
launch and completion through the CUDA runtime. Capturing transfers with
inference avoids repeated Python calls to copy and replay. The native call
releases the GIL while GPU work completes, while an engine lock protects graph
buffers. Returned arrays own their data.

C++ packing for one short question reduced its microbenchmark from 28.7 to
0.82 microseconds.
The complete request benefit from C++ packing and graph I/O was much smaller,
roughly 0.6% to 3.1%. Direct access to the existing Rust Tokenizers backend
removed additional Python wrapper overhead. Together the host changes reduced
median request latency by 9.1% for one short question and 1.6% for sixteen long
questions. The SDK's sequence builder and response formatter remain intact.
There is no newly written Rust model or full C++ server here.
[Host comparison and scope](../results/native-optimizations/host/summary.json)

A matched localhost HTTP experiment used the original FastAPI/Pydantic app,
thread pool, uvicorn, keepalive client and TCP_NODELAY for both engines. Host
changes reduced medians from 3.967 to 3.683 ms for one short question and 14.091
to 13.149 ms for sixteen short questions. One long question was nearly tied,
9.905 to 9.831 ms. Four alternating rounds each contained 40 measured requests.
An initial socket configuration produced delayed-ACK stalls and was excluded.
[HTTP measurements](../results/native-optimizations/host/http.json)

Thirty concurrent requests from four threads with different inputs sharing one
graph key returned exact results. Previous outputs remained unchanged after
subsequent replays. Closing the host adapter returned live device and active
pinned allocations to their prior levels; cached pinned blocks can remain.
Closing the underlying model returned its live allocated GPU memory to zero
in the isolated process. These checks test ownership and serialization,
not parallel execution of one engine.

## CUDA, Triton and PTX

The useful CUDA change fuses residual addition, LayerNorm and BF16 output
conversion. It preserves PyTorch's floating-point reduction order. A later
specialization for the fixed 1024 hidden dimension uses one synchronization
barrier and vectorized loads/stores. The implementation retains PyTorch's BSD
license and attribution.

Naive GELU and normalization replacements changed decisions. The accepted
GEGLU implementation evaluates every BF16 input bit pattern at initialization
against the installed PyTorch GELU, then derives four sparse corrections for
Triton's bundled math library. It preserves the intermediate BF16 rounding
before multiplying by the gate. A full 128 KB lookup table was also exact but
had less favorable small-input performance.

The handwritten PTX warp shuffle and the equivalent CUDA intrinsic compiled
to the same 504-instruction machine code, with identical instruction-text
SHA256. There was no independent PTX speedup. Fusion, memory traffic and the
reduction design explain the useful changes.
[PTX versus intrinsic evidence](../results/native-optimizations/kernels/ptx-vs-intrinsic.json)

The accepted attention path exploits the existing CUTLASS windowed kernel.
Appending 64 masked key/value positions and using a 129-token bottom-right
causal window exactly represents this checkpoint's inclusive bidirectional
window. RoPE also writes the masked tail, avoiding separate per-layer padding
copies. The implementation uses a private PyTorch operator and explicitly
restricts the tested package series.

## Compiled GPU option

Full-graph Inductor compilation is implemented as an opt-in mode. The outer
engine still owns CUDA Graph capture and replay. Preserving intermediate casts
alone was insufficient. Compilation initially switched CUDA math libraries
after GELU correction calibration, and other fused normalizations changed
rounding. The accepted policy pins the math library before calibration and
keeps the remaining LayerNorm and small GELU operations opaque to the compiler.

The corrected compiled model passed all 66 requests with bitwise-identical
logits and actions. Without the host wrapper its four warm medians were 2.463,
11.477, 7.957 and 93.046 ms. Compilation did not consistently improve on the
uncompiled native/window model. The packaged comparison and startup runs
separate this modest warm effect from new-shape compilation cost.

The final packaged compiled mode, with the vectorized normalization and native
host wrapper, also passed all 66 requests exactly. Its randomized five-round
medians were 2.106, 10.574, 7.511 and 91.545 ms for one short, sixteen short,
one long and sixteen long questions. Its paired existing-engine baseline was
2.820, 12.534, 8.682 and 101.622 ms.
[Final compiled comparison](../results/native-optimizations/final-compiled.json)

A subsequent direct native-versus-compiled comparison kept both modes resident
and randomized their order over five rounds of 30 samples. The short median
was 2.189 ms native versus 2.102 ms compiled, a 4.0% reduction. Sixteen long
questions took 88.036 ms native versus 91.701 ms compiled, a 4.2% increase.
Both checked workloads had bitwise-identical logits and actions.
Choose `compiled` for a stable, warmed short-request workload; keep
`native-window` for the default mixed workload and lower setup cost.
[Direct comparison](../results/native-optimizations/compiler/direct-native-vs-compiled.json)

A fresh Python process using the original exact native norm, window attention
and C++/Rust host path took 12.698 seconds for the first short request with empty
isolated Inductor/Triton caches. Reusing those caches in another new process
reduced the first request to 2.761 seconds. Time from Python entry through the
first response was 18.238 and 7.761 seconds respectively. Warm medians were
2.129 and 2.139 ms. Model files and native extensions already existed; these are
single startup observations, not fresh-install measurements or a startup
distribution. Imports and model/adapter setup are recorded separately.
[Fresh-cache startup](../results/native-optimizations/compiler/clean-startup-cold.json),
[Reused-cache startup](../results/native-optimizations/compiler/clean-startup-warm.json)

The matching uncompiled native/window startup, using the same original
normalization kernel and existing disk caches, took 0.619 seconds for its first
request and 5.294 seconds from Python entry to the first response. Imports took
1.311 seconds and model/adapter setup took 3.363 seconds. Its warm median was
2.233 ms. This separates first-shape setup from the roughly five-second total
process startup. It is not the original production engine's startup or a
comparison with empty native Triton caches.
[Native/window startup](../results/native-optimizations/compiler/clean-startup-native-window.json)

## Matrix autotuning with a numerical tradeoff

The `autotuned` mode compiles one shared Linear callable with max-autotune,
leaving the native nonlinear kernels and window attention intact. It tries
ordinary Triton, ATen and persistent TMA candidates. All 92 recorded winner
selections used ordinary Triton or ATen. No TMA candidate won, so enabling that
search does not demonstrate a TMA benefit.

The final packaged version includes the vectorized norm and native host path.
Five randomized rounds of 30 requests gave these medians against its paired
existing-engine baseline:

- One short question: 2.792 to 2.097 ms, 24.9% lower.
- Sixteen short questions: 12.446 to 10.660 ms, 14.3% lower.
- One long question: 8.596 to 7.246 ms, 15.7% lower.
- Sixteen long questions: 101.153 to 84.699 ms, 16.3% lower.

All 208 selected decisions match the existing engine and both upstream
references. Maximum choice-probability drift is 0.00431669, about 0.432
percentage points. Action logits differ by up to 32.0. The action softmax
probabilities and action argmax match exactly on all 208 checked decisions;
these particular action outputs are saturated. This does not establish action
equivalence on arbitrary requests. The regression now measures action
probabilities directly in addition to retaining raw action-logit differences.
[Final packaged autotuned comparison](../results/native-optimizations/final-autotuned.json)

The initial sixteen-long tuning pass took 89.823 seconds after loading; a
subsequent process reused its caches and took 2.189 seconds for that shape.
Other input shapes require their own specialization and tuning. The broader
screening completed all 66 requests in 323.7 seconds, including additional
shape setup. This mode is the fastest measured long-batch option here, but its
setup cost and numerical changes make it an explicit experimental choice.
The default remains the exact `native-window` mode.
[Full tuning regression and action bounds](../results/native-optimizations/compiler/gemm-native-padded-window-tma-full.json)

## Rejected paths

Selective FP8 MLP quantization produced 16-long medians as low as 82.865 ms,
but changed 1 to 4 of 208 decisions across the four policies. Maximum absolute
probability differences ranged from 0.035 to 0.214. Attention, residual and head
precision were retained in these tests. [Selective FP8 reports](../results/native-optimizations/precision/)

Three SmoothQuant-style channel-balancing variants calibrated only on the first
ten fixture requests. They reached roughly 82.8 to 83.1 ms for sixteen long
questions, but matched only 121, 124 and 124 of the 128 calibration-heldout
decisions. Those heldout cases were reused to assess all three preselected
alphas, so this is not a final untouched evaluation set. Short requests also
regressed to about 2.9 ms. None is recommended.
[Calibration and heldout results](../results/native-optimizations/precision/smooth/summary.json)

FlexAttention improved a local-attention microbenchmark substantially but
changed 2 of 208 final decisions. Forced cuDNN SDPA also changed 2 decisions.
Triton tree-reduction LayerNorm changed one decision. A two-pass LayerNorm
matched top decisions but moved probabilities by up to 0.034. Pruning the final
head's Q projection passed decision checks but showed no clear end-to-end win.
These outcomes are retained rather than treating operation speed as evidence
of a safe whole-model improvement.

## What this establishes about Blackwell

All new end-to-end comparisons hold RTX 5070 Ti's GPU fixed. They measure software
improvements on Blackwell and cannot assign a percentage to the hardware
generation. The exact BF16 improvements use techniques that also exist on
other CUDA architectures. The FP8 path does exercise native SM120 TMA GEMMs,
but its numerical failures prevent recommending it.

RTX 5070 Ti is SM120. Data-center Blackwell uses different capabilities; SM100
tensor-memory and `tcgen05` paths are not a universal implementation for RTX.
A kernel compiled for `sm_120` does not by itself prove a Blackwell-exclusive
instruction advantage. These sources document the distinction:
[NVIDIA capability list](https://developer.nvidia.com/cuda/gpus),
[CUTLASS Blackwell architectures](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms),
[Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/13.1.0/blackwell-tuning-guide/index.html).

Only this RTX 5070 Ti and checkpoint were validated here. This work does not
establish performance on every Blackwell GPU, a globally optimal kernel, an
NVFP4 implementation, or the benefit of a full framework-free C++/Rust rewrite.

## Source and publication

The opt-in engine, builds and benchmark runner are under
[experiments/native](../experiments/native/README.md). Raw results and source
hashes are under [results/native-optimizations](../results/native-optimizations/).
The three packaged modes each completed the 66-request regression. The native
mode also passed exact checks on the five extended workloads. Four CPU tests
verify that the nonexact-mode gate rejects changed decisions, changed actions
and excessive probability drift. The source passes Ruff, and the interactive
results view passes its TypeScript check. Production runtime files are unchanged.

The separate RTX A6000 startup observation remains an RTX A6000 measurement:
0.40 seconds
for the engine versus 36.7 seconds for the compiled FP16 GPU implementation,
after loading. Warm latency favored that compiled implementation. Those
figures exclude loading/download and are not clean-machine cold starts.
