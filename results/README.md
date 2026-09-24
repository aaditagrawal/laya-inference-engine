# Benchmark results

Results use hardware models and optimization names rather than private machine
names. Every comparison retains its hardware, workload and timing boundaries.

## Public-mode validation and timing

[fast-package.json](fast-package.json) records the current public modes on
RTX 5070 Ti. One short request measured **1.610069 ms fast** versus
**2.803832 ms balanced**, a **42.6% latency reduction**. Each fixed workload has
900 requests per mode across nine randomized paired rounds. The complete
request timer includes tokenization, transfers, inference and formatting;
loading, compilation, first capture and HTTP are excluded equally.

One long question measured 6.967680 versus 8.615601 ms, and sixteen short
questions measured 10.553087 versus 12.590964 ms. Cycling 128 changing short
fixtures over nine rounds measured 1.635720 ms fast versus 2.783642 ms balanced.
Sub-millisecond full requests were not achieved.

Fast matched its retained reference on 199 requests with 354 decisions, including
prepared inputs, raw logits/actions and public responses apart from runtime
metrics. Another 52 concurrent calls, output ownership, graph eviction and close
checks passed. Source and native-library hashes were unchanged during the run.
The retained short result in this run was 1.611524 ms, effectively tied with the
packaged runtime. No extra speedup is attributed to extraction.

[package-checks.json](package-checks.json) records the installed-wheel check.
[Reproduction commands](../docs/performance.md#dependencies-and-reproduction)
cover `scripts.validate_fast_package`, a simpler sequential public-mode benchmark,
and `scripts.validate_wheel`. These are synthetic implementation-parity checks,
not labeled task-accuracy measurements.

## Retained result before packaging

[frontier/summary.json](frontier/summary.json) records the retained BF16
implementation at **1.623769 ms** for a full warm short request, versus
**2.197088 ms** for its paired native baseline, a **26.1% latency reduction**.
Each mode has 250 requests across five paired rounds. The timer includes
tokenization, transfers, inference and formatting; it excludes loading,
compilation, first capture and HTTP. Sub-millisecond full requests were not
achieved.

The native baseline is an earlier optimized implementation, not the balanced
package default. The default's older 2.812 ms result belongs to the historical
upstream comparison below. Do not multiply that comparison's speedup by the
latest paired reduction.

This earlier result predates extraction into the public `FastEngine`. The runtime
lives under [src/laya_blackwell/fast](../src/laya_blackwell/fast), while
`frontier/` preserves optimization evidence, including rejected candidates.
Raw logits and actions matched the baseline on 66 original requests with
208 decisions and 128 holdout requests. Public responses matched on all
194 requests, excluding runtime metrics. These are synthetic parity checks,
not labeled task-accuracy tests.

- [Full paired samples](frontier/full-bf16-splitk-exact-short-compiled-attn-native-reduce-norm-token-tables-mlp-geglu-unpacked-head-kernels-host-batch-attention-special-global-attention-host-runtime-native-format.json)
- [Changing-input holdout](frontier/holdout-native-format.json)
- [Public response checks](frontier/native-format-full.json)
- [Measurement-time source and binary provenance](frontier/native-format-integration.json)
- [Optimization archive and reproduction](../experiments/frontier/README.md)
- [Chart data and source hashes](../docs/assets/chart-data.json)

## Historical comparisons

- `benchmark.json`, `upstream-fast.json`, `upstream-compile.json`: original
  serial request comparisons on RTX 5070 Ti.
- `benchmark-http.json`: matched localhost HTTP wrapper for the older backends,
  with startup excluded on both sides.
- `native-optimizations/`: native CUDA kernels, host I/O, window attention and
  compilation experiments.
- `latency-optimizations/`: projection fusion, concurrent streams, padding and
  precompiled deployment experiments.
- `rtx-a6000/`: CPU, GPU and hybrid comparison on RTX A6000.
- `history/`: consolidated tables and CSV exports, with startup, HTTP and
  concurrent measurements separated from warm serial latency.

Upstream `fast=True` is the Laya SDK's mode. It is separate from this
repository's public `--mode fast`. A6000 first-use setup, Blackwell AOT startup,
warm requests and concurrent throughput have different scopes; their figures
must remain separate. [Measurement guide](../docs/performance.md),
[public modes](../docs/performance-modes.md)

## Provenance

The earlier repository naming cleanup changed descriptive labels and file
paths, not saved measurement values. Its published environment paths were
normalized separately from the raw optimization archive.
`artifact-index.json` records the current files and the
original report hashes where applicable. Root experiment manifests describe
the current repository snapshot. Source hashes embedded in historical reports
describe the code at measurement time, before the naming refactor; they are not
claims that the current renamed source is byte-identical.

The cleanup verification is separate from historical performance measurements.
Original local artifacts were backed up before renaming. No historical benchmark
was rerun or replaced to make the naming changes.
