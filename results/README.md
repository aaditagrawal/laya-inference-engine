# Benchmark results

Results use hardware models and optimization names rather than private machine
names. Every comparison retains its hardware, workload and timing boundaries.

- `benchmark.json`, `upstream-fast.json`, `upstream-compile.json`: original
  serial request comparisons on RTX 5070 Ti.
- `native-optimizations/`: native CUDA kernels, host I/O, window attention and
  compilation experiments.
- `latency-optimizations/`: projection fusion, concurrent streams, padding and
  precompiled deployment experiments.
- `rtx-a6000/`: CPU, GPU and hybrid comparison on RTX A6000.
- `history/`: consolidated tables and CSV exports, with startup, HTTP and
  concurrent measurements separated from warm serial latency.

Repository naming cleanup changed descriptive labels and file paths, not saved
measurement values. Local environment paths are normalized for publication.
`artifact-index.json` records the current files and the
original report hashes where applicable. Root experiment manifests describe
the current repository snapshot. Source hashes embedded in historical reports
describe the code at measurement time, before the naming refactor; they are not
claims that the current renamed source is byte-identical.

The cleanup verification is separate from historical performance measurements.
Original local artifacts were backed up before renaming. No historical benchmark
was rerun or replaced to make the naming changes.
