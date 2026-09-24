# Benchmark chart images

The root README uses static PNGs painted with [Dither Kit](https://www.tripwire.sh/dither-kit)'s
ordered-dither canvas engine. Light and dark versions follow the reader's GitHub
theme. All axes start at zero; the text stays sharp while the bars carry the dither.

The renderer reads the committed result files directly. It does not run inference
or add dependencies to the Python package.

```bash
cd docs/charts
bun install --frozen-lockfile
bun run setup
bunx playwright install chromium
bun run check
bun run render
```

`setup` installs the upstream registry source at commit
`1e7faee9aa252e499651e6736ed65f7a07d9a6bd` through shadcn. The downloaded
components stay ignored and are not redistributed as this project's source.
The renderer uses Dither Kit's `paintColumn` with our own static layout, labels
and theme palette. Dependencies, fonts and the Chromium version are locked in
`bun.lock`. Images render at twice the 1,000-pixel layout width. No external
fonts, animations or random effects are used at render time.

`render.ts` writes ten PNGs and the unrounded plotted values in
[`../assets/chart-data.json`](../assets/chart-data.json). Commit those outputs
along with changes to the renderer. To update Dither Kit deliberately, change
the pinned registry revision in `package.json`, reinstall in a clean chart
checkout and inspect the generated images.

The data export records each chart's input paths and SHA-256 hashes. The public
modes chart recomputes p50 for balanced, fast and retained implementations from
raw samples, including the separate changing-input case, and checks every value
against the package summary. Its bars show the paired balanced and fast short
results. The pre-extraction chart also recomputes its medians and checks them
against the accepted frontier entry. A changed recommendation or mismatched
summary fails rendering instead of silently relabeling an older chart.

| Image | Input files | Measurement |
| --- | --- | --- |
| Public modes | [Package validation](../../results/fast-package.json) | One short question, 900 full requests per mode across nine randomized paired rounds. Balanced 2.803831943 ms, fast 1.610069477 ms, 42.6% lower latency. Includes tokenization, copies, inference and formatting; excludes loading, compilation, first capture and HTTP. |
| Historical warm latency | [Original](../../results/benchmark.json), [upstream fast](../../results/upstream-fast.json), [native summary](../../results/native-optimizations/summary.json) | Full warm request p50. Historical runs, not a single paired experiment. Earlier engine configurations retain their measured values. |
| Paired result before packaging | [Frontier summary](../../results/frontier/summary.json), [full raw report](../../results/frontier/full-bf16-splitk-exact-short-compiled-attn-native-reduce-norm-token-tables-mlp-geglu-unpacked-head-kernels-host-batch-attention-special-global-attention-host-runtime-native-format.json), [holdout](../../results/frontier/holdout-native-format.json) | One short question, 250 full requests per mode across five paired rounds. Native baseline 2.197088033 ms, optimized BF16 1.623769465 ms, 26.1% lower latency. Earlier native baseline differs from balanced mode. Retains the `latest-paired` filename for existing links. |
| Experimental gains | [Fusion summary](../../results/latency-optimizations/summary.json), [serving confirmation](../../results/latency-optimizations/serving/confirmation-summary.json) | Paired long-batch p50 and four-caller throughput, measured separately. |
| Startup | [Offline matrix](../../results/latency-optimizations/aot/final-offline/matrix-summary.json) | Module entry through the first completed response, median of three fresh processes. Warm OS caches and prebuilt artifacts. |

These are software comparisons on the same RTX 5070 Ti, not a measurement of
Blackwell's advantage over an older GPU. The images retain their workload,
units, exclusions and experimental status when shared outside the README.
The public-mode and pre-extraction reductions use different baselines. Both
are independent of the historical upstream, fusion and serving comparisons.
Do not multiply their speedups. Packaging effectively ties the retained
implementation; the chart does not claim a speedup from extraction itself. The startup chart
measures separate native, compiled and AOT deployment modes; it does not report
startup for the latest optimized BF16 configuration. Exactness checks use
synthetic implementation-parity fixtures, not labeled model-quality data.
