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

`render.ts` writes six PNGs and the unrounded plotted values in
[`../assets/chart-data.json`](../assets/chart-data.json). Commit those outputs
along with changes to the renderer. To update Dither Kit deliberately, change
the pinned registry revision in `package.json`, reinstall in a clean chart
checkout and inspect the generated images.

| Image | Input files | Measurement |
| --- | --- | --- |
| Warm latency | [Original](../../results/benchmark.json), [upstream fast](../../results/upstream-fast.json), [native summary](../../results/native-optimizations/summary.json) | Full warm request p50. Historical runs, not a single paired experiment. |
| Experimental gains | [Fusion summary](../../results/latency-optimizations/summary.json), [serving confirmation](../../results/latency-optimizations/serving/confirmation-summary.json) | Paired long-batch p50 and four-caller throughput, measured separately. |
| Startup | [Offline matrix](../../results/latency-optimizations/aot/final-offline/matrix-summary.json) | Module entry through the first completed response, median of three fresh processes. Warm OS caches and prebuilt artifacts. |

These are software comparisons on the same RTX 5070 Ti, not a measurement of
Blackwell's advantage over an older GPU. The images retain their workload,
units, exclusions and experimental status when shared outside the README.
