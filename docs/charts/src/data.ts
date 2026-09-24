// Read the recorded results directly. No benchmark values are copied into the renderer.
import original from "../../../results/benchmark.json";
import upstreamFast from "../../../results/upstream-fast.json";
import native from "../../../results/native-optimizations/summary.json";
import latency from "../../../results/latency-optimizations/summary.json";
import serving from "../../../results/latency-optimizations/serving/confirmation-summary.json";
import startup from "../../../results/latency-optimizations/aot/final-offline/matrix-summary.json";
import frontier from "../../../results/frontier/summary.json";
import latestFull from "../../../results/frontier/full-bf16-splitk-exact-short-compiled-attn-native-reduce-norm-token-tables-mlp-geglu-unpacked-head-kernels-host-batch-attention-special-global-attention-host-runtime-native-format.json";
import packaged from "../../../results/fast-package.json";

function required<T>(value: T | undefined): T {
  if (value === undefined) throw new Error("A benchmark row is missing; check the source schema.");
  return value;
}

const first = (backend: string) => required(original.rows.find(
  row => row.backend === backend && row.questions === 1 && row.state_length === "short",
)).p50_ms;
const optimized = (file: string) => required(required(native.comparisons.find(
  comparison => comparison.file === file,
)).rows.find(row => row.case === "1-short" && row.variant !== "baseline")).p50_ms;

export const warm = [
  {label: "Upstream default", detail: "Original comparison", value: first("upstream"), color: "neutral"},
  {label: "Upstream fast=True", detail: "Separate upstream run", value: required(upstreamFast.rows.find(
    row => row.questions === 1 && row.state_length === "short",
  )).p50_ms, color: "neutral"},
  {label: "First fused engine", detail: "Original implementation", value: first("fused"), color: "accent"},
  {label: "Native + window", detail: "Earlier native path", value: optimized("final-native.json"), color: "accent"},
  {label: "Compiled native", detail: "Earlier compiled path", value: optimized("final-compiled.json"), color: "accent"},
];
export const originalSpeedup = warm[0].value / warm[2].value;

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? required(sorted[middle]) : (required(sorted[middle - 1]) + required(sorted[middle])) / 2;
}

export const publicWorkloads = Object.entries(packaged.summary).map(([workload, summary]) => {
  const source = workload === "changing" ? packaged.changing.rows : packaged.fixed.rows;
  const variants = ["balanced", "fast", "retained"].map(variant => {
    const rounds = source.filter(row => row.case === workload && row.variant === variant);
    const samples = rounds.flatMap(row => row.samples_ms);
    const p50_ms = median(samples);
    const reported = required(Object.entries(summary).find(([key]) => key === variant))[1];
    if (typeof reported !== "object" || reported === null || !("p50_ms" in reported) || reported.p50_ms !== p50_ms || reported.samples !== samples.length) {
      throw new Error(`Raw package samples disagree with summary: ${workload}/${variant}`);
    }
    return {variant, p50_ms, samples: samples.length, rounds: rounds.length};
  });
  return {workload, balanced: variants[0], fast: variants[1], retained: variants[2], reduction_percent: 100 * (1 - variants[1].p50_ms / variants[0].p50_ms)};
});
const publicShort = required(publicWorkloads.find(row => row.workload === "1-short"));
if (!packaged.parity.all_exact || !packaged.parity.concurrent.all_exact || !packaged.lifecycle.all_owners_released) {
  throw new Error("The public-mode chart requires a passing package validation.");
}
export const publicModes = [
  {label: "Balanced", detail: "Default public mode", value: publicShort.balanced.p50_ms, color: "neutral"},
  {label: "Fast", detail: "Compiled SM120 configuration", value: publicShort.fast.p50_ms, color: "accent"},
];
export const publicReduction = publicShort.reduction_percent;
export const publicMeasurement = {
  date: packaged.metadata.created_utc,
  method: packaged.metadata.method,
  samples_per_mode: publicShort.fast.samples,
  rounds: publicShort.fast.rounds,
  parity_requests: packaged.parity.requests,
  parity_decisions: packaged.parity.decisions,
  validation_scope: packaged.metadata.validation_scope,
};

const latestSummary = required(frontier.rows.find(row => row.policy === latestFull.policy));
if (frontier.recommended_variant !== latestFull.policy || !latestSummary.accepted_after_both_suites) {
  throw new Error("Update the latest chart to the accepted recommended result.");
}
export const latestWorkloads = ["1-short", "1-long", "16-short"].map(workload => {
  const variants = ["native", latestFull.policy].map(variant => {
    const rounds = latestFull.timings.rows.filter(row => row.case === workload && row.variant === variant);
    const samples = rounds.flatMap(row => row.samples_ms);
    return {variant, p50_ms: median(samples), samples: samples.length, rounds: rounds.length};
  });
  return {workload, baseline: variants[0], optimized: variants[1], reduction_percent: 100 * (1 - variants[1].p50_ms / variants[0].p50_ms)};
});
const latestShort = latestWorkloads[0];
if (latestShort.optimized.p50_ms !== frontier.best_measured_accepted_ms || latestShort.baseline.p50_ms !== latestSummary.timings["1-short"].native.p50_ms) {
  throw new Error("Raw latest timing samples disagree with the frontier summary.");
}
export const latest = [
  {label: "Native baseline", detail: "Same paired experiment", value: latestShort.baseline.p50_ms, color: "neutral"},
  {label: "Optimized BF16", detail: "Retained before packaging", value: latestShort.optimized.p50_ms, color: "accent"},
];
export const latestReduction = latestShort.reduction_percent;
export const latestMeasurement = {
  date: latestFull.metadata.created_utc,
  method: latestFull.metadata.method,
  samples_per_mode: latestShort.baseline.samples,
  rounds: latestShort.baseline.rounds,
  validation: latestFull.metadata.validation_scope,
  original_decisions: latestSummary.original_decisions,
  original_exact: latestSummary.original_exact,
  holdout: latestSummary.holdout,
};
export const fusion = required(latency.fusion.rows.find(row => row.case === "16-long")).variants;
export const fusionReduction = 100 * (1 - fusion.fusion.p50_ms / fusion["native-window"].p50_ms);
export const concurrent = ["serial-current", "streams-4"].map(variant => required(serving.rows.find(
  row => row.variant === variant && row.profile === "1-short" && row.concurrency === 4,
)));
export const throughputGain = concurrent[1].requests_per_second / concurrent[0].requests_per_second;

function startupSeconds(mode: string, cache: string) {
  return required(startup.rows.find(row => row.mode === mode && row.cache_state === cache)).startup_median_ms / 1000;
}
export const cold = [
  {label: "torch.compile", detail: "Fresh compiler caches", value: startupSeconds("compile", "fresh"), color: "neutral"},
  {label: "torch.compile", detail: "Reused compiler caches", value: startupSeconds("compile", "reused"), color: "neutral"},
  {label: "Native + window", detail: "No full-model compilation", value: startupSeconds("native", "fresh"), color: "neutral"},
  {label: "AOT deployment", detail: "Prebuilt, fixed shapes", value: startupSeconds("aot-fast", "fresh"), color: "accent"},
];

// This also records unrounded values alongside the exported images for review.
export const chartSources = {
  "public-modes": ["results/fast-package.json"],
  "warm-latency": ["results/benchmark.json", "results/upstream-fast.json", "results/native-optimizations/summary.json"],
  "latest-paired": ["results/frontier/summary.json", latestSummary.source, "results/frontier/holdout-native-format.json"],
  "experimental-gains": ["results/latency-optimizations/summary.json", "results/latency-optimizations/serving/confirmation-summary.json"],
  startup: ["results/latency-optimizations/aot/final-offline/matrix-summary.json"],
};
export const chartData = {publicModes, publicReduction, publicWorkloads, publicMeasurement, warm, originalSpeedup, latest, latestReduction, latestWorkloads, latestMeasurement, fusion, fusionReduction, concurrent, throughputGain, cold};
