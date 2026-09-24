// Read the recorded results directly. No benchmark values are copied into the renderer.
import original from "../../../results/benchmark.json";
import upstreamFast from "../../../results/upstream-fast.json";
import native from "../../../results/native-optimizations/summary.json";
import latency from "../../../results/latency-optimizations/summary.json";
import serving from "../../../results/latency-optimizations/serving/confirmation-summary.json";
import startup from "../../../results/latency-optimizations/aot/final-offline/matrix-summary.json";

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
  {label: "This engine", detail: "Published default", value: first("fused"), color: "accent"},
  {label: "Native + window", detail: "Opt-in experiment", value: optimized("final-native.json"), color: "accent"},
  {label: "Compiled native", detail: "Opt-in experiment", value: optimized("final-compiled.json"), color: "accent"},
];
export const originalSpeedup = warm[0].value / warm[2].value;
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
export const chartData = {warm, originalSpeedup, fusion, fusionReduction, concurrent, throughputGain, cold};
