import { useLayoutEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import { paintColumn } from "./components/dither-kit/dither-paint";
import type { Rgb } from "./components/dither-kit/palette";
import { warm, originalSpeedup, fusion, fusionReduction, concurrent, throughputGain, cold, chartData } from "./data";
import "./style.css";

const theme = new URLSearchParams(location.search).get("theme") === "dark" ? "dark" : "light";
document.documentElement.dataset.theme = theme;
const palette: Record<string, Rgb> = theme === "dark"
  ? {accent: [106, 230, 156], neutral: [157, 178, 169]}
  : {accent: [11, 121, 76], neutral: [104, 123, 113]};
const format = (value: number, digits = 2) => value.toFixed(digits);

// Dither Kit supplies the ordered-dither paint engine. Our layout adds static,
// horizontal bars and publication labels, without animation or hover effects.
function DitherBar({value, max, color, vertical = false}: {
  value: number; max: number; color: string; vertical?: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useLayoutEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const {width, height} = canvas.getBoundingClientRect();
    // Two CSS pixels per dither cell, independent of screenshot DPR.
    canvas.width = Math.round(width / 2);
    canvas.height = Math.round(height / 2);
    const span = vertical ? canvas.height : canvas.width;
    const thickness = vertical ? canvas.width : canvas.height;
    const length = (span - 1) * value / max;
    const fill = palette[color];
    if (!fill || !Number.isFinite(value) || value < 0 || value > max) {
      throw new Error("Invalid chart value or palette key.");
    }
    if (!vertical) {
      ctx.translate(0, canvas.height);
      ctx.rotate(-Math.PI / 2);
    }
    // Vertical bars grow from the bottom; rotating turns this into a left origin.
    // Mirror the horizontal coordinate so its zero baseline stays at the left.
    if (!vertical) { ctx.translate(0, span); ctx.scale(1, -1); }
    for (let x = 0; x < thickness; x++) {
      paintColumn(ctx, x, span - 1 - length, span - 1, {fill, line: fill, star: fill}, {
        variant: "gradient", intensity: 0, dim: 1, stacked: false,
      });
    }
    canvas.dataset.ready = "true";
  }, [value, max, color, vertical]);
  return <canvas ref={ref} className={vertical ? "dither vertical" : "dither"} aria-hidden="true" />;
}

function Masthead({index, label}: {index: string; label: string}) {
  return <div className="masthead"><span className="brand"><i /> LAYA / PERFORMANCE</span><span>{index} &nbsp; {label}</span></div>;
}
function Footer({children}: {children: React.ReactNode}) {
  return <footer><div>{children}</div><span className="hardware">RTX 5070 Ti · 16 GB<br />BLACKWELL / SM120</span></footer>;
}
type Row = {label: string; detail: string; value: number; color: string};
function HorizontalBars({rows, max, ticks, unit}: {rows: Row[]; max: number; ticks: number[]; unit: string}) {
  return <div className="horizontal-chart">
    <div className="axis-row"><span /><div className="axis">{ticks.map(tick => <span key={tick} style={{left: `${100 * tick / max}%`}}>{tick}</span>)}</div><span className="axis-unit">{unit}</span></div>
    {rows.map((row, index) => <div className="chart-row" key={index}>
      <div className="row-label"><strong>{row.label}</strong><span>{row.detail}</span></div>
      <div className="track">
        {ticks.map(tick => <i className="gridline" key={tick} style={{left: `${100 * tick / max}%`}} />)}
        <DitherBar value={row.value} max={max} color={row.color} />
      </div>
      <div className={`row-value ${row.color}`}>{format(row.value)}<small>{unit}</small></div>
    </div>)}
  </div>;
}

function Pair({values, labels, max, ticks, unit, digits}: {
  values: number[]; labels: string[]; max: number; ticks: number[]; unit: string; digits: number;
}) {
  return <div className="pair-chart">
    <div className="pair-plot">
      {ticks.map(tick => <div className="pair-grid" key={tick} style={{bottom: `${100 * tick / max}%`}}><span>{tick}</span></div>)}
      {values.map((value, index) => <div className="pair-bar" key={index}>
        <div className={`pair-value ${index ? "accent" : "neutral"}`} style={{bottom: `calc(${100 * value / max}% + 10px)`}}>{format(value, digits)}<small>{unit}</small></div>
        <DitherBar value={value} max={max} color={index ? "accent" : "neutral"} vertical />
      </div>)}
    </div>
    <div className="pair-labels">{labels.map(label => <span key={label}>{label}</span>)}</div>
  </div>;
}

function App() {
  return <main>
    <section className="sheet" id="warm-latency">
      <Masthead index="01" label="WARM LATENCY" />
      <header><div><h1>Less waiting per request.</h1><p>One short question · full request p50 · lower is better</p></div>
        <div className="headline-stat"><strong>{format(originalSpeedup, 1)}×</strong><span>original engine speedup<br />vs. upstream default</span></div>
      </header>
      <HorizontalBars rows={warm} max={25} ticks={[0, 5, 10, 15, 20, 25]} unit="ms" />
      <Footer>Historical runs on the same GPU, not one paired experiment.<br />Includes tokenization, transfers, inference and formatting.<br />Excludes loading, warmup and HTTP. Later modes are opt-in.</Footer>
    </section>

    <section className="sheet" id="experimental-gains">
      <Masthead index="02" label="OPT-IN EXPERIMENTS" />
      <header><div><h1>More work. Less time.</h1><p>Two separate comparisons against the optimized native engine</p></div></header>
      <div className="panels">
        <article>
          <div className="panel-kicker">LONG BATCHES</div>
          <h2>{format(fusionReduction, 1)}% <span>lower latency</span></h2>
          <p>16 long questions · p50 · lower is better</p>
          <Pair values={[fusion["native-window"].p50_ms, fusion.fusion.p50_ms]} labels={["Native + window", "+ Projection fusion"]} max={120} ticks={[0, 40, 80, 120]} unit="ms" digits={2} />
          <div className="panel-note">90 calls per mode · randomized blocks<br />About 301 MB of extra packed GPU weights</div>
        </article>
        <article>
          <div className="panel-kicker">CONCURRENT REQUESTS</div>
          <h2>{format(throughputGain)}× <span>the throughput</span></h2>
          <p>4 callers · one short question · higher is better</p>
          <Pair values={concurrent.map(row => row.requests_per_second)} labels={["Serialized GPU", "4 CUDA streams"]} max={1200} ticks={[0, 400, 800, 1200]} unit="req/s" digits={0} />
          <div className="panel-note">p95 latency: {format(concurrent[0].p95_ms)} → {format(concurrent[1].p95_ms)} ms<br />Shared weights · extra graph buffers per stream</div>
        </article>
      </div>
      <Footer>Warm, in-process calls. No HTTP or startup on either side.<br />Serving latency includes queueing. Gains measured separately;<br />do not multiply them. Exact outputs on the tested fixtures.</Footer>
    </section>

    <section className="sheet" id="startup">
      <Masthead index="03" label="EXPERIMENTAL STARTUP" />
      <header><div><h1>Get to the first response sooner.</h1><p>Module entry to completed response · median · lower is better</p></div>
        <div className="headline-stat"><strong>{format(cold[3].value)}<small>s</small></strong><span>precompiled AOT<br />deployment path</span></div>
      </header>
      <HorizontalBars rows={cold} max={25} ticks={[0, 5, 10, 15, 20, 25]} unit="s" />
      <div className="startup-scope">AOT supports two fixed batch-one shapes. About 958 MB per artifact.<br />The measured AOT path also changes the loader and tokenizer initialization.</div>
      <Footer>3 fresh processes per mode · offline · warm OS file caches.<br />Includes imports, model loading, graph capture and first request.<br />Prebuilt native extensions. Artifact build, download and HTTP excluded.</Footer>
    </section>
  </main>;
}

// Exported for the renderer's provenance record, never substituted for source data.
Object.assign(window, {chartData});
createRoot(document.getElementById("root")!).render(<App />);
