import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/history"
OUT.mkdir(exist_ok=True)
SOURCES = {}
TABLES = []
CASES = ["1-short", "16-short", "1-long", "16-long"]


def read(name):
    path = ROOT / "results" / name
    SOURCES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def fmt(n, digits=3):
    return "Not measured" if n is None else f"{n:,.{digits}f}"


def add(key, title, headers, rows, caption, notes=""):
    assert rows and all(len(r) == len(headers) for r in rows), key
    TABLES.append(
        {
            "id": key,
            "title": title,
            "headers": headers,
            "rows": rows,
            "caption": caption,
            "notes": notes,
        }
    )


def warm_values(rows, variant=None):
    out = {}
    for r in rows:
        if variant is not None and r.get("backend", r.get("variant")) != variant:
            continue
        case = r.get("case") or f"{r['questions']}-{r['state_length']}"
        out[case] = r["p50_ms"]
    return [fmt(out.get(c)) for c in CASES]


def parity(v):
    if not v:
        return "Screen only"
    agreement, decisions = v.get("agreement"), v.get("decisions")
    text = (
        f"{agreement}/{decisions} choices" if decisions is not None else "Limited check"
    )
    if v.get("exact_logits_all"):
        return text + "; exact choice logits"
    if v.get("max_probability_error") is not None:
        text += f"; probability drift {v['max_probability_error']:.5f}"
    return text


base = read("benchmark.json")
fast = read("upstream-fast.json")
comp = read("upstream-compile.json")
ablation = read("ablation.json")
host = read("native-optimizations/host/summary.json")
v1 = read("native-optimizations/summary.json")
v2 = read("latency-optimizations/summary.json")
aot = read("latency-optimizations/aot/index.json")
comparison = read("rtx-a6000/summary.json")
progression = []


def progress(stage, label, vals, status, source):
    progression.append([stage, label, *vals, status, source])


progress(
    "Baseline",
    "Upstream default",
    warm_values(base["rows"], "upstream"),
    "Default SDK",
    "benchmark.json",
)
progress(
    "Baseline",
    "Upstream fast=True",
    warm_values(fast["rows"]),
    "Separate upstream option",
    "upstream-fast.json",
)
progress(
    "Baseline",
    "Upstream compile=True",
    warm_values(comp["rows"]),
    "Separate upstream option",
    "upstream-compile.json",
)
progress(
    "Ablation",
    "Our optimized model, graph replay disabled",
    warm_values(ablation["rows"], "fused_without_graph"),
    "Isolated diagnostic",
    "ablation.json",
)
progress(
    "First engine",
    "Resident BF16 + graph replay + fused RoPE + selective head",
    warm_values(base["rows"], "fused"),
    "Original validated engine",
    "benchmark.json",
)
for variant, label in [
    ("cpp-graph-io", "C++ request packing and graph I/O"),
    ("cpp-rust-graph-io", "C++ I/O + direct Rust tokenizer calls"),
]:
    vals = [
        fmt(
            next(r for r in host["timings"] if r["case"] == c)["variants"][variant][
                "pooled_p50_ms"
            ]
        )
        for c in CASES
    ]
    progress(
        "Round 1 host",
        label,
        vals,
        "Exact fixture outputs; host changes only",
        "native-optimizations/host/summary.json",
    )
names = {
    "combined-native.json": (
        "Round 1",
        "Native CUDA normalization + GEGLU + native host",
    ),
    "combined-window.json": ("Round 1", "Add exact window attention"),
    "final-native.json": (
        "Round 1 final",
        "Vectorized normalization + exact windows + native host",
    ),
    "final-compiled.json": (
        "Round 1 branch",
        "Full-model compiled engine + native host",
    ),
    "final-autotuned.json": ("Round 1 branch", "Matrix autotuning + native host"),
}
for name, (stage, label) in names.items():
    c = next(r for r in v1["comparisons"] if r["file"] == name)
    rows = [r for r in c["rows"] if r["variant"] != "baseline"]
    status = (
        "Opt-in; 208/208 choices, nonexact logits"
        if "autotuned" in name
        else "Exact outputs on 66-request fixture"
    )
    progress(stage, label, warm_values(rows), status, "native-optimizations/" + name)
vals = [
    fmt(
        next(r for r in v2["fusion"]["rows"] if r["case"] == c)["variants"]["fusion"][
            "p50_ms"
        ]
    )
    for c in CASES
]
progress(
    "Round 2",
    "Projection fusion: Wi + GEGLU and QKV + RoPE",
    vals,
    "Exact fixture outputs; adds 301 MB weights",
    "latency-optimizations/fusion/best.json",
)
aot_row = next(r for r in aot["startup_rows"] if r["mode"] == "aot-fast")
progress(
    "Round 2 branch",
    "AOT package + checked loader + lightweight tokenizer",
    [
        fmt(aot_row["warm_predict_median_ms"]),
        "Not supported",
        "See single-run smoke",
        "Not supported",
    ],
    "Two fixed batch-one shapes",
    "latency-optimizations/aot/index.json",
)
add(
    "warm-progression",
    "Warm request latency across development stages",
    [
        "Stage",
        "Implementation",
        "1 short, ms",
        "16 short, ms",
        "1 long, ms",
        "16 long, ms",
        "Status",
        "Source",
    ],
    progression,
    "RTX 5070 Ti. Median full predict, including preparation, transfers, inference and formatting. Serial, warmed requests. Startup and HTTP excluded. Saved runs from 23–24 September 2026.",
    "This is a history of separate runs, not one paired experiment. Branches are alternatives, not cumulative steps. Smaller differences can be run-to-run noise. Sixteen-question columns are total request latency. The AOT value is the median of three process-level warm medians; the other main rows pool request samples within each recorded run. See paired comparisons for measured incremental gains.",
)

paired = []
for c in v1["comparisons"]:
    for r in c["rows"]:
        if r["variant"] == "baseline":
            continue
        b = next(
            b
            for b in c["rows"]
            if b["case"] == r["case"] and b["variant"] == "baseline"
        )
        paired.append(
            [
                c["file"],
                r["case"],
                fmt(b["p50_ms"]),
                fmt(r["p50_ms"]),
                f"{r['latency_reduction_percent']:.2f}%",
                str(r["sample_count"]),
                "native-optimizations/" + c["file"],
            ]
        )
for r in v2["fusion"]["rows"]:
    b, c = r["variants"]["native-window"], r["variants"]["fusion"]
    paired.append(
        [
            "Round 2 projection fusion",
            r["case"],
            fmt(b["p50_ms"]),
            fmt(c["p50_ms"]),
            f"{100 * (1 - c['p50_ms'] / b['p50_ms']):.2f}%",
            str(c["samples"]),
            "latency-optimizations/fusion/best.json",
        ]
    )
direct = read("native-optimizations/compiler/direct-native-vs-compiled.json")[
    "timings"
]["rows"]
for case in dict.fromkeys(r["case"] for r in direct):
    values = {
        v: median(
            [
                s
                for r in direct
                if r["variant"] == v and r["case"] == case
                for s in r["samples_ms"]
            ]
        )
        for v in ["native-window", "compiled"]
    }
    b, c = values["native-window"], values["compiled"]
    paired.append(
        [
            "Native versus full-model compiled",
            case,
            fmt(b),
            fmt(c),
            f"{100 * (1 - c / b):.2f}%",
            "150",
            "native-optimizations/compiler/direct-native-vs-compiled.json",
        ]
    )
add(
    "paired-gains",
    "Paired comparisons and extended workloads",
    [
        "Comparison",
        "Workload",
        "Baseline p50, ms",
        "Candidate p50, ms",
        "Latency reduction",
        "Samples / variant",
        "Source",
    ],
    paired,
    "Randomized blocks, both engines resident on RTX 5070 Ti. Medians pooled from full-request samples. A negative reduction means the candidate was slower.",
    "Round 1 baseline is the original published engine. Round 2 baseline is the previous native-window engine. The direct compile comparison uses native-window as its baseline. Autotuning preserves selected decisions on the fixture but changes logits.",
)

full = []
for name, obj in [
    ("benchmark.json", base),
    ("upstream-fast.json", fast),
    ("upstream-compile.json", comp),
    ("ablation.json", ablation),
]:
    for r in obj["rows"]:
        full.append(
            [
                r["backend"],
                f"{r['questions']}-{r['state_length']}",
                fmt(r["p50_ms"]),
                fmt(r["p95_ms"]),
                fmt(r["mean_ms"]),
                str(len(r["samples_ms"])),
                name,
            ]
        )
add(
    "original-workloads",
    "Original engine, upstream modes and graph ablation",
    ["Backend", "Workload", "p50, ms", "p95, ms", "Mean, ms", "Samples", "Source"],
    full,
    "RTX 5070 Ti, warm serial predict with no HTTP. Original benchmark: 100 samples per case. Upstream fast and compile: 50. Graph ablation: three randomized blocks of 50.",
    "The graph ablation holds the optimized forward, buffers and preparation fixed. It is an order-dependent software comparison, not a Blackwell hardware attribution.",
)

host_rows = []
for r in host["timings"]:
    for name, v in r["variants"].items():
        host_rows.append(
            [
                r["case"],
                name,
                fmt(v["pooled_p50_ms"]),
                f"{v['latency_reduction_percent']:.2f}%",
                str(v["sample_count"]),
                "native-optimizations/host/summary.json",
            ]
        )
add(
    "host",
    "C++ and Rust host-path contributions",
    ["Workload", "Variant", "p50, ms", "Latency reduction", "Samples", "Source"],
    host_rows,
    "RTX 5070 Ti, five randomized blocks of 30 warm full requests. Model math unchanged.",
    "Rust means direct access to the existing Tokenizers backend. C++ handles packing and CUDA launch/copies. Python still constructs sequences and formats responses; this is not a full language rewrite.",
)

http = []
for r in read("benchmark-http.json")["rows"]:
    http.append(
        [
            "Original comparison",
            r["backend"],
            f"{r['questions']}-{r['state_length']}",
            fmt(r["p50_ms"]),
            "100",
            "benchmark-http.json",
        ]
    )
for r in host["http"]:
    for name, v in r["variants"].items():
        http.append(
            [
                "Host-path comparison",
                name,
                r["case"],
                fmt(v["pooled_p50_ms"]),
                str(v["samples"]),
                "native-optimizations/host/summary.json",
            ]
        )
add(
    "http",
    "Matched localhost HTTP latency",
    ["Experiment", "Backend", "Workload", "p50, ms", "Samples", "Source"],
    http,
    "RTX 5070 Ti. Serial localhost HTTP/1.1 keep-alive through matched FastAPI wrappers. Includes HTTP and JSON; excludes model/server startup and warmup.",
    "Original HTTP run used CUDA 13.0; the later host-path run used CUDA 13.2. These are separate paired experiments. Neither measures remote-network latency or maximum concurrent HTTP capacity.",
)

startup = []
for obj, name in [
    (base, "benchmark.json"),
    (fast, "upstream-fast.json"),
    (comp, "upstream-compile.json"),
]:
    for r in obj["rows"]:
        if r["questions"] == 1 and r["state_length"] == "short":
            startup.append(
                [
                    "RTX 5070 Ti original",
                    r["backend"],
                    "First predict after loading; CUDA already initialized",
                    fmt(r["first_request_ms"] / 1000),
                    "1",
                    "Existing caches may be reused",
                    name,
                ]
            )
for name in [
    "clean-startup-native-window.json",
    "clean-startup-cold.json",
    "clean-startup-warm.json",
]:
    r = read("native-optimizations/compiler/" + name)
    for field, boundary in [
        ("first_predict_ms", "First predict after setup"),
        (
            "python_entry_to_first_response_ms",
            "Python entry to first complete response",
        ),
    ]:
        startup.append(
            [
                "RTX 5070 Ti round 1",
                r["mode"],
                boundary,
                fmt(r[field] / 1000),
                "1",
                r["cache_state"],
                "native-optimizations/compiler/" + name,
            ]
        )
for r in aot["startup_rows"]:
    startup.append(
        [
            "RTX 5070 Ti round 2",
            r["mode"] + ", " + r["cache_state"] + " caches",
            "Module entry to first complete response",
            fmt(r["startup_median_ms"] / 1000),
            str(r["processes"]),
            f"Observed range {r['startup_min_ms'] / 1000:.3f} to {r['startup_max_ms'] / 1000:.3f} s",
            "latency-optimizations/aot/final-offline/matrix-summary.json",
        ]
    )
for profile in comparison["profiles"]:
    r = next(r for r in profile["cases"] if r["case"] == "1-short")
    startup.append(
        [
            "RTX A6000",
            profile["profile"],
            "First predict after loading",
            fmt(r["first_call_ms"] / 1000),
            "1",
            "Existing model and compiler caches; not clean install",
            "rtx-a6000/" + profile["source_file"],
        ]
    )
add(
    "startup",
    "Startup and first-use timings with explicit boundaries",
    [
        "Device / round",
        "Mode",
        "Timer boundary",
        "Seconds",
        "Processes",
        "Cache conditions",
        "Source",
    ],
    startup,
    "All rows use one short question. Single-process rows are observations; round 2 reports medians of three fresh processes.",
    "These are seconds, not warm-request milliseconds. Do not compare after-loading figures with process-entry figures. Final round 2 is fully offline with warm OS file caches and prebuilt native extensions; AOT export/package construction is excluded.",
)

aot_rows = [
    [
        r["mode"],
        r["cache_state"],
        "1-short",
        fmt(r["startup_median_ms"] / 1000),
        fmt(r["warm_predict_median_ms"]),
        str(r["processes"]),
        "latency-optimizations/aot/index.json",
    ]
    for r in aot["startup_rows"]
]
smoke = aot["long_request_smoke"]
aot_rows.append(
    [
        "aot-fast, smoke only",
        "fresh",
        "1-long",
        fmt(smoke["entry_to_first_response_ms"] / 1000),
        fmt(smoke["warm_predict_p50_ms"]),
        "1",
        "latency-optimizations/aot/index.json",
    ]
)
add(
    "aot",
    "Final precompiled deployment and warm-request results",
    [
        "Mode",
        "Compiler caches",
        "Workload",
        "First response, s",
        "Warm p50, ms",
        "Processes",
        "Source",
    ],
    aot_rows,
    "RTX 5070 Ti round 2. Offline, warm OS file caches, existing model files and native extensions. First-response timing starts at module entry. Warm p50 is the median of the per-process warm p50 values for the repeated short-request conditions.",
    "AOT supports only batch one, four options, 64 or 512 tokens and unmasked global attention. The long row is one smoke check, not a repeated benchmark. Packages are about 958 MB per shape; observed export took 5–6 s and compilation 24–26 s with existing development caches. Those build times are excluded from startup. Runtime still requires Python Torch and custom operations.",
)

comparison_rows = []
for profile in comparison["profiles"]:
    vals = warm_values(profile["cases"])
    comparison_rows.append(
        [
            profile["profile"],
            *vals,
            f"{sum(c['matching_top_options'] for c in profile['cases'])}/{sum(c['decisions'] for c in profile['cases'])}",
            "rtx-a6000/" + profile["source_file"],
        ]
    )
add(
    "rtx-a6000",
    "All implementations measured on RTX A6000",
    [
        "Implementation",
        "1 short, ms",
        "16 short, ms",
        "1 long, ms",
        "16 long, ms",
        "Reference choices",
        "Source",
    ],
    comparison_rows,
    "RTX A6000, Ampere SM86, with Threadripper PRO 7995WX. Twenty warm serial full requests per case, five warmups, no HTTP or startup. CUDA 13.0.",
    "The blackwell profile is this engine with a benchmark-only hardware-guard override. All rows ran on the RTX A6000 system. Precision, CPU threads and pool behavior differ by profile. Four synthetic requests are a limited parity check, not labeled accuracy. These numbers must not be ranked against the RTX 5070 Ti results as if hardware and host were held fixed.",
)

serving = []
for r in sorted(
    v2["serving"]["rows"], key=lambda r: (r["profile"], r["concurrency"], r["variant"])
):
    serving.append(
        [
            r["profile"],
            str(r["concurrency"]),
            r["variant"],
            fmt(r["requests_per_second"], 2),
            fmt(r["p50_ms"]),
            fmt(r["p95_ms"]),
            f"{r['throughput_ratio_vs_serial']:.3f}×",
            str(r["requests"]),
            "latency-optimizations/serving/confirmation-summary.json",
        ]
    )
add(
    "serving",
    "Concurrent serving throughput and queue-inclusive latency",
    [
        "Workload",
        "Callers",
        "Policy",
        "Requests/s",
        "p50, ms",
        "p95, ms",
        "Throughput ratio",
        "Requests",
        "Source",
    ],
    serving,
    "RTX 5070 Ti round 2. Three randomized blocks of 64 requests per configuration, 12,288 total. Local Python service, no HTTP. Latency includes preparation, queueing, copies, inference and formatting.",
    "All measured formatted responses matched and all warmed graph caches had zero misses. This tests native-window with independent stream lanes, not projection fusion combined with streams. Priority policies can increase bulk tail latency.",
)

padding = []
for label, field, source in [
    ("Exact batch", "rows", "padding/batch-exact-confirmed.json"),
    ("32-token buckets", "sequence_rows", "padding/sequence-32.json"),
]:
    for r in v2["padding"][field]:
        b, c = r["variants"]["baseline"], r["variants"]["candidate"]
        padding.append(
            [
                label,
                r["case"],
                fmt(b["p50_ms"]),
                fmt(c["p50_ms"]),
                f"{100 * (1 - c['p50_ms'] / b['p50_ms']):.2f}%",
                "Rejected as default",
                "latency-optimizations/" + source,
            ]
        )
add(
    "padding",
    "Padding policies: warm gains that failed expanded parity",
    [
        "Policy",
        "Workload",
        "Baseline p50, ms",
        "Candidate p50, ms",
        "Latency reduction",
        "Disposition",
        "Source",
    ],
    padding,
    "RTX 5070 Ti round 2, serial full predict with warm graphs. Exact-batch confirmation uses five randomized blocks and 150 samples per variant/workload.",
    "Exact batch changed 12/946 choices; preserving attention-mask dispatch reduced this to 5/946. Sequence buckets matched 208 selected choices but exceeded the 0.01 probability-drift bound. Cache churn makes exact batches much slower when shapes do not fit: 52.826 versus 6.181 ms p50 in the 1–10-question cycle.",
)

screens = []
for folder in ["kernels", "compiler", "precision"]:
    for path in sorted((ROOT / "results/native-optimizations" / folder).glob("*.json")):
        name = str(path.relative_to(ROOT / "results"))
        obj = json.loads(path.read_text())
        if (
            not isinstance(obj, dict)
            or obj.get("status") != "complete"
            or not isinstance(obj.get("rows"), list)
        ):
            continue
        rows = obj["rows"]
        if not rows or not all("p50_ms" in r and "case" in r for r in rows):
            continue
        read(name)
        screens.append(
            [
                folder,
                obj.get("kind", obj.get("variant", path.stem)),
                *warm_values(rows),
                parity(obj.get("validation")),
                name,
            ]
        )
smooth = read("native-optimizations/precision/smooth/summary.json")
for r in smooth["rows"]:
    screens.append(
        [
            "precision",
            f"Channel-balanced FP8, alpha {r['alpha']}",
            *[fmt(r["p50_ms"].get(c)) for c in CASES],
            f"Heldout {r['heldout']['agreement']}/{r['heldout']['decisions']} choices; rejected",
            "native-optimizations/precision/smooth/summary.json",
        ]
    )
add(
    "screens",
    "Standalone kernel, compiler and precision screens",
    [
        "Family",
        "Candidate",
        "1 short, ms",
        "16 short, ms",
        "1 long, ms",
        "16 long, ms",
        "Numerical check",
        "Source",
    ],
    screens,
    "RTX 5070 Ti round 1. Completed standalone warm full-request screens. These precede or complement the randomized combined confirmations; they are not isolated per-operation timings.",
    "Retain numerical status when reading these timings. Full-model compile variants with changed rounding, FP8 quantization and FlexAttention are not accepted replacements. Blank measurements were never taken. Handwritten PTX warp shuffle and its CUDA intrinsic compiled to identical 504-instruction machine code, so there was no separate PTX gain.",
)

fusion = read("latency-optimizations/fusion/summary.json")
add(
    "fusion",
    "Projection-fusion configuration comparisons",
    [
        "Configuration",
        "Workload",
        "Baseline p50, ms",
        "Fused p50, ms",
        "Latency reduction",
        "Samples / variant",
        "Source",
    ],
    [
        [
            r["report"],
            r["case"],
            fmt(r["baseline_p50_ms"]),
            fmt(r["fusion_p50_ms"]),
            f"{r['latency_reduction_pct']:.2f}%",
            str(r["samples_per_variant"]),
            "latency-optimizations/fusion/" + r["report"],
        ]
        for r in fusion["comparisons"]
    ],
    "RTX 5070 Ti round 2. Three randomized blocks per configuration. Each number is median full-request time. Final best.json is the accepted selector.",
    "The final selector avoids small shapes without a demonstrated gain and falls back above 8,192 rows. Exact intermediate rounding and a GELU lookup table preserve the tested outputs.",
)

micro = read("native-optimizations/host/components.json")
component_rows = []
for r in micro["rows"]:
    component_rows.append(
        [
            r["case"],
            r["mode"],
            *[fmt(r[k]["p50_ms"] * 1000, 2) for k in ["pack", "prepare", "format"]],
            fmt(r["run_prepared"]["p50_ms"]),
            fmt(r["predict"]["p50_ms"]),
            "native-optimizations/host/components.json",
        ]
    )
add(
    "components",
    "Exploratory host component timings",
    [
        "Workload",
        "Mode",
        "Packing, µs",
        "Preparation, µs",
        "Formatting, µs",
        "Prepared execution, ms",
        "Full predict, ms",
        "Source",
    ],
    component_rows,
    "RTX 5070 Ti round 1. Sequential exploratory mode sweep. Component microbenchmarks use different iteration counts from full predict.",
    "Microseconds and milliseconds are explicitly separated. Do not add component medians or present a packing speedup as a whole-model speedup. Prefer randomized host confirmations for final request-level claims.",
)

screen = read("latency-optimizations/serving/screen-summary.json")
add(
    "serving-screen",
    "Exploratory serving and microbatch sweep",
    [
        "Workload",
        "Callers",
        "Policy",
        "Requests/s",
        "p50, ms",
        "p95, ms",
        "Requests",
        "Source",
    ],
    [
        [
            r["profile"],
            str(r["concurrency"]),
            r["variant"],
            fmt(r["requests_per_second"], 2),
            fmt(r["p50_ms"]),
            fmt(r["p95_ms"]),
            str(r["requests"]),
            "latency-optimizations/serving/screen-summary.json",
        ]
        for r in screen["rows"]
    ],
    "RTX 5070 Ti round 2. Completed screen after the CUDA graph lifetime fix. Single measured block per configuration. Use the repeated serving confirmation for accepted stream-policy results.",
    "Microbatching passed the initial fixture but changed 14/245 decisions in the expanded deduplicated check, with maximum probability drift 0.019893. Its throughput here is a rejected-candidate result, not a recommended engine claim.",
)

data = {
    "title": "Laya benchmark history",
    "date": "23–24 September 2026",
    "tables": TABLES,
    "scope": "Historical request-level benchmarks, completed candidate screens and host component measurements. Failed/incomplete runs, compiler traces and individual tile-tuning microbenchmarks remain in their raw directories and are not treated as benchmark results.",
    "sources": [
        {"path": name, "sha256": digest} for name, digest in sorted(SOURCES.items())
    ],
}
(OUT / "tables.json").write_text(json.dumps(data, indent=2) + "\n")
for t in TABLES:
    with (OUT / (t["id"] + ".csv")).open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(t["headers"])
        writer.writerows(t["rows"])
print(
    json.dumps(
        {
            "tables": len(TABLES),
            "rows": sum(len(t["rows"]) for t in TABLES),
            "sources": len(SOURCES),
        }
    )
)
