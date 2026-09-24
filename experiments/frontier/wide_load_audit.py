"""Compile-only SM120 vector-memory audit. This script never opens a CUDA device."""

import collections
import fcntl
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-wide-load"
CUDA = Path("/usr/local/cuda")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {
        "command": [str(x) for x in args],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def instructions(sass):
    return dict(
        collections.Counter(
            re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)", sass)
        )
    )


def ptx(body):
    return (
        """.version 8.8
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 src, .param .u64 dst) {
 .reg .b64 a, b, q0, q1, q2, q3;
 .reg .b32 r<8>, s;
 .shared .align 32 .b8 scratch[32];
 .reg .b64 scratch64;
 ld.param.u64 a, [src];
 ld.param.u64 b, [dst];
 mov.u64 scratch64, scratch;
 cvta.to.shared.u64 scratch64, scratch64;
 cvt.u32.u64 s, scratch64;
"""
        + body
        + "\n ret;\n}\n"
    )


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    vector = "{r0, r1, r2, r3, r4, r5, r6, r7}"
    half = "{r0, r1, r2, r3}"
    cases = {
        "global_v8_b32": (
            f"ld.global.v8.b32 {vector}, [a];\nst.global.v8.b32 [b], {vector};",
            True,
        ),
        "global_v4_b64": (
            (
                "ld.global.v4.b64 {q0, q1, q2, q3}, [a];\n"
                "st.global.v4.b64 [b], {q0, q1, q2, q3};"
            ),
            True,
        ),
        "global_v8_to_shared": (
            (
                f"ld.global.v8.b32 {vector}, [a];\n"
                f"st.shared.v4.b32 [s], {half};\n"
                "st.shared.v4.b32 [s+16], {r4, r5, r6, r7};\n"
                "bar.sync 0;\n"
                f"ld.shared.v4.b32 {half}, [s];\n"
                "ld.shared.v4.b32 {r4, r5, r6, r7}, [s+16];\n"
                f"st.global.v8.b32 [b], {vector};"
            ),
            True,
        ),
        "shared_v4_b32": (
            f"ld.shared.v4.b32 {half}, [s];\nst.global.v4.b32 [b], {half};",
            True,
        ),
        "shared_v8_b32": (
            f"ld.shared.v8.b32 {vector}, [s];\nst.global.v8.b32 [b], {vector};",
            False,
        ),
        "store_shared_v8_b32": (
            f"ld.global.v8.b32 {vector}, [a];\nst.shared.v8.b32 [s], {vector};",
            False,
        ),
        "cp_async_16": (
            (
                "cp.async.cg.shared.global [s], [a], 16;\n"
                "cp.async.commit_group;\ncp.async.wait_group 0;\n"
                f"ld.shared.v4.b32 {half}, [s];\nst.global.v4.b32 [b], {half};"
            ),
            True,
        ),
        "cp_async_32": (
            (
                "cp.async.cg.shared.global [s], [a], 32;\n"
                "cp.async.commit_group;\ncp.async.wait_group 0;\n"
                f"ld.shared.v4.b32 {half}, [s];\nst.global.v4.b32 [b], {half};"
            ),
            False,
        ),
    }
    report = {
        "scope": "Compile/disassembly only; no device execution or performance claim",
        "ptxas": command([str(CUDA / "bin/ptxas"), "--version"]),
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))
        },
        "tool_sha256": {
            tool: digest(CUDA / "bin" / tool) for tool in ["ptxas", "cuobjdump"]
        },
        "retained_source_sha256": {
            name: digest(ROOT / name)
            for name in [
                "experiments/frontier/mlp_geglu.py",
                "experiments/native/kernels/candidates.py",
                "results/frontier/mlp-geglu-unpacked.json",
            ]
        },
        "cases": {},
        "retained_candidates": [],
        "sources": [
            "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async",
        ],
    }
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        for name, (body, expected) in cases.items():
            source = DIRECTORY / f"{name}.ptx"
            binary = DIRECTORY / f"{name}.cubin"
            source.write_text(ptx(body))
            build = command(
                [
                    str(CUDA / "bin/ptxas"),
                    "-arch=sm_120",
                    "-v",
                    str(source),
                    "-o",
                    str(binary),
                ]
            )
            row = {
                "expected_accepted": expected,
                "build": build,
                "ptx_sha256": digest(source),
            }
            assert (build["returncode"] == 0) == expected, row
            if expected:
                disassembly = command(
                    [str(CUDA / "bin/cuobjdump"), "-sass", str(binary)]
                )
                assert disassembly["returncode"] == 0, disassembly
                sass = DIRECTORY / f"{name}.sass"
                sass.write_text(disassembly["stdout"])
                row.update(
                    cubin_sha256=digest(binary),
                    sass_sha256=digest(sass),
                    instruction_counts=instructions(disassembly["stdout"]),
                )
            report["cases"][name] = row

        # Locate existing cache entries by their actual specialized IR, not a
        # guessed cache key. Keep every matching compiler/libdevice provenance.
        for metadata in Path.home().joinpath(".triton/cache").glob("*/_project.json"):
            entry = json.loads(metadata.read_text())
            if not (
                entry.get("shared") == 24576
                and entry.get("num_warps") == 4
                and entry.get("num_stages") == 3
                and not entry.get("enable_fp_fusion")
                and not entry.get("launch_pdl")
            ):
                continue
            base = metadata.parent
            ir = (base / "_project.ttir").read_text()
            assembly = (base / "_project.ptx").read_text()
            if not (
                "mlp_geglu.py" in assembly
                and "%wn = arith.divsi" in ir
                and "tensor<32x64xbf16>" in ir
                and "tensor<64x64xbf16>" in ir
                and 'symbol = "__nv_erff"' in ir
            ):
                continue
            binary = base / "_project.cubin"
            disassembly = command([str(CUDA / "bin/cuobjdump"), "-sass", str(binary)])
            resources = command(
                [str(CUDA / "bin/cuobjdump"), "-res-usage", str(binary)]
            )
            assert disassembly["returncode"] == resources["returncode"] == 0
            out = DIRECTORY / f"retained-{base.name}.sass"
            out.write_text(disassembly["stdout"])
            mainloop = assembly.split("$L__BB0_1:", 1)[1].split("bra \t$L__BB0_1;", 1)[
                0
            ]
            report["retained_candidates"].append(
                {
                    "cache_directory": str(base),
                    "specialization": [32, 64, 64, 4, 3, 2, False],
                    "sha256": {
                        suffix: digest(base / f"_project.{suffix}")
                        for suffix in ["ptx", "cubin", "ttir", "ttgir", "json"]
                    },
                    "sass_sha256": digest(out),
                    "ptx_cp_async_16_static_count": len(
                        re.findall(
                            r"cp.async.cg.shared.global.*?, (?:16|0x10),", assembly
                        )
                    ),
                    "ptx_mainloop_counts": {
                        "cp_async_16": mainloop.count("cp.async.cg.shared.global"),
                        "ldmatrix_x4": mainloop.count(
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
                        ),
                        "mma_m16n8k16": mainloop.count("mma.sync.aligned.m16n8k16"),
                    },
                    "instruction_counts": instructions(disassembly["stdout"]),
                    "resource_usage": resources["stdout"],
                }
            )
        assert report["retained_candidates"]
    report["decision"] = (
        "Native 256-bit global instructions exist. Current GEMM uses direct-to-shared "
        "128-bit asynchronous copies. No 256-bit shared vector or 32-byte cp.async "
        "exists in this ISA. A register-load replacement adds shared-store work and "
        "changes completion scheduling; static width alone does not justify a new GEMM."
    )
    report["limitations"] = [
        "Static instruction counts do not establish an instruction-issue bottleneck.",
        "The probes are compiled, never launched; no bank benchmark or output-parity claim.",
        "Retained cache entries are matched by specialized IR and metadata, not by a new model run.",
        "The prior native_lossless and ws_lossless controls already use 16-byte cp.async copies.",
    ]
    destination = ROOT / "results/frontier/wide-load-audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination)
    print(
        json.dumps(
            {
                k: v.get("instruction_counts", v["build"]["stderr"])
                for k, v in report["cases"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
