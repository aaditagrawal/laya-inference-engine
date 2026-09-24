"""Build separately so illegal CUTLASS tile layouts remain recorded failures."""

import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-irregular-geglu"
SOURCE = ROOT / "experiments/frontier/irregular_geglu.cu"
CUTLASS = ROOT / ".research/frontier-torch-cutlass"
CUDA = Path("/usr/local/cuda-13.1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(config):
    return "-".join(str(x) for x in config)


def configurations():
    # BM, BN, WM, WN, stages, direct accumulator epilogue.
    default = [(32, 64, 16, 32, 3, 0)]
    default += [(32, n, 16, n // 2, 3, 0) for n in (80, 96, 112, 160, 192)]
    default += [(32, 80, 32, 16, 3, 0)]
    direct = [(32, 64, 16, 32, 3, 1)]
    direct += [
        (32, n, 16, n if n in (80, 112) else n // 2, 3, 1)
        for n in (80, 96, 112, 160, 192)
    ]
    direct += [(64, n, 32, n // 2, 3, 1) for n in (96, 160, 192)]
    return default + direct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-two", action="store_true")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest_path = DIRECTORY / "build.json"
    report = json.loads(manifest_path.read_text()) if args.stage_two else {"rows": []}
    report.update(
        {
            "source_sha256": digest(SOURCE),
            "builder_sha256": digest(Path(__file__)),
            "cutlass_revision": subprocess.check_output(
                ["git", "-C", str(CUTLASS), "rev-parse", "HEAD"], text=True
            ).strip(),
        }
    )
    configs = configurations()
    if args.stage_two:
        configs = [(*c[:4], 2, c[5]) for c in configs if c[5]]
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        for config in configs:
            name = key(config)
            binary = DIRECTORY / (name + ".so")
            command = [
                str(CUDA / "bin/nvcc"),
                "-O3",
                "-std=c++20",
                "-arch=sm_120",
                "--compiler-options",
                "-fPIC",
                "--ptxas-options=-v",
                "-lineinfo",
                "--expt-relaxed-constexpr",
                "-I" + str(CUTLASS / "include"),
            ]
            command += [
                f"-DIRREGULAR_{k}={v}"
                for k, v in zip(("BM", "BN", "WM", "WN", "STAGES", "DIRECT"), config)
            ]
            command += ["-shared", str(SOURCE), "-o", str(binary)]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            log = DIRECTORY / (name + ".log")
            log.write_text(result.stdout + result.stderr)
            row = {
                "key": name,
                "config": config,
                "command": command,
                "returncode": result.returncode,
                "log_sha256": digest(log),
            }
            if result.returncode:
                row["errors"] = [
                    line
                    for line in result.stderr.splitlines()
                    if "error:" in line or "static assertion" in line
                ]
            else:
                sass = DIRECTORY / (name + ".sass")
                sass.write_text(
                    subprocess.check_output(
                        [str(CUDA / "bin/cuobjdump"), "-sass", str(binary)], text=True
                    )
                )
                row.update(library_sha256=digest(binary), sass_sha256=digest(sass))
            report["rows"].append(row)
            manifest_path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "command"}), flush=True
            )
        dependencies = subprocess.check_output(
            [
                str(CUDA / "bin/nvcc"),
                "-std=c++20",
                "-arch=sm_120",
                "--expt-relaxed-constexpr",
                "-I" + str(CUTLASS / "include"),
                "-M",
                str(SOURCE),
            ],
            text=True,
        )
    (DIRECTORY / "dependencies.txt").write_text(dependencies)
    report["cutlass_headers_sha256"] = {
        str(p.relative_to(CUTLASS)): digest(p)
        for p in sorted((CUTLASS / "include").rglob("*"))
        if p.is_file() and str(p) in dependencies.replace("\\ ", " ")
    }
    assert report["cutlass_headers_sha256"]
    assert report["source_sha256"] == digest(SOURCE)
    assert report["builder_sha256"] == digest(Path(__file__))
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
