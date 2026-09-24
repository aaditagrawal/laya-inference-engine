"""Build and inspect thread-block cluster support without claiming multicast."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-cluster-gemm"
SOURCE = ROOT / "experiments/frontier/cluster_gemm_capability.cu"
INSTRUCTION = ROOT / "experiments/frontier/cluster_gemm_multicast_ptx.cu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    nvcc = "/usr/local/cuda-13.1/bin/nvcc"
    report_path = DIRECTORY / "build.json"
    if args.build_only:
        outputs = [
            (
                "capability",
                [
                    nvcc,
                    "-O3",
                    "-std=c++20",
                    "-arch=sm_120",
                    str(SOURCE),
                    "-o",
                    str(DIRECTORY / "capability"),
                ],
            ),
            (
                "multicast_compile",
                [
                    nvcc,
                    "-O3",
                    "-std=c++20",
                    "-arch=sm_120",
                    "-cubin",
                    str(INSTRUCTION),
                    "-o",
                    str(DIRECTORY / "multicast.cubin"),
                ],
            ),
            (
                "multicast_ptx",
                [
                    nvcc,
                    "-O3",
                    "-std=c++20",
                    "-arch=compute_120",
                    "-ptx",
                    str(INSTRUCTION),
                    "-o",
                    str(DIRECTORY / "multicast.ptx"),
                ],
            ),
        ]
        report = {
            "source_sha256": {
                s.name: hashlib.sha256(s.read_bytes()).hexdigest()
                for s in [SOURCE, INSTRUCTION]
            },
            "builds": {},
        }
        for name, command in outputs:
            result = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            report["builds"][name] = {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            print(name, result.returncode, result.stderr, flush=True)
        if (DIRECTORY / "capability").exists():
            report["executable_sha256"] = hashlib.sha256(
                (DIRECTORY / "capability").read_bytes()
            ).hexdigest()
        if (DIRECTORY / "multicast.cubin").exists():
            report["cubin_sha256"] = hashlib.sha256(
                (DIRECTORY / "multicast.cubin").read_bytes()
            ).hexdigest()
        if report["builds"]["multicast_compile"]["returncode"] == 0:
            dumper = "/usr/local/cuda-13.1/bin/cuobjdump"
            cubin = str(DIRECTORY / "multicast.cubin")
            sass = subprocess.run(
                [dumper, "--dump-sass", cubin],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            elf = subprocess.run(
                [dumper, "--dump-elf", cubin],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            (DIRECTORY / "instructions.sass").write_text(sass)
            (DIRECTORY / "instructions.elf.txt").write_text(elf)
            report["code_generation"] = {
                "multicast_syscall_relocation_present": "__cuda_syscall_cp_async_bulk_tensor_2d_tile_multicast"
                in elf,
                "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(),
                "functions": {
                    name: {
                        "native_tma_instruction_count": body.count("UTMALDG"),
                        "indirect_call_present": "CALL.ABS.NOINC" in body,
                    }
                    for name, body in re.findall(
                        r"Function : (\w+)(.*?)(?=Function : |\Z)", sass, re.DOTALL
                    )
                },
            }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if report["builds"]["capability"]["returncode"]:
            raise RuntimeError("Native capability probe failed to compile")
        return
    report = json.loads(report_path.read_text())
    for source in [SOURCE, INSTRUCTION]:
        if (
            report["source_sha256"][source.name]
            != hashlib.sha256(source.read_bytes()).hexdigest()
        ):
            raise RuntimeError("Rebuild the changed capability source")
    if (
        report["executable_sha256"]
        != hashlib.sha256((DIRECTORY / "capability").read_bytes()).hexdigest()
    ):
        raise RuntimeError("Native capability executable hash differs")
    result = subprocess.run(
        [str(DIRECTORY / "capability")], capture_output=True, text=True, check=True
    )
    report["device_probe"] = json.loads(result.stdout)
    report["sources"] = [
        "https://docs.nvidia.com/cutlass/4.5.1/media/docs/cpp/blackwell_functionality.html#cluster-size",
        "https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu",
        "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor",
    ]
    report["interpretation"] = (
        "The runtime probe establishes generic thread-block cluster support and distributed shared-memory reads only. Compilation acceptance does not establish hardware TMA multicast acceleration. NVIDIA CUTLASS documentation states GeForce SM120 has no TMA multicast feature and uses cluster shape 1x1x1 for GEMM. No multicast GEMM was implemented or benchmarked."
    )
    output = ROOT / "results/frontier/cluster-gemm-capability.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["device_probe"]), flush=True)


if __name__ == "__main__":
    main()
