"""Build a narrow BF16 x INT8 Marlin subset from a pinned vLLM checkout.

The Apache-2.0 upstream sources remain under .research with their notices.
Run with the shared build lock; no changes to the inference environment.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import torch

REVISION = "00b7847c8036b667742b4efb21aab1de51fd4721"


def main():
    source = Path(".research/frontier-vllm").resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError("Unexpected vLLM revision")
    destination = Path(".research/frontier-marlin").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    marlin = destination / "marlin"
    shutil.copytree(
        source / "csrc/libtorch_stable/quantization/marlin", marlin, dirs_exist_ok=True
    )
    for filename in ("marlin.cu", "gptq_marlin_repack.cu"):
        path = marlin / filename
        path.write_text(
            path.read_text().replace(
                "STABLE_TORCH_LIBRARY_IMPL(_C,",
                "STABLE_TORCH_LIBRARY_IMPL(laya_frontier_marlin,",
            )
        )
    sys.argv = ["generate_kernels.py", "12.0"]
    spec = importlib.util.spec_from_file_location(
        "frontier_marlin_generator", marlin / "generate_kernels.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.QUANT_CONFIGS = [
        {
            "a_type": ["kBFloat16"],
            "c_type": ["kBFloat16"],
            "b_type": "kU8B128",
            "thread_configs": generator.THREAD_CONFIGS,
            "thread_m_blocks": [1, 2, 3, 4],
            "group_blocks": [2, 4],
        }
    ]
    generator.remove_old_kernels()
    generator.generate_new_kernels()
    bindings = destination / "bindings.cpp"
    bindings.write_text("""#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(laya_frontier_marlin, m) {
 m.def("marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, Tensor? b_bias_or_none, Tensor b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor workspace, int b_type_id, SymInt size_m, SymInt size_n, SymInt size_k, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
 m.def("gptq_marlin_repack(Tensor b_q_weight, SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
}
""")
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13.1"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    os.environ["MAX_JOBS"] = "2"
    sources = [
        marlin / "marlin.cu",
        marlin / "gptq_marlin_repack.cu",
        *marlin.glob("sm80_kernel*.cu"),
        bindings,
    ]
    start = perf_counter()
    # Explicit argument arrays also support workspace paths containing spaces.
    torch_root = Path(torch.__file__).parent
    includes = [
        "-I" + str(p)
        for p in [
            source / "csrc",
            marlin,
            torch_root / "include",
            torch_root / "include/torch/csrc/api/include",
            Path(sysconfig.get_paths()["include"]),
            Path(os.environ["CUDA_HOME"]) / "include",
        ]
    ]

    def compile_source(path):
        obj = destination / (path.stem + ".o")
        common = [
            "-O3",
            "-std=c++20",
            "-DUSE_CUDA",
            *includes,
            "-c",
            str(path),
            "-o",
            str(obj),
        ]
        if path.suffix == ".cu":
            command = [
                os.environ["CUDA_HOME"] + "/bin/nvcc",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-static-global-template-stub=false",
                "-gencode=arch=compute_120,code=sm_120",
                "--compiler-options",
                "-fPIC",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
                *common,
            ]
        else:
            command = ["c++", "-fPIC", *common]
        subprocess.run(command, check=True)
        return obj

    with ThreadPoolExecutor(max_workers=2) as pool:
        objects = list(pool.map(compile_source, sources))
    library = destination / "laya_frontier_marlin.so"
    subprocess.run(
        [
            "c++",
            *map(str, objects),
            "-shared",
            "-L" + str(torch_root / "lib"),
            "-lc10",
            "-lc10_cuda",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-ltorch",
            "-ltorch_python",
            "-L" + os.environ["CUDA_HOME"] + "/lib64",
            "-lcudart",
            "-o",
            str(library),
        ],
        check=True,
    )
    torch.ops.load_library(str(library))
    report = {
        "vllm_revision": revision,
        "seconds": perf_counter() - start,
        "torch": torch.__version__,
        "cuda_toolkit": "13.1",
        "architecture": "sm_120",
        "subset": "BF16 activation, UINT8 biased by 128, groups 32 and 64",
    }
    (destination / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
