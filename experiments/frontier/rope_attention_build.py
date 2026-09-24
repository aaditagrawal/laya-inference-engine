"""Generate private staging-fusion headers and compile pinned attention arithmetic."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.cpp_extension import include_paths

TORCH_REVISION = "08187d9e0fba026dc8217405802ab5381dc88d90"
CUTLASS_REVISION = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
FLASH_REVISION = "14c377950125c70b7a9dabf9c561fca53715ac7d"


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Pinned header replacement changed: {old[:80]}")
    return source.replace(old, new)


def generate(directory, torch_root, flash, direct=False):
    original = (
        torch_root
        / "include/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h"
    )
    source = original.read_text().replace(
        "PyTorchMemEffAttention", "RopeMemEffAttention"
    )
    source = once(
        source,
        "  struct Params {",
        "  struct Params {\n    const float* rope_cos = nullptr;\n    const float* rope_sin = nullptr;",
    )
    old = "      // Construct thread-scoped matrix multiply\n"
    new = """      // Q/K are loaded once into the existing complete-matrix shared storage.
      static_assert(MM0::Mma::kSmemContainsEntireMat);
      MM0::Mma::prologue(shared_storage.mm0, iterator_A, iterator_B,
                        thread_id(), problem_size_0_k);
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();
      rope_attention_detail::rotate_cutlass<kQueriesPerBlock>(
          shared_storage.mm0.operand_A.ref(), shared_storage.mm0.operand_B.ref(),
          p.rope_cos, p.rope_sin, query_start, iter_key_start,
          thread_id(), kNumThreads);
      __syncthreads();
      // Construct thread-scoped matrix multiply
"""
    if direct:
        new = """      static_assert(MM0::Mma::kSmemContainsEntireMat);
      rope_attention_detail::load_cutlass<kQueriesPerBlock>(
          shared_storage.mm0.operand_A.ref(), shared_storage.mm0.operand_B.ref(),
          p.query_ptr, p.key_ptr, p.rope_cos, p.rope_sin, query_start,
          thread_id(), kNumThreads);
      __syncthreads();
      // Construct thread-scoped matrix multiply
"""
    source = once(source, old, new)
    source = once(
        source,
        "      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);",
        "      mma.set_prologue_done(true);\n      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);",
    )
    local = directory / "rope_attention_cutlass_kernel.h"
    local.write_text(source)

    flash_original = flash / "csrc/flash_attn/src/flash_fwd_kernel.h"
    source = flash_original.read_text()
    old = """        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            FLASH_NAMESPACE::copy"""
    new = """        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        static_assert(kBlockN == 64 && !Kernel_traits::Is_Q_in_regs);
        // These specializations have exactly one key tile. Rotate each pair
        // once in its existing shared staging, with no global scratch copy.
        rope_attention_detail::rotate_flash<kBlockM>(
            sQ, sK, params.rope_cos, params.rope_sin,
            m_block * kBlockM, n_block * kBlockN, tidx, Kernel_traits::kNThreads);
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            FLASH_NAMESPACE::copy"""
    if direct:
        old = """    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);"""
        source = once(
            source,
            old,
            """    static_assert(kBlockN == 64 && !Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem);
    rope_attention_detail::load_flash<kBlockM>(
        sQ, sK,
        reinterpret_cast<const cutlass::bfloat16_t*>(params.q_ptr) + bidh * 64,
        reinterpret_cast<const cutlass::bfloat16_t*>(params.k_ptr) + bidh * 64,
        params.rope_cos, params.rope_sin, m_block * kBlockM, tidx, Kernel_traits::kNThreads);""",
        )
        old = """    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);"""
        source = once(
            source, old, "    // Q/K already rotated into their shared staging buffers."
        )
    else:
        source = once(source, old, new)
    global_header = directory / "rope_attention_flash_kernel.h"
    global_header.write_text(source)
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [original, flash_original, local, global_header]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    if torch.version.git_version != TORCH_REVISION:
        raise RuntimeError("Requires pinned Torch")
    torch_root = Path(torch.__file__).parent
    cutlass = Path(".research/frontier-torch-cutlass").resolve()
    flash = Path(".research/frontier-torch-flash").resolve()
    for checkout, expected in [(cutlass, CUTLASS_REVISION), (flash, FLASH_REVISION)]:
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise RuntimeError(f"Pinned dependency changed: {checkout}")
    directory = Path(".research/frontier-rope-attention").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    headers = generate(directory, torch_root, flash, args.direct)
    source = Path(__file__).with_name("rope_attention_kernel.cu").resolve()
    helper = source.with_name("rope_attention_helpers.h")
    includes = [
        "-I" + str(directory),
        "-I" + str(source.parent),
        "-I" + str(cutlass / "include"),
        "-I" + str(flash / "csrc/flash_attn/src"),
    ]
    includes += ["-I" + p for p in include_paths(device_type="cuda")]
    cuda = Path("/usr/local/cuda-13.1")
    obj = directory / "rope_attention.o"
    library = directory / "laya_rope_attention.so"
    start = perf_counter()
    subprocess.run(
        [
            str(cuda / "bin/nvcc"),
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_120,code=sm_120",
            "--compiler-options",
            "-fPIC",
            "-DFLASH_NAMESPACE=laya_rope_flash",
            "-DFLASHATTENTION_DISABLE_DROPOUT",
            "-DUNFUSE_FMA",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            *includes,
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
    )
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            str(obj),
            "-shared",
            "-L" + str(torch_root / "lib"),
            "-lc10",
            "-lc10_cuda",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-ltorch",
            "-ltorch_python",
            "-L" + str(cuda / "lib64"),
            "-lcudart",
            "-o",
            str(library),
        ],
        check=True,
    )
    torch.ops.load_library(str(library))
    report = {
        "direct_staging": args.direct,
        "seconds": perf_counter() - start,
        "torch_revision": TORCH_REVISION,
        "cutlass_revision": CUTLASS_REVISION,
        "flash_revision": FLASH_REVISION,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "headers_sha256": headers,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "architecture": "sm_120",
        "fast_math": False,
        "rope_math": "Separate FP32 multiply and add rounding, then BF16 RN",
    }
    (directory / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
