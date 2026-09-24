// Instantiate the installed PyTorch attention kernel with a smaller query tile.
// PyTorch headers: BSD-3-Clause, Meta Platforms and NVIDIA contributors.
// See pytorch-LICENSE.txt and the pinned CUTLASS checkout license.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

template<int Q>
at::Tensor run_attention(const at::Tensor& query, const at::Tensor& key,
                        const at::Tensor& value, const std::optional<at::Tensor>& bias) {
  using Kernel = PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t,
      cutlass::arch::Sm80, true, Q, 64, 64, false, true>;
  typename Kernel::Params p;
  const auto batch = query.size(0), heads = query.size(1), length = query.size(2);
  auto result = at::empty({batch, length, heads, 64}, query.options());
  p.query_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(query.data_ptr());
  p.key_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(key.data_ptr());
  p.value_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(value.data_ptr());
  p.output_ptr = reinterpret_cast<cutlass::bfloat16_t*>(result.data_ptr());
  p.scale = 0.125f;
  p.head_dim = p.head_dim_value = 64;
  p.num_queries = p.num_keys = p.num_keys_absolute = length;
  p.num_batches = batch;
  p.num_heads = heads;
  p.q_strideB = query.stride(0); p.q_strideH = query.stride(1); p.q_strideM = query.stride(2);
  p.k_strideB = key.stride(0); p.k_strideH = key.stride(1); p.k_strideM = key.stride(2);
  p.v_strideB = value.stride(0); p.v_strideH = value.stride(1); p.v_strideM = value.stride(2);
  p.o_strideM = heads * 64;
  if (bias.has_value()) {
    const auto& b = *bias;
    p.attn_bias_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(b.data_ptr());
    p.bias_strideB = b.size(0) == 1 ? 0 : b.stride(0);
    p.bias_strideH = b.size(1) == 1 ? 0 : b.stride(1);
    p.bias_strideM = b.size(2) == 1 ? 0 : b.stride(2);
  }
  auto kernel = PyTorchMemEffAttention::attention_kernel_batched_impl<Kernel>;
  constexpr auto shared = sizeof(typename Kernel::SharedStorage);
  if constexpr (shared > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  }
  kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), shared, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result.transpose(1, 2);
}

at::Tensor attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     const std::optional<at::Tensor>& bias, int64_t query_tile) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 4 &&
              q.size(2) == 64 && q.size(3) == 64 && q.stride(3) == 1,
              "Expected CUDA BF16 Q/K/V with sequence=64 and head_dim=64");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() &&
              k.device() == q.device() && v.device() == q.device() &&
              k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
              k.stride(3) == 1 && v.stride(3) == 1, "Invalid K/V");
  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == q.device() && bias->scalar_type() == q.scalar_type() &&
                bias->dim() == 4 && bias->size(3) == 64 && bias->stride(3) == 1 &&
                (bias->size(0) == 1 || bias->size(0) == q.size(0)) &&
                (bias->size(1) == 1 || bias->size(1) == q.size(1)) &&
                (bias->size(2) == 1 || bias->size(2) == q.size(2)), "Invalid attention bias");
  }
  c10::cuda::CUDAGuard guard(q.device());
  if (query_tile == 32) return run_attention<32>(q, k, v, bias);
  if (query_tile == 64) return run_attention<64>(q, k, v, bias);
  TORCH_CHECK(false, "Query tile must be 32 or 64");
}

TORCH_LIBRARY(laya_fast_attention, m) {
  m.def("forward(Tensor q, Tensor k, Tensor v, Tensor? bias, int query_tile) -> Tensor");
}
TORCH_LIBRARY_IMPL(laya_fast_attention, CUDA, m) {
  m.impl("forward", &attention);
}
