// Specializations of installed PyTorch BSD-3-Clause attention templates.
// See pytorch-LICENSE.txt and pinned CUTLASS LICENSE.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

template<class Kernel, bool KeepBias>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
special_attention(typename Kernel::Params p) {
  if constexpr (!KeepBias) p.attn_bias_ptr = nullptr;
  p.seqstart_q_ptr = p.seqstart_k_ptr = p.seqlen_k_ptr = nullptr;
  p.logsumexp_ptr = nullptr;
  p.output_accum_ptr = nullptr;
  p.scale = .125f;
  p.head_dim = p.head_dim_value = 64;
  p.num_queries = p.num_keys = p.num_keys_absolute = 64;
  p.num_heads = 16;
  p.num_batches = 1;
  p.q_heads_per_kv = 1;
  p.custom_mask_type = 0;
  p.window_size = 0;
  p.causal_diagonal_offset = 0;
  p.use_dropout = false;
  p.dropout_prob = 0.f;
  p.o_strideM = 16 * 64;
  if (p.advance_to_block()) Kernel::attention_kernel(p);
}

template<int Q, bool Bias, bool Special, bool KeepBias>
at::Tensor run(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::optional<at::Tensor>& bias) {
  using Kernel = PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t,
      cutlass::arch::Sm80, true, Q, 64, 64, false, Bias>;
  typename Kernel::Params p;
  auto out = at::empty({1, 64, 16, 64}, q.options());
  p.query_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(q.data_ptr());
  p.key_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(k.data_ptr());
  p.value_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(v.data_ptr());
  p.output_ptr = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
  p.scale = .125f;
  p.head_dim = p.head_dim_value = 64;
  p.num_queries = p.num_keys = p.num_keys_absolute = 64;
  p.num_heads = 16;
  p.num_batches = 1;
  p.q_strideB = q.stride(0); p.q_strideH = q.stride(1); p.q_strideM = q.stride(2);
  p.k_strideB = k.stride(0); p.k_strideH = k.stride(1); p.k_strideM = k.stride(2);
  p.v_strideB = v.stride(0); p.v_strideH = v.stride(1); p.v_strideM = v.stride(2);
  p.o_strideM = 16 * 64;
  if constexpr (KeepBias) {
    const auto& b = *bias;
    p.attn_bias_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(b.data_ptr());
    p.bias_strideB = b.size(0) == 1 ? 0 : b.stride(0);
    p.bias_strideH = b.size(1) == 1 ? 0 : b.stride(1);
    p.bias_strideM = b.size(2) == 1 ? 0 : b.stride(2);
  }
  auto kernel = Special ? special_attention<Kernel, KeepBias>
    : PyTorchMemEffAttention::attention_kernel_batched_impl<Kernel>;
  constexpr auto shared = sizeof(typename Kernel::SharedStorage);
  if constexpr (shared > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), shared, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.transpose(1, 2);
}

at::Tensor attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     int64_t query_tile, bool bias_support, bool special, const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 4 &&
    q.size(0) == 1 && q.size(1) == 16 && q.size(2) == 64 && q.size(3) == 64 && q.stride(3) == 1,
    "Expected BF16 CUDA (1,16,64,64) input");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() &&
    k.device() == q.device() && v.device() == q.device() &&
    k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
    k.stride(3) == 1 && v.stride(3) == 1, "Invalid K/V");
  if (bias.has_value()) {
    TORCH_CHECK(bias_support && bias->device() == q.device() && bias->scalar_type() == q.scalar_type() &&
      bias->dim() == 4 && bias->size(3) == 64 && bias->stride(3) == 1 && bias->size(0) == 1 &&
      (bias->size(1) == 1 || bias->size(1) == 16) &&
      (bias->size(2) == 1 || bias->size(2) == 64), "Invalid attention bias");
  }
  c10::cuda::CUDAGuard guard(q.device());
  if (bias.has_value()) {
    if (query_tile == 32 && special) return run<32,true,true,true>(q,k,v,bias);
    if (query_tile == 64 && special) return run<64,true,true,true>(q,k,v,bias);
    if (query_tile == 32 && !special) return run<32,true,false,true>(q,k,v,bias);
    if (query_tile == 64 && !special) return run<64,true,false,true>(q,k,v,bias);
    TORCH_CHECK(false, "Query tile must be 32 or 64");
  }
#define DISPATCH(Q, B, S) if (query_tile == Q && bias_support == B && special == S) return run<Q,B,S,false>(q,k,v,bias);
  DISPATCH(32,true,false) DISPATCH(64,true,false)
  DISPATCH(32,false,false) DISPATCH(64,false,false)
  DISPATCH(32,true,true) DISPATCH(64,true,true)
  DISPATCH(32,false,true) DISPATCH(64,false,true)
#undef DISPATCH
  TORCH_CHECK(false, "Query tile must be 32 or 64");
}
TORCH_LIBRARY(laya_fast_attention_special, m) {
  m.def("forward(Tensor q, Tensor k, Tensor v, int query_tile, bool bias_support, bool special, Tensor? bias=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(laya_fast_attention_special, CUDA, m) { m.impl("forward", &attention); }
