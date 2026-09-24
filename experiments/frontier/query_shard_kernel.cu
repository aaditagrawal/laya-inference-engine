// Query-row sharding of installed PyTorch BSD-3-Clause attention templates.
// See experiments/native/kernels/pytorch-LICENSE and the pinned CUTLASS LICENSE.
// MMA, softmax, key reduction, and epilogue templates remain 32Q x 64K x 64D.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

using Kernel = PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t,
    cutlass::arch::Sm80, true, 32, 64, 64, false, true>;

template<int Rows, bool KeepBias>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
sharded_attention(Kernel::Params p) {
  const int head = blockIdx.y;
  const int query = blockIdx.x * Rows;
  // Do not call advance_to_block: it would add blockIdx.x * 32 a second time.
  p.query_ptr = warp_uniform(p.query_ptr + query * p.q_strideM + head * p.q_strideH);
  p.key_ptr = warp_uniform(p.key_ptr + head * p.k_strideH);
  p.value_ptr = warp_uniform(p.value_ptr + head * p.v_strideH);
  p.output_ptr = warp_uniform(p.output_ptr + query * (16 * 64) + head * 64);
  if constexpr (KeepBias) {
    p.attn_bias_ptr = warp_uniform(p.attn_bias_ptr +
        int64_t(head) * p.bias_strideH + int64_t(query) * p.bias_strideM);
  } else p.attn_bias_ptr = nullptr;
  p.output_accum_ptr = reinterpret_cast<float*>(p.output_ptr);
  p.seqstart_q_ptr = p.seqstart_k_ptr = p.seqlen_k_ptr = nullptr;
  p.logsumexp_ptr = nullptr;
  p.scale = .125f;
  p.head_dim = p.head_dim_value = 64;
  p.num_queries = Rows;
  p.num_keys = p.num_keys_absolute = 64;
  p.num_heads = 16;
  p.num_batches = 0;
  p.q_heads_per_kv = 1;
  p.custom_mask_type = 0;
  p.window_size = 0;
  p.causal_diagonal_offset = 0;
  p.use_dropout = false;
  p.dropout_prob = 0.f;
  p.o_strideM = 16 * 64;
  // attention_kernel reads blockIdx.x only inside disabled causal/window/dropout
  // paths. Its Q load and output iterator extents use p.num_queries == Rows.
  Kernel::attention_kernel(p);
}

template<int Rows, bool KeepBias>
at::Tensor run(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
               const std::optional<at::Tensor>& bias) {
  Kernel::Params p;
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
    p.bias_strideB = 0;
    p.bias_strideH = b.size(1) == 1 ? 0 : b.stride(1);
    p.bias_strideM = b.size(2) == 1 ? 0 : b.stride(2);
  }
  Kernel::check_supported(p);
  auto kernel = sharded_attention<Rows, KeepBias>;
  constexpr auto shared = sizeof(Kernel::SharedStorage);
  if constexpr (shared > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  kernel<<<dim3(64 / Rows, 16, 1), p.getThreadsGrid(), shared,
             at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.transpose(1, 2);
}

at::Tensor forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                   int64_t rows, const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 4 &&
    q.size(0) == 1 && q.size(1) == 16 && q.size(2) == 64 && q.size(3) == 64 && q.stride(3) == 1,
    "Expected BF16 CUDA (1,16,64,64) input");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() &&
    k.device() == q.device() && v.device() == q.device() &&
    k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
    k.stride(3) == 1 && v.stride(3) == 1, "Invalid K/V");
  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == q.device() && bias->scalar_type() == q.scalar_type() &&
      bias->dim() == 4 && bias->size(3) == 64 && bias->stride(3) == 1 && bias->size(0) == 1 &&
      (bias->size(1) == 1 || bias->size(1) == 16) &&
      (bias->size(2) == 1 || bias->size(2) == 64), "Invalid attention bias");
  }
  c10::cuda::CUDAGuard guard(q.device());
#define DISPATCH(Rows) if (rows == Rows) { \
  if (bias.has_value()) return run<Rows,true>(q,k,v,bias); \
  return run<Rows,false>(q,k,v,bias); }
  DISPATCH(32) DISPATCH(16) DISPATCH(8)
#undef DISPATCH
  TORCH_CHECK(false, "Query shard must contain 8, 16, or 32 rows");
}

TORCH_LIBRARY(laya_frontier_query_shard, m) {
  m.def("forward(Tensor q, Tensor k, Tensor v, int rows, Tensor? bias=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(laya_frontier_query_shard, CUDA, m) { m.impl("forward", &forward); }
