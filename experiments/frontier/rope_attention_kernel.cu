// Private staging fusion of pinned PyTorch CUTLASS and FlashAttention kernels.
// See experiments/native/kernels/pytorch-LICENSE and global_attention_FLASH_LICENSE.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "rope_attention_helpers.h"
#include "rope_attention_cutlass_kernel.h"
#include "flash.h"
#include "rope_attention_flash_kernel.h"

template<class Kernel, bool KeepBias>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
rope_cutlass_kernel(typename Kernel::Params p) {
  if constexpr (!KeepBias) p.attn_bias_ptr = nullptr;
  p.seqstart_q_ptr = p.seqstart_k_ptr = p.seqlen_k_ptr = nullptr;
  p.logsumexp_ptr = nullptr; p.output_accum_ptr = nullptr;
  p.scale = .125f; p.head_dim = p.head_dim_value = 64;
  p.num_queries = p.num_keys = p.num_keys_absolute = 64;
  p.num_heads = 16; p.num_batches = 1; p.q_heads_per_kv = 1;
  p.custom_mask_type = 0; p.window_size = 0; p.causal_diagonal_offset = 0;
  p.use_dropout = false; p.dropout_prob = 0.f; p.o_strideM = 1024;
  if (p.advance_to_block()) Kernel::attention_kernel(p);
}

template<int Q, bool KeepBias>
at::Tensor run_cutlass(const at::Tensor& qkv, const at::Tensor& cos,
    const at::Tensor& sin, const std::optional<at::Tensor>& bias) {
  using Kernel = RopeMemEffAttention::AttentionKernel<cutlass::bfloat16_t,
      cutlass::arch::Sm80, true, Q, 64, 64, false, true>;
  typename Kernel::Params p;
  auto out = at::empty({1,64,16,64},qkv.options());
  const auto* base=reinterpret_cast<const cutlass::bfloat16_t*>(qkv.data_ptr());
  p.query_ptr=base; p.key_ptr=base+1024; p.value_ptr=base+2048;
  p.output_ptr=reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
  p.rope_cos=cos.data_ptr<float>(); p.rope_sin=sin.data_ptr<float>();
  p.scale=.125f; p.head_dim=p.head_dim_value=64;
  p.num_queries=p.num_keys=p.num_keys_absolute=64;
  p.num_heads=16; p.num_batches=1;
  p.q_strideB=p.k_strideB=p.v_strideB=196608;
  p.q_strideH=p.k_strideH=p.v_strideH=64;
  p.q_strideM=p.k_strideM=p.v_strideM=3072;
  p.o_strideM=1024;
  if constexpr (KeepBias) {
    const auto& b=*bias;
    p.attn_bias_ptr=reinterpret_cast<const cutlass::bfloat16_t*>(b.data_ptr());
    p.bias_strideB=0;
    p.bias_strideH=b.size(1)==1 ? 0 : b.stride(1);
    p.bias_strideM=b.size(2)==1 ? 0 : b.stride(2);
  }
  auto kernel=rope_cutlass_kernel<Kernel,KeepBias>;
  constexpr auto shared=sizeof(typename Kernel::SharedStorage);
  if constexpr (shared>48*1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  kernel<<<p.getBlocksGrid(),p.getThreadsGrid(),shared,at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.transpose(1,2);
}

struct RopeFlashParams : FLASH_NAMESPACE::Flash_fwd_params {
  const float* rope_cos;
  const float* rope_sin;
};

template<class Traits>
__global__ void rope_flash_kernel(const __grid_constant__ RopeFlashParams supplied) {
  auto p=supplied;
  p.b=1; p.h=p.h_k=16; p.h_h_k_ratio=1;
  p.seqlen_q=p.seqlen_k=64; p.d=p.d_rounded=64;
  p.seqlen_q_rounded=p.seqlen_k_rounded=128;
  p.scale_softmax=.125f; p.scale_softmax_log2=float(.125*M_LOG2E);
  p.rp_dropout=p.p_dropout=1.f; p.p_dropout_in_uint8_t=255;
  p.scale_softmax_rp_dropout=.125f;
  p.cu_seqlens_q=p.cu_seqlens_k=p.leftpad_k=p.seqused_k=nullptr;
  p.knew_ptr=p.vnew_ptr=nullptr; p.p_ptr=nullptr; p.seqlen_knew=0;
  p.is_seqlens_k_cumulative=true; p.unpadded_lse=false;
  p.seqlenq_ngroups_swapped=false; p.is_bf16=true; p.is_causal=false;
  p.window_size_left=p.window_size_right=-1; p.alibi_slopes_ptr=nullptr;
  p.softcap=0.f;
  p.q_batch_stride=p.k_batch_stride=p.v_batch_stride=196608;
  p.q_head_stride=p.k_head_stride=p.v_head_stride=64;
  p.q_row_stride=p.k_row_stride=p.v_row_stride=3072;
  p.o_row_stride=1024; p.o_head_stride=64; p.o_batch_stride=65536;
  FLASH_NAMESPACE::compute_attn<Traits,false,false,false,false,true,true,false,false>(p);
}

template<int Q,int Warps>
at::Tensor run_flash(const at::Tensor& qkv,const at::Tensor& cos,const at::Tensor& sin) {
  using Traits=Flash_fwd_kernel_traits<64,Q,64,Warps,false,false,cutlass::bfloat16_t>;
  auto out=at::empty({1,64,16,64},qkv.options());
  auto lse=at::empty({1,16,64},qkv.options().dtype(at::kFloat));
  auto* base=reinterpret_cast<cutlass::bfloat16_t*>(qkv.data_ptr());
  RopeFlashParams p{};
  p.q_ptr=base; p.k_ptr=base+1024; p.v_ptr=base+2048;
  p.o_ptr=out.data_ptr(); p.softmax_lse_ptr=lse.data_ptr();
  p.rope_cos=cos.data_ptr<float>(); p.rope_sin=sin.data_ptr<float>();
  auto kernel=rope_flash_kernel<Traits>;
  constexpr auto shared=Traits::kSmemSize;
  if constexpr (shared>=48*1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  kernel<<<dim3(64/Q,1,16),Traits::kNThreads,shared,at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.transpose(1,2);
}

at::Tensor attention(const at::Tensor& qkv,const at::Tensor& cos,const at::Tensor& sin,
    int64_t config,const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type()==at::kBFloat16 && qkv.is_contiguous() &&
    qkv.sizes()==at::IntArrayRef({1,64,3,16,64}),"Expected contiguous BF16 CUDA QKV (1,64,3,16,64)");
  for (const auto& t : {cos,sin})
    TORCH_CHECK(t.device()==qkv.device() && t.scalar_type()==at::kFloat && t.is_contiguous() &&
      t.numel()>=64*64,"Expected contiguous FP32 rotary tables");
  if (bias.has_value())
    TORCH_CHECK(config<2 && bias->device()==qkv.device() && bias->scalar_type()==at::kBFloat16 &&
      bias->dim()==4 && bias->size(0)==1 && (bias->size(1)==1 || bias->size(1)==16) &&
      (bias->size(2)==1 || bias->size(2)==64) && bias->size(3)==64 && bias->stride(3)==1,"Invalid bias");
  c10::cuda::CUDAGuard guard(qkv.device());
  switch(config) {
    case 0: return bias.has_value() ? run_cutlass<32,true>(qkv,cos,sin,bias) : run_cutlass<32,false>(qkv,cos,sin,bias);
    case 1: return bias.has_value() ? run_cutlass<64,true>(qkv,cos,sin,bias) : run_cutlass<64,false>(qkv,cos,sin,bias);
    case 2: return run_flash<64,4>(qkv,cos,sin);
    case 3: return run_flash<32,2>(qkv,cos,sin);
    case 4: return run_flash<16,1>(qkv,cos,sin);
    default: TORCH_CHECK(false,"Unknown configuration");
  }
}
TORCH_LIBRARY(laya_rope_attention,m) {
  m.def("forward(Tensor qkv, Tensor cos, Tensor sin, int config, Tensor? bias=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(laya_rope_attention,CUDA,m) { m.impl("forward",&attention); }
