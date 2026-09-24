// Instantiate the exact FlashAttention submodule pinned by installed PyTorch.
// FlashAttention source is BSD-3-Clause, copyright Tri Dao. See pinned LICENSE.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "flash_fwd_launch_template.h"

using Params = FLASH_NAMESPACE::Flash_fwd_params;

template<class Traits, bool Fixed, bool Even>
__global__ void global_attention_kernel(const __grid_constant__ Params supplied) {
  Params p = supplied;
  if constexpr (Fixed) {
    p.b = 1; p.h = p.h_k = 16; p.h_h_k_ratio = 1;
    p.seqlen_q = p.seqlen_k = 64; p.d = p.d_rounded = 64;
    p.seqlen_q_rounded = p.seqlen_k_rounded = 128;
    p.scale_softmax = .125f;
    p.scale_softmax_log2 = float(.125 * M_LOG2E);
    p.rp_dropout = p.p_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.scale_softmax_rp_dropout = .125f;
    p.cu_seqlens_q = p.cu_seqlens_k = p.leftpad_k = p.seqused_k = nullptr;
    p.knew_ptr = p.vnew_ptr = nullptr;
    p.p_ptr = nullptr;
    p.seqlen_knew = 0;
    p.is_seqlens_k_cumulative = true;
    p.unpadded_lse = false;
    p.seqlenq_ngroups_swapped = false;
    p.is_bf16 = true; p.is_causal = false;
    p.window_size_left = p.window_size_right = -1;
    p.alibi_slopes_ptr = nullptr;
    p.softcap = 0.f;
    p.o_row_stride = 1024; p.o_head_stride = 64; p.o_batch_stride = 65536;
  }
  FLASH_NAMESPACE::compute_attn<Traits, false, false, false, false, Even, true, false, false>(p);
}

template<int M, int N, int Warps, bool Fixed>
at::Tensor run(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  using Traits = Flash_fwd_kernel_traits<64, M, N, Warps, false, false, cutlass::bfloat16_t>;
  auto result = at::empty({1,64,16,64}, q.options());
  auto lse = at::empty({1,16,64}, q.options().dtype(at::kFloat));
  Params p{};
  p.q_ptr=q.data_ptr(); p.k_ptr=k.data_ptr(); p.v_ptr=v.data_ptr();
  p.o_ptr=result.data_ptr(); p.softmax_lse_ptr=lse.data_ptr();
  p.q_batch_stride=q.stride(0); p.q_head_stride=q.stride(1); p.q_row_stride=q.stride(2);
  p.k_batch_stride=k.stride(0); p.k_head_stride=k.stride(1); p.k_row_stride=k.stride(2);
  p.v_batch_stride=v.stride(0); p.v_head_stride=v.stride(1); p.v_row_stride=v.stride(2);
  p.o_row_stride=1024; p.o_head_stride=64; p.o_batch_stride=65536;
  p.b=1; p.h=p.h_k=16; p.h_h_k_ratio=1;
  p.seqlen_q=p.seqlen_k=64; p.d=p.d_rounded=64;
  p.seqlen_q_rounded=p.seqlen_k_rounded=128;
  p.scale_softmax=.125f; p.scale_softmax_log2=float(.125*M_LOG2E);
  p.rp_dropout=p.p_dropout=1.f; p.p_dropout_in_uint8_t=255;
  p.scale_softmax_rp_dropout=.125f;
  p.is_seqlens_k_cumulative=true; p.is_bf16=true;
  p.window_size_left=p.window_size_right=-1;
  constexpr bool Even=(64 % M == 0 && 64 % N == 0);
  auto kernel = global_attention_kernel<Traits, Fixed, Even>;
  constexpr size_t shared=Traits::kSmemSize;
  if constexpr (shared>=48*1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  kernel<<<dim3((64+M-1)/M,1,16),Traits::kNThreads,shared,at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result.transpose(1,2);
}

at::Tensor attention(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,int64_t config) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==4 &&
    q.size(0)==1 && q.size(1)==16 && q.size(2)==64 && q.size(3)==64 && q.stride(3)==1,
    "Expected BF16 CUDA (1,16,64,64)");
  TORCH_CHECK(k.sizes()==q.sizes() && v.sizes()==q.sizes() && k.device()==q.device() &&
    v.device()==q.device() && k.scalar_type()==q.scalar_type() && v.scalar_type()==q.scalar_type() &&
    k.stride(3)==1 && v.stride(3)==1,"Invalid K/V");
  c10::cuda::CUDAGuard guard(q.device());
  switch(config) {
    case 0:return run<128,128,4,false>(q,k,v);
    case 1:return run<128,128,4,true>(q,k,v);
    case 2:return run<64,128,4,false>(q,k,v);
    case 3:return run<64,128,4,true>(q,k,v);
    case 4:return run<32,128,2,false>(q,k,v);
    case 5:return run<32,128,2,true>(q,k,v);
    case 6:return run<16,128,1,false>(q,k,v);
    case 7:return run<16,128,1,true>(q,k,v);
    case 8:return run<64,64,4,true>(q,k,v);
    case 9:return run<32,64,2,true>(q,k,v);
    default:TORCH_CHECK(false,"Unknown attention configuration");
  }
}
TORCH_LIBRARY(laya_global_attention,m) {
  m.def("forward(Tensor q, Tensor k, Tensor v, int config) -> Tensor");
}
TORCH_LIBRARY_IMPL(laya_global_attention,CUDA,m) { m.impl("forward",&attention); }
