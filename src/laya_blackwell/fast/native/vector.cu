// Welford ordering adapted from PyTorch layer_norm_kernel_vector.cu, BSD-3-Clause.
// PyTorch commit 08187d9e0fba026dc8217405802ab5381dc88d90. See pytorch-LICENSE.txt.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>
struct W {float mean, var, count;};
struct alignas(8) BF4 { __nv_bfloat16 v[4]; };
__device__ __forceinline__ W online(float x, W a) {
  float d=x-a.mean, n=a.count+1.f;
  float m=a.mean+d*(1.f/n);
  return {m,a.var+d*(x-m),n};
}
__device__ __forceinline__ W combine(W b, W a) {
  float d=b.mean-a.mean, n=a.count+b.count;
  float c=1.f/n, na=a.count*c, nb=b.count*c;
  return {na*a.mean+nb*b.mean,a.var+b.var+d*d*a.count*nb,n};
}
__device__ __forceinline__ float down(float x, int delta) {
  float r;
  asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=f"(r) : "f"(x),"r"(delta));
  // The matching reduction uses down, not butterfly. The bfly helper is retained
  // only for an experimental variant below; no arithmetic benefit is claimed.
  return r;
}
template<bool HAS_R, bool RFLOAT, bool HAS_B, bool PTX>
__global__ void norm_kernel_vector(const float* __restrict__ x, const void* __restrict__ residual,
 const float* __restrict__ weight, const float* __restrict__ bias,
 float* __restrict__ y, __nv_bfloat16* __restrict__ z, int d, float eps) {
  __shared__ float sm[8];
  int lane=threadIdx.x, warp=threadIdx.y, t=lane+32*warp;
  int row=blockIdx.x, base=row*d;
  float v[8];
  W w={0.f,0.f,0.f};
  #pragma unroll
  for(int j=0;j<2;j++) {
    float4 a=reinterpret_cast<const float4*>(x+base)[t+j*128];
    float* p=reinterpret_cast<float*>(&a);
    float4 rf;
    BF4 rb;
    if constexpr(HAS_R) {
      if constexpr(RFLOAT) rf=reinterpret_cast<const float4*>(static_cast<const float*>(residual)+base)[t+j*128];
      else rb=reinterpret_cast<const BF4*>(static_cast<const __nv_bfloat16*>(residual)+base)[t+j*128];
    }
    #pragma unroll
    for(int k=0;k<4;k++) {
      int col=4*(t+j*128)+k;
      float value=p[k];
      if constexpr(HAS_R) {
        float r;
        if constexpr(RFLOAT) r=reinterpret_cast<float*>(&rf)[k];
        else r=__bfloat162float(rb.v[k]);
        value=__fadd_rn(value,r);
      }
      v[j*4+k]=value;
      w=online(value,w);
    }
  }
  #pragma unroll
  for(int offset=16;offset>0;offset>>=1) {
    W b;
    if constexpr(PTX) {
      asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=f"(b.mean) : "f"(w.mean),"r"(offset));
      asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=f"(b.var) : "f"(w.var),"r"(offset));
      asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=f"(b.count) : "f"(w.count),"r"(offset));
    } else b={__shfl_down_sync(0xffffffff,w.mean,offset),__shfl_down_sync(0xffffffff,w.var,offset),__shfl_down_sync(0xffffffff,w.count,offset)};
    w=combine(w,b);
  }
  if(lane==0) {sm[2*warp]=w.mean;sm[2*warp+1]=w.var;}
  __syncthreads();
  // Preserve PyTorch's warp0+warp2, warp1+warp3, then pair0+pair1 tree.
  // Width1024 and128threads mean every warp represents exactly256 values.
  W first=combine({sm[0],sm[1],256.f},{sm[4],sm[5],256.f});
  W second=combine({sm[2],sm[3],256.f},{sm[6],sm[7],256.f});
  W whole=combine(first,second);
  float mean=whole.mean, rstd=rsqrtf(whole.var*(1.f/1024.f)+eps);
  #pragma unroll
  for(int j=0;j<2;j++) {
    float4 out;
    float4 w4=reinterpret_cast<const float4*>(weight)[t+j*128];
    float4 b4;
    if constexpr(HAS_B) b4=reinterpret_cast<const float4*>(bias)[t+j*128];
    BF4 zout;
    float* p=reinterpret_cast<float*>(&out);
    #pragma unroll
    for(int k=0;k<4;k++) {
      int col=4*(t+j*128)+k;
      float value=v[j*4+k];p[k]=value;
      float result=reinterpret_cast<float*>(&w4)[k]*(rstd*(value-mean));
      if constexpr(HAS_B) result+=reinterpret_cast<float*>(&b4)[k];
      zout.v[k]=__float2bfloat16_rn(result);
    }
    reinterpret_cast<float4*>(y+base)[t+j*128]=out;
    reinterpret_cast<BF4*>(z+base)[t+j*128]=zout;
  }
}
std::vector<at::Tensor> fused_norm_vector(at::Tensor x, c10::optional<at::Tensor> r, at::Tensor weight,
 c10::optional<at::Tensor> bias, double eps, bool ptx) {
 TORCH_CHECK(x.is_cuda() && x.dim()>0 && x.scalar_type()==at::kFloat && x.is_contiguous() && x.size(-1)==1024,"FP32 contiguous CUDA width 1024 required");
 TORCH_CHECK(weight.device()==x.device() && weight.scalar_type()==at::kFloat && weight.is_contiguous() && weight.numel()==1024,"FP32 contiguous weight on input device required");
 if(r.has_value()) TORCH_CHECK(r->device()==x.device() && r->sizes()==x.sizes() && r->is_contiguous() && (r->scalar_type()==at::kFloat || r->scalar_type()==at::kBFloat16),"Matching contiguous FP32/BF16 residual required");
 if(bias.has_value()) TORCH_CHECK(bias->device()==x.device() && bias->scalar_type()==at::kFloat && bias->is_contiguous() && bias->numel()==1024,"FP32 contiguous bias on input device required");
 c10::cuda::CUDAGuard device_guard(x.device());
 auto y=at::empty_like(x);auto z=at::empty_like(x,x.options().dtype(at::kBFloat16));
 if(x.numel()==0) return {y,z};
 int rows=x.numel()/1024; const void* rp=r.has_value()?r->data_ptr():nullptr;
 const float* bp=bias.has_value()?bias->data_ptr<float>():nullptr;
 auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
 #define LAUNCH(R,RF,B,P) norm_kernel_vector<R,RF,B,P><<<rows,dim3(32,4),0,stream>>>(x.data_ptr<float>(),rp,weight.data_ptr<float>(),bp,y.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(z.data_ptr()),1024,eps)
 #define LAUNCH_P(R,RF,B) if(ptx){LAUNCH(R,RF,B,true);}else{LAUNCH(R,RF,B,false);}
 #define LAUNCH_B(R,RF) if(bias.has_value()){LAUNCH_P(R,RF,true);}else{LAUNCH_P(R,RF,false);}
 if(r.has_value()) {if(r->scalar_type()==at::kFloat){LAUNCH_B(true,true);}else{LAUNCH_B(true,false);}}else{LAUNCH_B(false,false);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,z};
}

TORCH_LIBRARY(laya_fast_vector,m) {
 m.def("norm(Tensor x, Tensor? residual, Tensor weight, Tensor? bias, float eps, bool ptx=False) -> Tensor[]");
 m.impl("norm",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(fused_norm_vector)));
}
