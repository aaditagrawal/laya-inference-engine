// Exact Welford prologue-only feasibility control for repeated MLP N tiles.
// Welford order follows native_vector.cu / PyTorch, BSD-3-Clause.
// See norm_geglu_LICENSE and norm_geglu_NOTICE.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

struct W {float mean,var,count;};
struct alignas(8) BF4 {__nv_bfloat16 v[4];};
__device__ __forceinline__ W online(float x,W a) {
  float d=x-a.mean,n=a.count+1.f;
  float m=a.mean+d*(1.f/n);
  return {m,a.var+d*(x-m),n};
}
__device__ __forceinline__ W combine(W b,W a) {
  float d=b.mean-a.mean,n=a.count+b.count;
  float c=1.f/n,na=a.count*c,nb=b.count*c;
  return {na*a.mean+nb*b.mean,a.var+b.var+d*d*a.count*nb,n};
}

template<int BM,bool Validate>
__global__ __launch_bounds__(128) void norm_prologue(
  float const* x,__nv_bfloat16 const* r,float const* weight,float const* bias,
  float* stats,float* residual,__nv_bfloat16* normalized,float eps) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  for(int offset=0;offset<BM;offset+=4) {
    int row=blockIdx.y*BM+warp+offset;
    W states[4];
    float values[4][8];
    #pragma unroll
    for(int original_warp=0;original_warp<4;original_warp++) {
      int t=lane+original_warp*32;
      W state={0.f,0.f,0.f};
      #pragma unroll
      for(int j=0;j<2;j++) {
        float4 a=reinterpret_cast<float4 const*>(x+row*1024)[t+j*128];
        BF4 b=reinterpret_cast<BF4 const*>(r+row*1024)[t+j*128];
        #pragma unroll
        for(int k=0;k<4;k++) {
          float value=__fadd_rn(reinterpret_cast<float*>(&a)[k],__bfloat162float(b.v[k]));
          if constexpr(Validate)values[original_warp][j*4+k]=value;
          state=online(value,state);
        }
      }
      #pragma unroll
      for(int delta=16;delta>0;delta>>=1) {
        W b={__shfl_down_sync(0xffffffff,state.mean,delta),
             __shfl_down_sync(0xffffffff,state.var,delta),
             __shfl_down_sync(0xffffffff,state.count,delta)};
        state=combine(state,b);
      }
      states[original_warp]=state;
    }
    float mean=0.f,rstd=0.f;
    if(lane==0) {
      W first=combine({states[0].mean,states[0].var,256.f},{states[2].mean,states[2].var,256.f});
      W second=combine({states[1].mean,states[1].var,256.f},{states[3].mean,states[3].var,256.f});
      W whole=combine(first,second);
      mean=whole.mean;rstd=rsqrtf(whole.var*(1.f/1024.f)+eps);
      stats[(blockIdx.x*64+row)*2]=mean;
      stats[(blockIdx.x*64+row)*2+1]=rstd;
    }
    if constexpr(Validate) {
      mean=__shfl_sync(0xffffffff,mean,0);
      rstd=__shfl_sync(0xffffffff,rstd,0);
      if(blockIdx.x==0) {
        #pragma unroll
        for(int original_warp=0;original_warp<4;original_warp++) {
          int t=lane+original_warp*32;
          #pragma unroll
          for(int j=0;j<2;j++) {
            float4 y;
            BF4 z;
            float4 w=reinterpret_cast<float4 const*>(weight)[t+j*128];
            float4 b;
            if(bias)b=reinterpret_cast<float4 const*>(bias)[t+j*128];
            #pragma unroll
            for(int k=0;k<4;k++) {
              float value=values[original_warp][j*4+k];
              reinterpret_cast<float*>(&y)[k]=value;
              float result=reinterpret_cast<float*>(&w)[k]*(rstd*(value-mean));
              if(bias)result+=reinterpret_cast<float*>(&b)[k];
              z.v[k]=__float2bfloat16_rn(result);
            }
            reinterpret_cast<float4*>(residual+row*1024)[t+j*128]=y;
            reinterpret_cast<BF4*>(normalized+row*1024)[t+j*128]=z;
          }
        }
      }
    }
  }
}

template<int BM,bool Validate>
int launch(void const* x,void const* r,void const* w,void const* bias,void* stats,void* residual,
           void* normalized,int tiles,float eps,cudaStream_t stream,int* info) {
  auto kernel=norm_prologue<BM,Validate>;
  if(info) {
    cudaFuncAttributes a;
    auto error=cudaFuncGetAttributes(&a,kernel);
    if(error)return error;
    info[0]=a.numRegs;info[1]=a.sharedSizeBytes;info[2]=a.localSizeBytes;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(info+3,kernel,128,0);
  }
  kernel<<<dim3(tiles,64/BM),128,0,stream>>>(static_cast<float const*>(x),static_cast<__nv_bfloat16 const*>(r),
    static_cast<float const*>(w),static_cast<float const*>(bias),static_cast<float*>(stats),
    static_cast<float*>(residual),static_cast<__nv_bfloat16*>(normalized),eps);
  return cudaGetLastError();
}
extern "C" int norm_geglu_prologue(void const* x,void const* r,void const* w,void const* b,void* stats,
                                   void* residual,void* normalized,int bm,int tiles,int validate,float eps,
                                   cudaStream_t stream,int* info) {
  if(bm==32)return validate ? launch<32,true>(x,r,w,b,stats,residual,normalized,tiles,eps,stream,info)
                            : launch<32,false>(x,r,w,b,stats,residual,normalized,tiles,eps,stream,info);
  if(bm==16)return validate ? launch<16,true>(x,r,w,b,stats,residual,normalized,tiles,eps,stream,info)
                            : launch<16,false>(x,r,w,b,stats,residual,normalized,tiles,eps,stream,info);
  return cudaErrorInvalidValue;
}
