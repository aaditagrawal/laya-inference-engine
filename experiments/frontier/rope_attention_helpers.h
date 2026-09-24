#pragma once
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
namespace rope_attention_detail {
__device__ __forceinline__ cutlass::bfloat16_t rotated(float a, float b, float c, float s) {
  return cutlass::bfloat16_t(__fadd_rn(__fmul_rn(a,c), __fmul_rn(b,s)));
}
template<int Q, class RefA, class RefB>
__device__ __forceinline__ void load_cutlass(RefA a, RefB b,
    const cutlass::bfloat16_t* gq,const cutlass::bfloat16_t* gk,
    const float* c,const float* s,int q0,int tid,int threads) {
#pragma unroll
  for(int i=tid;i<64*32;i+=threads) {
    int row=i/32,d=i%32;
    float cl=c[row*64+d],sl=s[row*64+d];
    float ch=c[row*64+d+32],sh=s[row*64+d+32];
    float kl=float(gk[row*3072+d]),kh=float(gk[row*3072+d+32]);
    b.at({d,row})=rotated(kl,-kh,cl,sl);
    b.at({d+32,row})=rotated(kh,kl,ch,sh);
    if(row>=q0 && row<q0+Q) {
      int qr=row-q0;
      float ql=float(gq[qr*3072+d]),qh=float(gq[qr*3072+d+32]);
      a.at({qr,d})=rotated(ql,-qh,cl,sl);
      a.at({qr,d+32})=rotated(qh,ql,ch,sh);
    }
  }
}
template<int Q,class TensorA,class TensorB>
__device__ __forceinline__ void load_flash(TensorA a,TensorB b,
    const cutlass::bfloat16_t* gq,const cutlass::bfloat16_t* gk,
    const float* c,const float* s,int q0,int tid,int threads) {
#pragma unroll
  for(int i=tid;i<64*32;i+=threads) {
    int row=i/32,d=i%32;
    float cl=c[row*64+d],sl=s[row*64+d];
    float ch=c[row*64+d+32],sh=s[row*64+d+32];
    float kl=float(gk[row*3072+d]),kh=float(gk[row*3072+d+32]);
    b(row,d)=rotated(kl,-kh,cl,sl);
    b(row,d+32)=rotated(kh,kl,ch,sh);
    if(row>=q0 && row<q0+Q) {
      int qr=row-q0;
      float ql=float(gq[row*3072+d]),qh=float(gq[row*3072+d+32]);
      a(qr,d)=rotated(ql,-qh,cl,sl);
      a(qr,d+32)=rotated(qh,ql,ch,sh);
    }
  }
}
template<int Q, class RefA, class RefB>
__device__ __forceinline__ void rotate_cutlass(RefA a, RefB b, const float* c, const float* s,
    int q0,int k0,int tid,int threads) {
#pragma unroll
  for(int i=tid;i<Q*32;i+=threads) {
    int row=i/32, d=i%32, pos=q0+row;
    float lo=float(a.at({row,d})), hi=float(a.at({row,d+32}));
    auto out_lo=rotated(lo,-hi,c[pos*64+d],s[pos*64+d]);
    auto out_hi=rotated(hi,lo,c[pos*64+d+32],s[pos*64+d+32]);
    a.at({row,d})=out_lo; a.at({row,d+32})=out_hi;
  }
#pragma unroll
  for(int i=tid;i<64*32;i+=threads) {
    int row=i/32,d=i%32,pos=k0+row;
    float lo=float(b.at({d,row})),hi=float(b.at({d+32,row}));
    auto out_lo=rotated(lo,-hi,c[pos*64+d],s[pos*64+d]);
    auto out_hi=rotated(hi,lo,c[pos*64+d+32],s[pos*64+d+32]);
    b.at({d,row})=out_lo; b.at({d+32,row})=out_hi;
  }
}
template<int Q, class TensorA, class TensorB>
__device__ __forceinline__ void rotate_flash(TensorA a, TensorB b, const float* c, const float* s,
    int q0,int k0,int tid,int threads) {
#pragma unroll
  for(int i=tid;i<Q*32;i+=threads) {
    int row=i/32,d=i%32,pos=q0+row;
    float lo=float(a(row,d)),hi=float(a(row,d+32));
    auto out_lo=rotated(lo,-hi,c[pos*64+d],s[pos*64+d]);
    auto out_hi=rotated(hi,lo,c[pos*64+d+32],s[pos*64+d+32]);
    a(row,d)=out_lo; a(row,d+32)=out_hi;
  }
#pragma unroll
  for(int i=tid;i<64*32;i+=threads) {
    int row=i/32,d=i%32,pos=k0+row;
    float lo=float(b(row,d)),hi=float(b(row,d+32));
    auto out_lo=rotated(lo,-hi,c[pos*64+d],s[pos*64+d]);
    auto out_hi=rotated(hi,lo,c[pos*64+d+32],s[pos*64+d+32]);
    b(row,d)=out_lo; b(row,d+32)=out_hi;
  }
}
}
