// Warp-specialized producer decode/copy and consumer BF16 MMA. MIT.
// The fixed 13-bit checkpoint format reuses native_lossless.py's exact encoding.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ uint32_t shared_address(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void init(uint64_t* p, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(shared_address(p)), "r"(count) : "memory");
}
__device__ __forceinline__ void arrive(uint64_t* p) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(shared_address(p)) : "memory");
}
__device__ __forceinline__ void wait(uint64_t* p, int phase) {
  asm volatile("{ .reg .pred done; again: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, 0x989680; @!done bra again; }"
    :: "r"(shared_address(p)), "r"(phase) : "memory");
}
__device__ __forceinline__ void copy16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared_address(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ uint32_t expand(uint32_t sm, uint32_t exp) {
  uint32_t e0=exp&31, e1=(exp>>5)&31;
  uint32_t a=(sm&127)|((sm&128)<<8)|((e0 ? e0+102 : 0)<<7);
  uint32_t b=((sm>>8)&127)|(sm&32768)|((e1 ? e1+102 : 0)<<7);
  return a|(b<<16);
}
template<bool GLOBAL>
__device__ __forceinline__ uint32_t read(const uint32_t* p) {if constexpr(GLOBAL)return __ldg(p);else return *p;}
template<bool GLOBAL=true>
__device__ __forceinline__ void decode(const uint32_t* p, int lane, uint32_t& b0, uint32_t& b1) {
  uint32_t sm=read<GLOBAL>(p+lane);
  int word=lane*20/32, shift=lane*20%32;
  uint32_t low=read<GLOBAL>(p+32+word), high=word<19 ? read<GLOBAL>(p+33+word) : 0;
  uint32_t exponent=(low>>shift)|(shift ? high<<(32-shift) : 0);
  b0=expand(sm,exponent); b1=expand(sm>>16,exponent>>10);
}
__device__ __forceinline__ void mma(float* d,const uint32_t* a,uint32_t b0,uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}

template<int WM,int WN,int NT,int PRODUCERS,int STAGES,int PACKED>
__global__ __launch_bounds__((PRODUCERS+WM*WN)*32)
void ws_kernel(const __nv_bfloat16* __restrict__ x,const uint32_t* __restrict__ w,
               __nv_bfloat16* __restrict__ y,int N,int K) {
  constexpr int BM=WM*16,BN=WN*NT*8,PT=PRODUCERS*32,CT=WM*WN*32;
  constexpr int FRAGMENTS=BN/8*4;
  __shared__ __align__(16) __nv_bfloat16 a[STAGES][BM*64];
  __shared__ __align__(16) uint32_t b[STAGES][FRAGMENTS*64];
  // Only producer warps access the encoded scratch, each owning disjoint
  // fragments. It does not need a separate copy for every consumer stage.
  __shared__ __align__(16) uint32_t encoded[PACKED==2 ? FRAGMENTS*52 : 1];
  __shared__ __align__(8) uint64_t full[STAGES],empty[STAGES];
  if(threadIdx.x<STAGES) {
    init(full+threadIdx.x,PT);
    init(empty+threadIdx.x,CT);
  }
  __syncthreads();
  const int tid=threadIdx.x,lane=tid%32;
  if(tid<PT) {
    // Producers can run STAGES tiles ahead, independently of consumer MMA.
    for(int step=0;step<K/64;step++) {
      int slot=step%STAGES;
      if(step>=STAGES)wait(empty+slot,(step/STAGES-1)&1);
      for(int i=tid;i<BM*8;i+=PT) {
        int row=i/8,col=i%8*8;
        copy16(a[slot]+row*64+(col^((row&7)*8)),x+(blockIdx.x*BM+row)*K+step*64+col);
      }
      if constexpr(PACKED) {
        if constexpr(PACKED==2) {
          for(int fragment=tid/32;fragment<FRAGMENTS;fragment+=PRODUCERS) {
            int nb=fragment/4,kt=fragment%4;
            const uint32_t* src=w+((blockIdx.y*(BN/8)+nb)*(K/16)+step*4+kt)*52;
            if(lane<13)copy16(encoded+fragment*52+lane*4,src+lane*4);
          }
          asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
          __syncwarp();
        }
        for(int fragment=tid/32;fragment<FRAGMENTS;fragment+=PRODUCERS) {
          int nb=fragment/4,kt=fragment%4;
          const uint32_t* src=w+((blockIdx.y*(BN/8)+nb)*(K/16)+step*4+kt)*52;
          uint32_t b0,b1;
          if constexpr(PACKED==2)decode<false>(encoded+fragment*52,lane,b0,b1);
          else decode(src,lane,b0,b1);
          b[slot][fragment*64+lane]=b0;
          b[slot][fragment*64+32+lane]=b1;
        }
        if constexpr(PACKED==2)__syncwarp();
      } else {
        for(int i=tid;i<FRAGMENTS*16;i+=PT) {
          int fragment=i/16,sub=i%16;
          int nb=fragment/4,kt=fragment%4;
          const uint32_t* src=w+((blockIdx.y*(BN/8)+nb)*(K/16)+step*4+kt)*64+sub*4;
          copy16(b[slot]+fragment*64+sub*4,src);
        }
      }
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
      // Every producing thread publishes its own writes; the final arrival
      // releases the tile for all consumers, with no full-CTA loop barrier.
      arrive(full+slot);
    }
  } else {
    int warp=(tid-PT)/32,mr=(warp/WN)*16,nr=(warp%WN)*NT;
    int group=lane/4,thread=lane%4;
    float accum[NT][4]={};
    for(int step=0;step<K/64;step++) {
      int slot=step%STAGES;
      wait(full+slot,(step/STAGES)&1);
      #pragma unroll
      for(int kt=0;kt<4;kt++) {
        uint32_t af[4];int k=kt*16+thread*2;
        af[0]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group)*64+(k^(((mr+group)&7)*8)));
        af[1]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group+8)*64+(k^(((mr+group+8)&7)*8)));
        af[2]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group)*64+((k+8)^(((mr+group)&7)*8)));
        af[3]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group+8)*64+((k+8)^(((mr+group+8)&7)*8)));
        #pragma unroll
        for(int j=0;j<NT;j++) {
          const uint32_t* ptr=b[slot]+((nr+j)*4+kt)*64;
          mma(accum[j],af,ptr[lane],ptr[32+lane]);
        }
      }
      arrive(empty+slot);
    }
    #pragma unroll
    for(int j=0;j<NT;j++) {
      int col=blockIdx.y*BN+(nr+j)*8+thread*2;
      int base=(blockIdx.x*BM+mr+group)*N+col;
      *reinterpret_cast<__nv_bfloat162*>(y+base)=__floats2bfloat162_rn(accum[j][0],accum[j][1]);
      *reinterpret_cast<__nv_bfloat162*>(y+base+8*N)=__floats2bfloat162_rn(accum[j][2],accum[j][3]);
    }
  }
}

template<bool PACKED>
__global__ void decode_kernel(const uint32_t* w,uint16_t* y,int K) {
  int nb=blockIdx.x,kb=blockIdx.y,lane=threadIdx.x;
  const uint32_t* p=w+(nb*(K/16)+kb)*(PACKED?52:64);
  uint32_t b0,b1;
  if constexpr(PACKED)decode(p,lane,b0,b1);
  else {b0=p[lane];b1=p[32+lane];}
  int n=nb*8+lane/4,k=kb*16+lane%4*2;
  *reinterpret_cast<uint32_t*>(y+n*K+k)=b0;
  *reinterpret_cast<uint32_t*>(y+n*K+k+8)=b1;
}
extern "C" int ws_unpack(const void* w,void* y,int N,int K,int packed,void* stream) {
  if(packed)decode_kernel<true><<<dim3(N/8,K/16),32,0,(cudaStream_t)stream>>>((const uint32_t*)w,(uint16_t*)y,K);
  else decode_kernel<false><<<dim3(N/8,K/16),32,0,(cudaStream_t)stream>>>((const uint32_t*)w,(uint16_t*)y,K);
  return cudaGetLastError();
}
template<int PRODUCERS,int STAGES,int PACKED>
void dispatch(const void* x,const void* w,void* y,int N,int K,int tile,cudaStream_t stream) {
  #define RUN(WM,WN,NT) ws_kernel<WM,WN,NT,PRODUCERS,STAGES,PACKED><<<dim3(64/(WM*16),N/(WN*NT*8)),(PRODUCERS+WM*WN)*32,0,stream>>>((const __nv_bfloat16*)x,(const uint32_t*)w,(__nv_bfloat16*)y,N,K)
  switch(tile) {case 0:RUN(2,2,2);break;case 1:RUN(4,1,4);break;case 2:RUN(2,2,4);break;}
  #undef RUN
}
extern "C" int ws_gemm(const void* x,const void* w,void* y,int N,int K,int tile,int producers,int stages,int packed,void* stream) {
  #define DISPATCH(P) if(producers==1) {if(stages==2)dispatch<1,2,P>(x,w,y,N,K,tile,(cudaStream_t)stream);else dispatch<1,3,P>(x,w,y,N,K,tile,(cudaStream_t)stream);} else {if(stages==2)dispatch<2,2,P>(x,w,y,N,K,tile,(cudaStream_t)stream);else dispatch<2,3,P>(x,w,y,N,K,tile,(cudaStream_t)stream);}
  if(packed==2){DISPATCH(2);}else if(packed){DISPATCH(1);}else{DISPATCH(0);}
  #undef DISPATCH
  return cudaGetLastError();
}
