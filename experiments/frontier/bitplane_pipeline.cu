// Exact BF16 bitplanes in PTX MMA lane order. Original experiment, MIT.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ uint32_t transpose32(uint32_t word,int lane){
  #pragma unroll
  for(int shift=0;shift<5;shift++){
    int s=1<<shift;
    uint32_t mask=0xffffffffu/((1u<<s)+1u),high=~mask;
    uint32_t peer=__shfl_xor_sync(0xffffffffu,word,s);
    word=(lane&s)==0 ? (word&mask)|((peer&mask)<<s) : ((peer&high)>>s)|(word&high);
  }
  return word;
}
__device__ __forceinline__ uint32_t decode_bits(uint32_t word){
  uint32_t z=word>>8;
  int exponent=119+((int)(z>>1)^-(int)(z&1));
  return (word&127)|(((word>>7)&1)<<15)|((uint32_t)exponent<<7);
}
template<int GROUPS>
__device__ __forceinline__ uint32_t fragment_value(const uint32_t* w,int group,int lane){
  uint32_t value=lane<16 ? w[lane*GROUPS+(group^lane)] : 0;
  return decode_bits(transpose32(value,lane));
}
template<int GROUPS>
__device__ __forceinline__ uint32_t fragment_pair(const uint32_t* w,int group,int lane){
  int plane=lane&15, g=group+(lane>>4);
  uint32_t value=transpose32(w[plane*GROUPS+(g^plane)],lane);
  return decode_bits(value&65535u)|(decode_bits(value>>16)<<16);
}
__device__ __forceinline__ void mma(float* d,const uint32_t* a,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void copy4(void* dst,const void* src){
  uint32_t d=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d),"l"(src));
}
__device__ __forceinline__ void copy16(void* dst,const void* src){
  uint32_t d=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d),"l"(src));
}
template<int WM,int WN,int NT>
__device__ __forceinline__ void load_stage(const __nv_bfloat16* x,const uint32_t* w,
    __nv_bfloat16* a,uint32_t* b,int K,int step){
  constexpr int BM=WM*16,BN=WN*NT*8,GROUPS=BN*2,THREADS=WM*WN*32;
  for(int i=threadIdx.x;i<BM*8;i+=THREADS){
    int row=i/8,col=(i%8)*8;
    copy16(a+row*64+(col^((row&7)*8)),x+(blockIdx.x*BM+row)*K+step*64+col);
  }
  const uint32_t* src=w+(blockIdx.y*(K/64)+step)*16*GROUPS;
  for(int i=threadIdx.x;i<16*GROUPS;i+=THREADS){
    int plane=i/GROUPS,group=i%GROUPS;
    copy4(b+plane*GROUPS+(group^plane),src+i);
  }
  asm volatile("cp.async.commit_group;");
}
template<int WM,int WN,int NT,int STAGES,bool PAIRED>
__global__ void kernel(const __nv_bfloat16* __restrict__ x,const uint32_t* __restrict__ w,
    __nv_bfloat16* __restrict__ y,int N,int K){
  constexpr int BM=WM*16,BN=WN*NT*8,GROUPS=BN*2;
  __shared__ __align__(16) __nv_bfloat16 a[STAGES][BM*64];
  __shared__ __align__(16) uint32_t b[STAGES][16*GROUPS];
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int mr=(warp/WN)*16,nr=(warp%WN)*NT;
  int group=lane/4,thread=lane%4;
  float accum[NT][4]={};
  #pragma unroll
  for(int i=0;i<STAGES;i++)load_stage<WM,WN,NT>(x,w,a[i],b[i],K,i);
  for(int step=0;step<K/64;step++){
    int slot=step%STAGES;
    int pending=min(STAGES-1,K/64-step-1);
    if(pending==0)asm volatile("cp.async.wait_group 0;");
    else if(pending==1)asm volatile("cp.async.wait_group 1;");
    else asm volatile("cp.async.wait_group 2;");
    __syncthreads();
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t af[4];int k=kt*16+thread*2;
      af[0]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group)*64+(k^(((mr+group)&7)*8)));
      af[1]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group+8)*64+(k^(((mr+group+8)&7)*8)));
      af[2]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group)*64+((k+8)^(((mr+group)&7)*8)));
      af[3]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group+8)*64+((k+8)^(((mr+group+8)&7)*8)));
      #pragma unroll
      for(int j=0;j<NT;j++){
        int g=((nr+j)*4+kt)*4;
        if constexpr(PAIRED){
          uint32_t b0=fragment_pair<GROUPS>(b[slot],g,lane);
          uint32_t b1=fragment_pair<GROUPS>(b[slot],g+2,lane);
          mma(accum[j],af,b0,b1);
        }else{
          uint32_t v0=fragment_value<GROUPS>(b[slot],g,lane);
          uint32_t v1=fragment_value<GROUPS>(b[slot],g+1,lane);
          uint32_t v2=fragment_value<GROUPS>(b[slot],g+2,lane);
          uint32_t v3=fragment_value<GROUPS>(b[slot],g+3,lane);
          mma(accum[j],af,v0|(v1<<16),v2|(v3<<16));
        }
      }
    }
    __syncthreads();
    if(step+STAGES<K/64)load_stage<WM,WN,NT>(x,w,a[slot],b[slot],K,step+STAGES);
  }
  #pragma unroll
  for(int j=0;j<NT;j++){
    int col=blockIdx.y*BN+(nr+j)*8+thread*2;
    int offset=(blockIdx.x*BM+mr+group)*N+col;
    *reinterpret_cast<__nv_bfloat162*>(y+offset)=__floats2bfloat162_rn(accum[j][0],accum[j][1]);
    *reinterpret_cast<__nv_bfloat162*>(y+offset+8*N)=__floats2bfloat162_rn(accum[j][2],accum[j][3]);
  }
}
__global__ void unpack_kernel(const uint32_t* w,uint16_t* y,int N,int K,int BN){
  int lane=threadIdx.x,groups=BN*2;
  int group=blockIdx.z,nb=group/16,kt=(group/4)%4,value=group%4;
  const uint32_t* ptr=w+(blockIdx.x*(K/64)+blockIdx.y)*16*groups;
  uint32_t word=lane<16?ptr[lane*groups+group]:0;
  uint32_t bits=decode_bits(transpose32(word,lane));
  int n=blockIdx.x*BN+nb*8+lane/4;
  int k=blockIdx.y*64+kt*16+(lane%4)*2+value%2+(value/2)*8;
  y[n*K+k]=(uint16_t)bits;
}
extern "C" int bitplane_unpack(const void* w,void* y,int N,int K,int BN,void* stream){
  unpack_kernel<<<dim3(N/BN,K/64,BN*2),32,0,(cudaStream_t)stream>>>((const uint32_t*)w,(uint16_t*)y,N,K,BN);
  return cudaGetLastError();
}
template<int STAGES,bool PAIRED>
void dispatch(const void* x,const void* w,void* y,int N,int K,int tile,cudaStream_t stream){
  #define RUN(WM,WN,NT) kernel<WM,WN,NT,STAGES,PAIRED><<<dim3(64/(WM*16),N/(WN*NT*8)),WM*WN*32,0,stream>>>((const __nv_bfloat16*)x,(const uint32_t*)w,(__nv_bfloat16*)y,N,K)
  switch(tile){case 0:RUN(2,2,2);break;case 1:RUN(4,1,4);break;case 2:RUN(2,2,4);break;case 3:RUN(4,1,2);break;}
  #undef RUN
}
extern "C" int bitplane_gemm(const void* x,const void* w,void* y,int N,int K,int tile,int stages,int paired,void* stream){
  if(paired){
    if(stages==2)dispatch<2,true>(x,w,y,N,K,tile,(cudaStream_t)stream);
    else dispatch<3,true>(x,w,y,N,K,tile,(cudaStream_t)stream);
  }else{
    if(stages==2)dispatch<2,false>(x,w,y,N,K,tile,(cudaStream_t)stream);
    else dispatch<3,false>(x,w,y,N,K,tile,(cudaStream_t)stream);
  }
  return cudaGetLastError();
}
