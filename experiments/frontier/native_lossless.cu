// Direct MMA-register decoding of checkpoint-specific lossless BF16 weights.
// Original experiment code, MIT. PTX lane layout follows NVIDIA PTX ISA.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ uint32_t expand(uint32_t sm, uint32_t exp) {
    uint32_t e0=exp&31, e1=(exp>>5)&31;
    uint32_t a=(sm&127) | ((sm&128)<<8) | ((e0 ? e0+102 : 0)<<7);
    uint32_t b=((sm>>8)&127) | ((sm&32768)) | ((e1 ? e1+102 : 0)<<7);
    return a | (b<<16);
}
template<bool PACKED>
__device__ __forceinline__ void weight_fragment(const uint32_t* ptr, int lane, uint32_t &b0, uint32_t &b1) {
    if constexpr(PACKED) {
        uint32_t sm=__ldg(ptr+lane);
        int word=lane*20/32, shift=lane*20%32;
        uint32_t low=__ldg(ptr+32+word), high=word<19 ? __ldg(ptr+33+word) : 0;
        uint32_t exp=(low>>shift) | (shift ? high<<(32-shift) : 0);
        b0=expand(sm,exp);
        b1=expand(sm>>16,exp>>10);
    } else {
        b0=__ldg(ptr+lane); b1=__ldg(ptr+32+lane);
    }
}
__device__ __forceinline__ void mma(float *d, const uint32_t *a, uint32_t b0,uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}

template<int WM, int WN, int NT, bool PACKED, int UNROLL>
__global__ void kernel(const __nv_bfloat16* __restrict__ X, const uint32_t* __restrict__ W,
                       __nv_bfloat16* __restrict__ Y, int N, int K, int split) {
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    int mr=blockIdx.x*(WM*16)+(warp/WN)*16;
    int nr=blockIdx.y*(WN*NT*8)+(warp%WN)*NT*8;
    int group=lane/4, thread=lane%4;
    int steps=((K+63)/64+split-1)/split;
    int first=blockIdx.z*steps*4, last=min(first+steps*4,K/16);
    float accum[NT][4]={};
    constexpr int STRIDE=PACKED ? 52 : 64;
    #pragma unroll UNROLL
    for(int kt=first;kt<last;kt++) {
        uint32_t a[4];
        int k=kt*16+thread*2;
        a[0]=__ldg(reinterpret_cast<const uint32_t*>(X+(mr+group)*K+k));
        a[1]=__ldg(reinterpret_cast<const uint32_t*>(X+(mr+group+8)*K+k));
        a[2]=__ldg(reinterpret_cast<const uint32_t*>(X+(mr+group)*K+k+8));
        a[3]=__ldg(reinterpret_cast<const uint32_t*>(X+(mr+group+8)*K+k+8));
        #pragma unroll
        for(int j=0;j<NT;j++) {
            uint32_t b0,b1;
            const uint32_t *p=W+((nr/8+j)*(K/16)+kt)*STRIDE;
            weight_fragment<PACKED>(p,lane,b0,b1);
            mma(accum[j],a,b0,b1);
        }
    }
    #pragma unroll
    for(int j=0;j<NT;j++) {
        int col=nr+j*8+thread*2;
        int base=blockIdx.z*64*N+(mr+group)*N+col;
        __nv_bfloat162 v0=__floats2bfloat162_rn(accum[j][0],accum[j][1]);
        __nv_bfloat162 v1=__floats2bfloat162_rn(accum[j][2],accum[j][3]);
        *reinterpret_cast<__nv_bfloat162*>(Y+base)=v0;
        *reinterpret_cast<__nv_bfloat162*>(Y+base+8*N)=v1;
    }
}


__device__ __forceinline__ void copy_async(void* dst,const void* src) {
    uint32_t d=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void copy_zero(void* dst) {
    *reinterpret_cast<uint4*>(dst)=make_uint4(0,0,0,0);
}
template<int WM,int WN,int NT,bool PACKED>
__device__ __forceinline__ void stage_copy(const __nv_bfloat16* X,const uint32_t* W,
                                          __nv_bfloat16* a,uint32_t* b,int N,int K,int tileK) {
    constexpr int BM=WM*16, BN=WN*NT*8, STRIDE=PACKED?52:64;
    constexpr int THREADS=WM*WN*32;
    for(int v=threadIdx.x;v<BM*8;v+=THREADS) {
        int row=v/8,col=(v%8)*8;
        if(tileK*64+col<K) copy_async(a+row*64+(col^((row&7)*8)),X+(blockIdx.x*BM+row)*K+tileK*64+col);
        else copy_zero(a+row*64+(col^((row&7)*8)));
    }
    for(int v=threadIdx.x;v<(BN/8)*4*(STRIDE/4);v+=THREADS) {
        int tile=v/(STRIDE/4), sub=v%(STRIDE/4);
        int nb=tile/4,kt=tile%4;
        if(tileK*4+kt<K/16)
            copy_async(b+tile*STRIDE+sub*4,W+((blockIdx.y*(BN/8)+nb)*(K/16)+tileK*4+kt)*STRIDE+sub*4);
        else copy_zero(b+tile*STRIDE+sub*4);
    }
    asm volatile("cp.async.commit_group;");
}
template<bool PACKED>
__device__ __forceinline__ void shared_fragment(const uint32_t* ptr,int lane,uint32_t &b0,uint32_t &b1) {
    if constexpr(PACKED) {
        uint32_t sm=ptr[lane];
        int word=lane*20/32,shift=lane*20%32;
        uint32_t low=ptr[32+word],high=word<19?ptr[33+word]:0;
        uint32_t exp=(low>>shift) | (shift?high<<(32-shift):0);
        b0=expand(sm,exp); b1=expand(sm>>16,exp>>10);
    } else { b0=ptr[lane]; b1=ptr[32+lane]; }
}
template<int WM,int WN,int NT,bool PACKED>
__global__ void pipelined(const __nv_bfloat16* __restrict__ X,const uint32_t* __restrict__ W,
                          __nv_bfloat16* __restrict__ Y,int N,int K,int split) {
    constexpr int BM=WM*16,BN=WN*NT*8,STRIDE=PACKED?52:64;
    __shared__ __align__(16) __nv_bfloat16 a[2][BM*64];
    __shared__ __align__(16) uint32_t b[2][(BN/8)*4*STRIDE];
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int mr=(warp/WN)*16,nr=(warp%WN)*NT;
    int group=lane/4,thread=lane%4;
    int steps=((K+63)/64+split-1)/split;
    int first=blockIdx.z*steps,last=min(first+steps,K/64);
    float accum[NT][4]={};
    stage_copy<WM,WN,NT,PACKED>(X,W,a[0],b[0],N,K,first);
    stage_copy<WM,WN,NT,PACKED>(X,W,a[1],b[1],N,K,first+1);
    for(int iter=first;iter<last;iter++) {
        int slot=(iter-first)%2;
        if(iter+1<last) asm volatile("cp.async.wait_group 1;");
        else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        #pragma unroll
        for(int kt=0;kt<4;kt++) {
            uint32_t af[4];
            int k=kt*16+thread*2;
            af[0]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group)*64+(k^(((mr+group)&7)*8)));
            af[1]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group+8)*64+(k^(((mr+group+8)&7)*8)));
            af[2]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group)*64+((k+8)^(((mr+group)&7)*8)));
            af[3]=*reinterpret_cast<const uint32_t*>(a[slot]+(mr+group+8)*64+((k+8)^(((mr+group+8)&7)*8)));
            #pragma unroll
            for(int j=0;j<NT;j++) {
                uint32_t b0,b1;
                shared_fragment<PACKED>(b[slot]+((nr+j)*4+kt)*STRIDE,lane,b0,b1);
                mma(accum[j],af,b0,b1);
            }
        }
        __syncthreads();
        if(iter+2<last) stage_copy<WM,WN,NT,PACKED>(X,W,a[slot],b[slot],N,K,iter+2);
        // Keep a second committed group, including an empty terminal group,
        // so wait_group 1 always waits for the current tile.
        else asm volatile("cp.async.commit_group;");
    }
    #pragma unroll
    for(int j=0;j<NT;j++) {
        int col=blockIdx.y*BN+(nr+j)*8+thread*2;
        int base=blockIdx.z*64*N+(blockIdx.x*BM+mr+group)*N+col;
        *reinterpret_cast<__nv_bfloat162*>(Y+base)=__floats2bfloat162_rn(accum[j][0],accum[j][1]);
        *reinterpret_cast<__nv_bfloat162*>(Y+base+8*N)=__floats2bfloat162_rn(accum[j][2],accum[j][3]);
    }
}
template<bool PACKED>
__global__ void decode_kernel(const uint32_t* W,uint16_t* Y,int K) {
    int nb=blockIdx.x,kb=blockIdx.y,lane=threadIdx.x;
    uint32_t b0,b1;
    weight_fragment<PACKED>(W+(nb*(K/16)+kb)*(PACKED?52:64),lane,b0,b1);
    int n=nb*8+lane/4,k=kb*16+(lane%4)*2;
    *reinterpret_cast<uint32_t*>(Y+n*K+k)=b0;
    *reinterpret_cast<uint32_t*>(Y+n*K+k+8)=b1;
}
at::Tensor decode(const at::Tensor &w,int64_t n,int64_t k,bool packed) {
    TORCH_CHECK(w.is_cuda()&&w.is_contiguous()&&w.scalar_type()==at::kUInt32);
    TORCH_CHECK(n>0&&k>0&&n%8==0&&k%16==0);
    TORCH_CHECK(w.numel()==n/8*(k/16)*(packed?52:64));
    c10::cuda::CUDAGuard guard(w.device());
    auto y=at::empty({n,k},w.options().dtype(at::kBFloat16));
    auto wp=reinterpret_cast<const uint32_t*>(w.data_ptr());
    auto yp=reinterpret_cast<uint16_t*>(y.data_ptr());
    auto stream=at::cuda::getCurrentCUDAStream();
    if(packed) decode_kernel<true><<<dim3(n/8,k/16),32,0,stream>>>(wp,yp,k);
    else decode_kernel<false><<<dim3(n/8,k/16),32,0,stream>>>(wp,yp,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

template<int WM,int WN,int NT,bool PACKED>
void launch(const at::Tensor &x,const at::Tensor &w,at::Tensor &y,int N,int K,int split,int unroll,cudaStream_t stream) {
    dim3 grid(64/(WM*16),N/(WN*NT*8),split);
    auto xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto wp=reinterpret_cast<const uint32_t*>(w.data_ptr());
    auto yp=reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
    if(unroll==0) pipelined<WM,WN,NT,PACKED><<<grid,32*WM*WN,0,stream>>>(xp,wp,yp,N,K,split);
    else if(unroll==4) kernel<WM,WN,NT,PACKED,4><<<grid,32*WM*WN,0,stream>>>(xp,wp,yp,N,K,split);
    else kernel<WM,WN,NT,PACKED,1><<<grid,32*WM*WN,0,stream>>>(xp,wp,yp,N,K,split);
}
at::Tensor run(const at::Tensor& x,const at::Tensor& w,int64_t n,int64_t split,int64_t tile,int64_t unroll,bool packed) {
    TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.is_contiguous()&&w.is_contiguous());
    TORCH_CHECK(x.device()==w.device());
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kUInt32);
    TORCH_CHECK(x.numel()==64*x.size(-1) && x.size(-1)%64==0);
    TORCH_CHECK(split==1||split==4);
    TORCH_CHECK(tile>=0&&tile<=3 && n>0 && n%32==0);
    TORCH_CHECK(tile!=2 || n%64==0);
    TORCH_CHECK(unroll==0 || unroll==1 || unroll==4);
    TORCH_CHECK(w.numel()==n/8*(x.size(-1)/16)*(packed?52:64));
    c10::cuda::CUDAGuard guard(x.device());
    auto y=at::empty({split,64,n},x.options());
    auto stream=at::cuda::getCurrentCUDAStream();
    int k=x.size(-1);
    #define DISPATCH(P) switch(tile) { \
      case 0: launch<2,2,2,P>(x,w,y,n,k,split,unroll,stream);break; \
      case 1: launch<4,1,4,P>(x,w,y,n,k,split,unroll,stream);break; \
      case 2: launch<2,2,4,P>(x,w,y,n,k,split,unroll,stream);break; \
      case 3: launch<4,1,2,P>(x,w,y,n,k,split,unroll,stream);break; }
    if(packed) { DISPATCH(true); } else { DISPATCH(false); }
    #undef DISPATCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
TORCH_LIBRARY(laya_native_lossless,m) {
    m.def("decode(Tensor w, int n, int k, bool packed) -> Tensor", &decode);
    m.def("run(Tensor x, Tensor w, int n, int split, int tile, int unroll, bool packed) -> Tensor", &run);
}
