// Native CUTLASS BF16 mainloop with an exact adjacent-pair GEGLU visitor.
// Mainloop/visitor integration follows NVIDIA CUTLASS, BSD-3-Clause.
// See cutlass_geglu_LICENSE and cutlass_geglu_NOTICE.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Bf16 = cutlass::bfloat16_t;

template<class Iterator, bool Fused>
struct GeGLUVisitor {
  static int const kIterations = Iterator::kIterations;
  static int const kElementsPerAccess = Iterator::kElementsPerAccess;
  using AccumulatorFragment = cutlass::Array<float, kElementsPerAccess>;
  using ThreadMap = typename Iterator::ThreadMap;
  using Output = cutlass::AlignedArray<Bf16, Fused ? kElementsPerAccess/2 : kElementsPerAccess>;
  Iterator coordinate_iterator;
  Bf16* output;
  Bf16 const* lut;

  CUTLASS_DEVICE GeGLUVisitor(Bf16* y, Bf16 const* table, int thread,
                              cutlass::MatrixCoord tile):
    coordinate_iterator(typename Iterator::Params(cutlass::layout::RowMajor(5248)),
                        nullptr, {64,5248}, thread, tile), output(y), lut(table) {}
  CUTLASS_DEVICE void begin_epilogue() {}
  CUTLASS_DEVICE void begin_step(int) {}
  CUTLASS_DEVICE void begin_row(int) {}
  CUTLASS_DEVICE void end_row(int) {}
  CUTLASS_DEVICE void end_step(int) { ++coordinate_iterator; }
  CUTLASS_DEVICE void end_epilogue() {}
  CUTLASS_DEVICE void visit(int, int, int, int fragment, AccumulatorFragment const& acc) {
    auto coord = coordinate_iterator.thread_start() + ThreadMap::iteration_offset(fragment);
    Output values;
    if constexpr(Fused) {
      #pragma unroll
      for(int j=0;j<kElementsPerAccess/2;j++) {
        auto activation = __float2bfloat16_rn(acc[2*j]);
        auto gate = __float2bfloat16_rn(acc[2*j+1]);
        uint16_t code = __bfloat16_as_ushort(activation);
        float gelu = static_cast<float>(lut[code]);
        values[j] = Bf16(__fmul_rn(gelu, __bfloat162float(gate)));
      }
    } else {
      #pragma unroll
      for(int j=0;j<kElementsPerAccess;j++) values[j] = Bf16(acc[j]);
    }
    bool valid = coord.row()<64 && coord.column()<5248;
    int offset = Fused ? coord.row()*2624+coord.column()/2 : coord.row()*5248+coord.column();
    cutlass::arch::global_store<Output, sizeof(Output)>(values, output+offset, valid);
  }
};

template<int BM,int BN,int Stages,bool Fused>
struct Traits {
  static int const WM = BN==32 ? 16 : BM/2;
  static int const WN = BN==32 ? 32 : BN/2;
  using Shape = cutlass::gemm::GemmShape<BM,BN,64>;
  using Warp = cutlass::gemm::GemmShape<WM,WN,64>;
  using Base = typename cutlass::gemm::kernel::DefaultGemm<
    Bf16,cutlass::layout::RowMajor,8,
    Bf16,cutlass::layout::ColumnMajor,8,
    Bf16,cutlass::layout::RowMajor,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    Shape,Warp,cutlass::gemm::GemmShape<16,8,16>,
    cutlass::epilogue::thread::LinearCombination<Bf16,8,float,float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,false,cutlass::arch::OpMultiplyAdd,
    cutlass::gemm::SharedMemoryClearOption::kZfill>::GemmKernel;
  using Mma = typename Base::Mma;
  using Visitor = GeGLUVisitor<typename Base::Epilogue::OutputTileIterator,Fused>;
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    Visitor,typename Base::Epilogue>::Epilogue;
  static int const Threads = Mma::WarpCount::kCount*32;
  union SharedStorage {
    typename Mma::SharedStorage mainloop;
    typename Epilogue::SharedStorage epilogue;
  };
  struct Params {
    typename Mma::IteratorA::Params a;
    typename Mma::IteratorB::Params b;
    Params():a(cutlass::layout::RowMajor(1024)),b(cutlass::layout::ColumnMajor(1024)){}
  };
};

template<int BM,int BN,int Stages,bool Fused>
__global__ __launch_bounds__(Traits<BM,BN,Stages,Fused>::Threads)
void cutlass_geglu_kernel(Bf16 const* x,Bf16 const* w,Bf16 const* lut,Bf16* y,
                         typename Traits<BM,BN,Stages,Fused>::Params params) {
  using T = Traits<BM,BN,Stages,Fused>;
  using Mma = typename T::Mma;
  extern __shared__ __align__(16) char smem[];
  auto& storage = *reinterpret_cast<typename T::SharedStorage*>(smem);
  int thread=threadIdx.x, warp=__shfl_sync(0xffffffff,thread/32,0), lane=thread%32;
  // CUTLASS read iterators use mutable pointer typedefs but only load operands.
  typename Mma::IteratorA a(params.a,const_cast<Bf16*>(x),{64,1024},thread,{int(blockIdx.x)*BM,0});
  typename Mma::IteratorB b(params.b,const_cast<Bf16*>(w),{1024,5248},thread,{0,int(blockIdx.y)*BN});
  Mma mma(storage.mainloop,thread,warp,lane);
  typename Mma::FragmentC accum;
  accum.clear();
  mma(16,accum,a,b,accum);
  __syncthreads();
  typename T::Visitor visitor(y,lut,thread,{int(blockIdx.x)*BM,int(blockIdx.y)*BN});
  typename T::Epilogue epilogue(storage.epilogue,thread,warp,lane);
  epilogue(visitor,accum);
}

template<int BM,int BN,int Stages,bool Fused>
int run(void const* x,void const* w,void const* lut,void* y,cudaStream_t stream,int* info) {
  using T=Traits<BM,BN,Stages,Fused>;
  auto kernel=cutlass_geglu_kernel<BM,BN,Stages,Fused>;
  constexpr int shared=sizeof(typename T::SharedStorage);
  if constexpr(shared>49152) {
    static bool configured=false;
    if(!configured) {
      auto error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
      if(error)return error;
      configured=true;
    }
  }
  if(info) {
    cudaFuncAttributes a;
    auto error=cudaFuncGetAttributes(&a,kernel);
    if(error)return error;
    int active=0;
    error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,T::Threads,shared);
    if(error)return error;
    info[0]=a.numRegs; info[1]=shared; info[2]=T::Threads; info[3]=active; info[4]=a.localSizeBytes;
    return 0;
  }
  kernel<<<dim3((64+BM-1)/BM,(5248+BN-1)/BN),T::Threads,shared,stream>>>(
    static_cast<Bf16 const*>(x),static_cast<Bf16 const*>(w),static_cast<Bf16 const*>(lut),
    static_cast<Bf16*>(y),typename T::Params());
  return cudaGetLastError();
}

extern "C" int cutlass_geglu(void const* x,void const* w,void const* lut,void* y,
                              int tile,int fused,cudaStream_t stream,int* info) {
  #define CASE(I,M,N,S) case I: return fused ? run<M,N,S,true>(x,w,lut,y,stream,info) : run<M,N,S,false>(x,w,lut,y,stream,info)
  switch(tile) {
    CASE(0,32,64,3); CASE(1,32,64,2); CASE(2,32,64,4);
    CASE(3,64,64,2); CASE(4,64,64,3); CASE(5,64,64,4);
    CASE(6,32,32,3); CASE(7,64,32,3);
    CASE(8,32,128,2); CASE(9,32,128,3);
    CASE(10,64,128,2); CASE(11,64,128,3);
    default:return cudaErrorInvalidValue;
  }
}
