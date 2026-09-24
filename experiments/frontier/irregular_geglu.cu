// Irregular CUTLASS BF16 tiles with checked loader coverage and exact GEGLU.
// IRREGULAR_DIRECT uses the pinned MMA accumulator mapping directly.
// It changes output staging only; every MMA and rounding remains unchanged.
// Mainloop/visitor integration follows NVIDIA CUTLASS, BSD-3-Clause.
// See irregular_geglu_LICENSE and irregular_geglu_NOTICE.
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
  static int const WM = IRREGULAR_WM;
  static int const WN = IRREGULAR_WN;
  using Shape = cutlass::gemm::GemmShape<BM,BN,64>;
  using Warp = cutlass::gemm::GemmShape<WM,WN,64>;
  static_assert(BM % WM == 0 && BN % WN == 0, "Exact warp tiling required");
  static_assert(WM % 16 == 0 && WN % 8 == 0, "Exact MMA tiling required");
#if IRREGULAR_DIRECT
  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
    Bf16,cutlass::layout::RowMajor,8,
    Bf16,cutlass::layout::ColumnMajor,8,
    float,cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    Shape,Warp,cutlass::gemm::GemmShape<16,8,16>,Stages,
    cutlass::arch::OpMultiplyAdd,false,
    cutlass::gemm::SharedMemoryClearOption::kZfill>::ThreadblockMma;
  struct SharedStorage { typename Mma::SharedStorage mainloop; };
#else
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
  union SharedStorage {
    typename Mma::SharedStorage mainloop;
    typename Epilogue::SharedStorage epilogue;
  };
  using Out = typename Base::Epilogue::OutputTileIterator;
  using OutMap = typename Out::ThreadMap;
  static_assert(Out::kIterations * OutMap::Iterations::kCount * OutMap::kElementsPerAccess * Mma::WarpCount::kCount * 32 == BM * BN,
                "Default epilogue does not cover output exactly");
  static_assert(OutMap::Detail::kAccessWidth * OutMap::Detail::kAccessRows == 32,
                "Default epilogue lane map does not cover one warp exactly");
#endif
  static int const Threads = Mma::WarpCount::kCount*32;
  using MapA = typename Mma::IteratorA::ThreadMap;
  using MapB = typename Mma::IteratorB::ThreadMap;
  static_assert(MapA::Iterations::kCount * MapA::kElementsPerAccess * Threads == BM * 64,
                "A loader does not cover its tile exactly");
  static_assert(MapB::Iterations::kCount * MapB::kElementsPerAccess * Threads == BN * 64,
                "B loader does not cover its tile exactly");
  static_assert(Mma::FragmentC::kElements * Threads == BM * BN,
                "Accumulator fragments do not cover the tile exactly");
  using WarpLoadA = typename Mma::Operator::IteratorA::Base::Policy;
  using WarpLoadB = typename Mma::Operator::IteratorB::Base::Policy;
  static_assert(WarpLoadA::LdsmIterations::kStrided * WarpLoadA::LdsmShape::kStrided * 8 == WM,
                "A warp ldmatrix does not cover warp rows exactly");
  static_assert(WarpLoadB::LdsmIterations::kStrided * WarpLoadB::LdsmShape::kStrided * 8 == WN,
                "B warp ldmatrix does not cover warp columns exactly");
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
#if IRREGULAR_DIRECT
  // See pinned mma_tensor_op_tile_iterator.h RowMajor accumulator iterator.
  int warp_m = warp % (BM / T::WM), warp_n = warp / (BM / T::WM);
  #pragma unroll
  for(int n=0;n<T::WN/8;n++) {
    #pragma unroll
    for(int m=0;m<T::WM/16;m++) {
      #pragma unroll
      for(int r=0;r<2;r++) {
        int index=4*(n*(T::WM/16)+m)+2*r;
        int row=int(blockIdx.x)*BM+warp_m*T::WM+m*16+r*8+lane/4;
        int col=int(blockIdx.y)*BN+warp_n*T::WN+n*8+(lane%4)*2;
        if(row<64 && col<5248) {
          auto activation=__float2bfloat16_rn(accum[index]);
          auto gate=__float2bfloat16_rn(accum[index+1]);
          if constexpr(Fused) {
            float gelu=static_cast<float>(lut[__bfloat16_as_ushort(activation)]);
            y[row*2624+col/2]=Bf16(__fmul_rn(gelu,__bfloat162float(gate)));
          } else {
            y[row*5248+col]=Bf16(__bfloat162float(activation));
            y[row*5248+col+1]=Bf16(__bfloat162float(gate));
          }
        }
      }
    }
  }
#else
  __syncthreads();
  typename T::Visitor visitor(y,lut,thread,{int(blockIdx.x)*BM,int(blockIdx.y)*BN});
  typename T::Epilogue epilogue(storage.epilogue,thread,warp,lane);
  epilogue(visitor,accum);
#endif
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

extern "C" int irregular_geglu(void const* x,void const* w,void const* lut,void* y,
                                int fused,cudaStream_t stream,int* info) {
  return fused ? run<IRREGULAR_BM,IRREGULAR_BN,IRREGULAR_STAGES,true>(x,w,lut,y,stream,info)
               : run<IRREGULAR_BM,IRREGULAR_BN,IRREGULAR_STAGES,false>(x,w,lut,y,stream,info);
}
