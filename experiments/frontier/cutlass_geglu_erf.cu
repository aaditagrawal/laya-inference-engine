// Same native CUTLASS geometry and output layout, with exact erf GEGLU.
// CUTLASS integration licensing: cutlass_geglu_LICENSE / cutlass_geglu_NOTICE.
#include "cutlass_geglu_core.cuh"
#include "cutlass_geglu_erf_math.cuh"
#include "cutlass_geglu_erf_corrections.cuh"

template<class Iterator>
struct ErfVisitor : GeGLUVisitor<Iterator,true> {
  using Parent=GeGLUVisitor<Iterator,true>;
  using Parent::Parent;
  using Output=typename Parent::Output;
  using Fragment=typename Parent::AccumulatorFragment;
  using Map=typename Iterator::ThreadMap;
  CUTLASS_DEVICE void visit(int,int,int,int fragment,Fragment const& acc) {
    auto coord=this->coordinate_iterator.thread_start()+Map::iteration_offset(fragment);
    Output values;
    #pragma unroll
    for(int j=0;j<Parent::kElementsPerAccess/2;j++) {
      auto activation=__float2bfloat16_rn(acc[2*j]);
      auto gate=__float2bfloat16_rn(acc[2*j+1]);
      auto gelu=corrected_erf_gelu(activation);
      values[j]=Bf16(__fmul_rn(__bfloat162float(gelu),__bfloat162float(gate)));
    }
    bool valid=coord.row()<64 && coord.column()<5248;
    int offset=coord.row()*2624+coord.column()/2;
    cutlass::arch::global_store<Output,sizeof(Output)>(values,this->output+offset,valid);
  }
};

using ErfTraits=Traits<32,64,2,true>;
using ErfMma=typename ErfTraits::Mma;
using ErfVisit=ErfVisitor<typename ErfTraits::Base::Epilogue::OutputTileIterator>;
using ErfEpilogue=typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
  ErfVisit,typename ErfTraits::Base::Epilogue>::Epilogue;
union ErfStorage {
  typename ErfMma::SharedStorage mainloop;
  typename ErfEpilogue::SharedStorage epilogue;
};

__global__ __launch_bounds__(128)
void cutlass_erf_kernel(Bf16 const* x,Bf16 const* w,Bf16 const* lut,Bf16* y,ErfTraits::Params params) {
  extern __shared__ __align__(16) char smem[];
  auto& storage=*reinterpret_cast<ErfStorage*>(smem);
  int thread=threadIdx.x,warp=__shfl_sync(0xffffffff,thread/32,0),lane=thread%32;
  typename ErfMma::IteratorA a(params.a,const_cast<Bf16*>(x),{64,1024},thread,{int(blockIdx.x)*32,0});
  typename ErfMma::IteratorB b(params.b,const_cast<Bf16*>(w),{1024,5248},thread,{0,int(blockIdx.y)*64});
  ErfMma mma(storage.mainloop,thread,warp,lane);
  ErfMma::FragmentC accum;
  accum.clear();
  mma(16,accum,a,b,accum);
  __syncthreads();
  ErfVisit visitor(y,lut,thread,{int(blockIdx.x)*32,int(blockIdx.y)*64});
  ErfEpilogue epilogue(storage.epilogue,thread,warp,lane);
  epilogue(visitor,accum);
}

__global__ void corrected_domain(unsigned short* output) {
  int code=blockIdx.x*blockDim.x+threadIdx.x;
  output[code]=__bfloat16_as_ushort(corrected_erf_gelu(__ushort_as_bfloat16(code)));
}
extern "C" int cutlass_geglu_corrected_domain(void* output,cudaStream_t stream) {
  corrected_domain<<<256,256,0,stream>>>(static_cast<unsigned short*>(output));
  return cudaGetLastError();
}
extern "C" int cutlass_geglu_erf(void const* x,void const* w,void const* lut,void* y,
                                  int mode,cudaStream_t stream,int* info) {
  if(mode==0)return run<32,64,2,true>(x,w,lut,y,stream,info);
  if(mode!=1)return cudaErrorInvalidValue;
  if(info) {
    cudaFuncAttributes a;
    auto error=cudaFuncGetAttributes(&a,cutlass_erf_kernel);
    if(error)return error;
    int active=0;
    error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,cutlass_erf_kernel,128,sizeof(ErfStorage));
    if(error)return error;
    info[0]=a.numRegs;info[1]=sizeof(ErfStorage);info[2]=128;info[3]=active;info[4]=a.localSizeBytes;
    return 0;
  }
  cutlass_erf_kernel<<<dim3(2,82),128,sizeof(ErfStorage),stream>>>(static_cast<Bf16 const*>(x),
    static_cast<Bf16 const*>(w),static_cast<Bf16 const*>(lut),static_cast<Bf16*>(y),ErfTraits::Params());
  return cudaGetLastError();
}
