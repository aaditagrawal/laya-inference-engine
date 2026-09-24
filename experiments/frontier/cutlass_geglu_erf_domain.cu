#include "cutlass_geglu_erf_math.cuh"

__global__ void erf_domain_kernel(unsigned short* output) {
  int code=blockIdx.x*blockDim.x+threadIdx.x;
  if(code<65536)output[code]=__bfloat16_as_ushort(native_erf_gelu(__ushort_as_bfloat16(code)));
}
extern "C" int cutlass_geglu_erf_domain(void* output,cudaStream_t stream) {
  erf_domain_kernel<<<256,256,0,stream>>>(static_cast<unsigned short*>(output));
  return cudaGetLastError();
}
