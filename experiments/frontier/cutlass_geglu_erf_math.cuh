// Native CUDA GELU expression, with explicit scalar rounding boundaries.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ __nv_bfloat16 native_erf_gelu(__nv_bfloat16 input) {
  float x=__bfloat162float(input);
  float half=__fmul_rn(0.5f,x);
  float z=__fmul_rn(x,0.7071067811865476f);
  float factor=__fadd_rn(1.f,erff(z));
  return __float2bfloat16_rn(__fmul_rn(half,factor));
}
