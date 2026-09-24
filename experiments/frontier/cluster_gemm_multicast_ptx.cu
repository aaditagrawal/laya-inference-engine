// Compile-only instruction support probe. This kernel is never launched.
// Original experiment code, MIT.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

extern "C" __global__ void multicast_instruction_probe(const __grid_constant__ CUtensorMap map){
  extern __shared__ __align__(128) char storage[];
  uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(storage));
  uint32_t barrier=dst+8192;
  uint16_t mask=3;
  // Compile only: mbarrier initialization and lifecycle deliberately omitted.
  // An executable version must establish those and cluster-wide ownership.
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2,%3}], [%4], %5;"
    :: "r"(dst),"l"(&map),"r"(0),"r"(0),"r"(barrier),"h"(mask):"memory");
}

extern "C" __global__ void ordinary_instruction_probe(const __grid_constant__ CUtensorMap map){
  extern __shared__ __align__(128) char storage[];
  uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(storage));
  uint32_t barrier=dst+8192;
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
    :: "r"(dst),"l"(&map),"r"(0),"r"(0),"r"(barrier):"memory");
}
