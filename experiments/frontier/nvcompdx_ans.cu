/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from nvCOMPDx's Apache-2.0 ANS block examples. The experiment adds
 * contiguous-bank addressing, compact streams, exact reconstruction and a
 * matched raw/shared/checksum control. See nvcompdx_NOTICE and nvcompdx_LICENSE.
 */
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <nvcompdx.hpp>
#include <cstdint>
#include <cassert>

using namespace nvcompdx;
using U64=unsigned long long;
template<int CHUNK,int BLOCK,bool HALF,direction D>
using Codec=decltype(Algorithm<algorithm::ans>()+DataType<HALF?datatype::float16:datatype::uint8>()+
  Direction<D>()+MaxUncompChunkSize<CHUNK>()+Block()+BlockWarp<BLOCK/32,true>()+SM<1200>());

template<int CHUNK,int BLOCK,bool HALF>
__global__ void compress_bank(const uint8_t* __restrict__ input,uint8_t* __restrict__ output,
  U64 stride,U64* __restrict__ sizes,uint8_t* __restrict__ scratch) {
  using C=Codec<CHUNK,BLOCK,HALF,direction::compress>;
  NVCOMPDX_SKIP_IF_NOT_APPLICABLE_SM(C);
  extern __shared__ __align__(16) unsigned char shared[];
  C().execute(input+U64(blockIdx.x)*CHUNK,output+U64(blockIdx.x)*stride,CHUNK,sizes+blockIdx.x,
    shared,C().tmp_size_group()?scratch+U64(blockIdx.x)*C().tmp_size_group():nullptr);
}

template<int CHUNK,int BLOCK,bool HALF,bool DECOMPRESS,bool RESTORE>
__global__ __launch_bounds__(BLOCK)
void shared_checksum(const uint8_t* __restrict__ input,const U64* __restrict__ offsets,
  const U64* __restrict__ sizes,uint8_t* __restrict__ restored,U64* __restrict__ checksum,
  U64* __restrict__ decoded_sizes) {
  using D=Codec<CHUNK,BLOCK,HALF,direction::decompress>;
  NVCOMPDX_SKIP_IF_NOT_APPLICABLE_SM(D);
  extern __shared__ __align__(16) unsigned char scratch[];
  __shared__ __align__(16) unsigned char output[CHUNK];
  __shared__ U64 output_size;
  if constexpr(DECOMPRESS) {
    static_assert(D().tmp_size_group()==0);
    D().execute(input+offsets[blockIdx.x],output,sizes[blockIdx.x],&output_size,scratch,nullptr);
  } else {
    for(int i=threadIdx.x;i<CHUNK/16;i+=BLOCK) {
      uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(output+i*16));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst),"l"(input+U64(blockIdx.x)*CHUNK+i*16) : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    if(threadIdx.x==0)output_size=CHUNK;
  }
  // Required by the vendor API before another thread consumes decoded bytes.
  __syncthreads();
  if(threadIdx.x==0)decoded_sizes[blockIdx.x]=output_size;
  if constexpr(RESTORE) {
    for(int i=threadIdx.x;i<CHUNK/16;i+=BLOCK)
      reinterpret_cast<uint4*>(restored+U64(blockIdx.x)*CHUNK)[i]=reinterpret_cast<const uint4*>(output)[i];
  }
  U64 local=0;
  for(int i=threadIdx.x;i<CHUNK/4;i+=BLOCK)local+=reinterpret_cast<const uint32_t*>(output)[i];
  using Reduce=cub::BlockReduce<U64,BLOCK>;
  auto& temp=*reinterpret_cast<typename Reduce::TempStorage*>(scratch);
  U64 sum=Reduce(temp).Sum(local);
  if(threadIdx.x==0)checksum[blockIdx.x]=sum;
}

__global__ void compact_bank(const uint8_t* input,U64 stride,const U64* offsets,const U64* sizes,uint8_t* output) {
  U64 size=sizes[blockIdx.x];
  for(U64 i=threadIdx.x;i<size;i+=blockDim.x)output[offsets[blockIdx.x]+i]=input[U64(blockIdx.x)*stride+i];
}
extern "C" int nvdx_compact(const void* input,U64 stride,const void* offsets,const void* sizes,void* output,int count,void* stream) {
  compact_bank<<<count,256,0,(cudaStream_t)stream>>>((const uint8_t*)input,stride,(const U64*)offsets,(const U64*)sizes,(uint8_t*)output);
  return cudaGetLastError();
}

template<int CHUNK,int BLOCK,bool HALF>
int info(U64* out) {
  using C=Codec<CHUNK,BLOCK,HALF,direction::compress>;
  using D=Codec<CHUNK,BLOCK,HALF,direction::decompress>;
  out[0]=C().max_comp_chunk_size(); out[1]=C().shmem_size_group();out[2]=C().tmp_size_group();
  out[3]=D().shmem_size_group();out[4]=D().tmp_size_group();out[5]=C().input_alignment();
  out[6]=C().output_alignment();out[7]=D().input_alignment();out[8]=D().output_alignment();
  out[9]=D().shmem_alignment();out[10]=CHUNK;out[11]=BLOCK;
  cudaFuncAttributes attr;
  auto kernel=shared_checksum<CHUNK,BLOCK,HALF,true,false>;
  cudaError_t error=cudaFuncGetAttributes(&attr,kernel);
  if(error!=cudaSuccess)return error;
  out[12]=attr.numRegs;out[13]=attr.sharedSizeBytes;out[14]=attr.localSizeBytes;
  int active=0;error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,BLOCK,D().shmem_size_group());
  out[15]=active;return error;
}
template<int CHUNK,int BLOCK,bool HALF>
int compress(const void* input,void* output,U64 stride,void* sizes,void* scratch,int count,void* stream) {
  using C=Codec<CHUNK,BLOCK,HALF,direction::compress>;
  compress_bank<CHUNK,BLOCK,HALF><<<count,BLOCK,C().shmem_size_group(),(cudaStream_t)stream>>>(
    (const uint8_t*)input,(uint8_t*)output,stride,(U64*)sizes,(uint8_t*)scratch);
  return cudaGetLastError();
}
template<int CHUNK,int BLOCK,bool HALF>
int checksum(const void* input,const void* offsets,const void* sizes,void* restored,void* checksums,void* decoded_sizes,int count,int mode,void* stream) {
  using D=Codec<CHUNK,BLOCK,HALF,direction::decompress>;
  #define RUN(DEC,RESTORE) shared_checksum<CHUNK,BLOCK,HALF,DEC,RESTORE><<<count,BLOCK,D().shmem_size_group(),(cudaStream_t)stream>>>((const uint8_t*)input,(const U64*)offsets,(const U64*)sizes,(uint8_t*)restored,(U64*)checksums,(U64*)decoded_sizes)
  if(mode==2){RUN(true,true);}else if(mode==1){RUN(true,false);}else{RUN(false,false);}
  #undef RUN
  return cudaGetLastError();
}

#define PICK(CALL) if(chunk==4096){if(block==128){if(half)return CALL(4096,128,true);else return CALL(4096,128,false);}else{if(half)return CALL(4096,256,true);else return CALL(4096,256,false);}}else{if(block==128){if(half)return CALL(16384,128,true);else return CALL(16384,128,false);}else{if(half)return CALL(16384,256,true);else return CALL(16384,256,false);}}
extern "C" int nvdx_info(int chunk,int block,int half,U64* out) {
  #define CALL(C,B,H) info<C,B,H>(out)
  PICK(CALL)
  #undef CALL
}
extern "C" int nvdx_compress(int chunk,int block,int half,const void* input,void* output,U64 stride,void* sizes,void* scratch,int count,void* stream) {
  #define CALL(C,B,H) compress<C,B,H>(input,output,stride,sizes,scratch,count,stream)
  PICK(CALL)
  #undef CALL
}
extern "C" int nvdx_checksum(int chunk,int block,int half,const void* input,const void* offsets,const void* sizes,void* restored,void* checksums,void* decoded_sizes,int count,int mode,void* stream) {
  #define CALL(C,B,H) checksum<C,B,H>(input,offsets,sizes,restored,checksums,decoded_sizes,count,mode,stream)
  PICK(CALL)
  #undef CALL
}
#undef PICK
