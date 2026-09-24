// Local wrapper for the Apache-2.0 Turbo-Lossless adaptation.
// See turbo_NOTICE and turbo_LICENSE for attribution and modifications.
#include "turbo_kernel.cuh"
#include <cstdio>
#include <type_traits>

struct alignas(64) DescriptorSet {
  CUtensorMap sm, gr, x;
  int n, k, rows, tile;
};

extern "C" void* turbo_prepare(void* sm, void* gr, void* x, int n, int k,
                               int rows, int tile) {
  if (n%32 || k%32 || rows!=64 || (tile!=16&&tile!=32&&tile!=64))return nullptr;
  auto* d = new DescriptorSet;
  d->n=n; d->k=k; d->rows=rows; d->tile=tile;
  // Weight row tile is updated by a separate descriptor for each selected TM.
  return d;
}

extern "C" int turbo_descriptors(void* handle, void* sm, void* gr, void* x, int tm) {
  auto* d=static_cast<DescriptorSet*>(handle);
  cuuint64_t dims[2]={(cuuint64_t)d->k,(cuuint64_t)d->n};
  cuuint64_t strides[1]={(cuuint64_t)d->k};
  cuuint32_t box[2]={64,(cuuint32_t)tm}, step[2]={1,1};
  CUresult r=cuTensorMapEncodeTiled(&d->sm,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,sm,
      dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r)return r;
  dims[0]/=2;strides[0]/=2;box[0]/=2;
  r=cuTensorMapEncodeTiled(&d->gr,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,gr,
      dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r)return r;
  dims[0]=d->k;dims[1]=d->rows;strides[0]=d->k*2;box[0]=64;box[1]=d->tile;
  return cuTensorMapEncodeTiled(&d->x,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,x,
      dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
extern "C" void turbo_free(void* handle){delete static_cast<DescriptorSet*>(handle);}

template<int TN, int TM, bool EXACT>
void launch(const DescriptorSet& d, int base, void* out, const void* ro,
            const void* co, const void* cv, int split, cudaStream_t stream){
  constexpr int smem=TM*64*2+TM*32*2+64*TN*4+128;
  split12_fused_gemm_v3<TN,TN/8,TM,EXACT><<<dim3(64/TN,d.n/TM,split),TM*2,smem,stream>>>(
      d.sm,d.gr,d.x,base,(__nv_bfloat16*)out,d.n,d.k,64,d.n,split,
      (const int32_t*)ro,(const int32_t*)co,(const int16_t*)cv,nullptr);
}
extern "C" int turbo_run(void* handle,int base,void* out,const void* ro,
                         const void* co,const void* cv,int split,int tm,
                         int exact,void* stream){
  auto& d=*static_cast<DescriptorSet*>(handle);
  auto call=[&](auto TN,auto TM,auto EX){launch<decltype(TN)::value,decltype(TM)::value,
    decltype(EX)::value>(d,base,out,ro,co,cv,split,(cudaStream_t)stream);};
  auto tiles=[&](auto EX){
    if(tm==32){
      if(d.tile==16)call(std::integral_constant<int,16>{},std::integral_constant<int,32>{},EX);
      else if(d.tile==32)call(std::integral_constant<int,32>{},std::integral_constant<int,32>{},EX);
      else call(std::integral_constant<int,64>{},std::integral_constant<int,32>{},EX);
    }else{
      if(d.tile==16)call(std::integral_constant<int,16>{},std::integral_constant<int,64>{},EX);
      else if(d.tile==32)call(std::integral_constant<int,32>{},std::integral_constant<int,64>{},EX);
      else call(std::integral_constant<int,64>{},std::integral_constant<int,64>{},EX);
    }
  };
  if(exact)tiles(std::true_type{});else tiles(std::false_type{});
  return cudaGetLastError();
}

__global__ void decode_kernel(const uint8_t* sm,const uint8_t* gr,int base,
    const int32_t* ro,const int32_t* co,const int16_t* cv,uint16_t* y,int n,int k){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*k)return;
  int group=(gr[i/2]>>((i%2)*4))&15;
  uint16_t bits=((uint16_t)(sm[i]>>7)<<15)|((base+group)<<7)|(sm[i]&127);
  if(group==0)bits=escape_value(i/k,i%k,ro,co,cv);
  y[i]=bits;
}
extern "C" int turbo_decode(const void* sm,const void* gr,int base,const void* ro,
   const void* co,const void* cv,void* y,int n,int k,void* stream){
  decode_kernel<<<(n*k+255)/256,256,0,(cudaStream_t)stream>>>((const uint8_t*)sm,
    (const uint8_t*)gr,base,(const int32_t*)ro,(const int32_t*)co,
    (const int16_t*)cv,(uint16_t*)y,n,k);
  return cudaGetLastError();
}
