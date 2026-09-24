// Probe generic thread-block clusters separately from TMA multicast support.
// Original experiment code, MIT.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>

__global__ void cluster_probe(int* output){
  auto cluster=cooperative_groups::this_cluster();
  __shared__ int value;
  if(threadIdx.x==0)value=100+cluster.block_rank();
  cluster.sync();
  if(threadIdx.x==0){
    int rank=cluster.block_rank(),size=cluster.num_blocks();
    int* peer=cluster.map_shared_rank(&value,(rank+1)%size);
    output[rank*3]=rank;
    output[rank*3+1]=size;
    output[rank*3+2]=*peer;
  }
  cluster.sync();
}

int main(){
  int device=0,cluster_attribute=-1,runtime=0,driver=0;
  cudaError_t init=cudaSetDevice(device);
  if(init!=cudaSuccess){std::printf("{\"initialization_error\":%d}\n",int(init));return 1;}
  cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,device);
  cudaDeviceGetAttribute(&cluster_attribute,cudaDevAttrClusterLaunch,device);
  cudaRuntimeGetVersion(&runtime);cudaDriverGetVersion(&driver);
  cudaLaunchConfig_t config{};config.gridDim=dim3(8,1,1);config.blockDim=dim3(128,1,1);
  int potential=-1;
  cudaError_t potential_status=cudaOccupancyMaxPotentialClusterSize(&potential,cluster_probe,&config);
  std::printf("{\"device\":\"%s\",\"compute_capability\":[%d,%d],\"runtime\":%d,\"driver\":%d,\"cluster_launch_attribute\":%d,\"max_potential_cluster_size\":%d,\"potential_query_status\":%d,\"launches\":[",prop.name,prop.major,prop.minor,runtime,driver,cluster_attribute,potential,int(potential_status));
  bool first=true;
  for(int size: {1,2,4,8}){
    int* output=nullptr;cudaMalloc(&output,24*sizeof(int));cudaMemset(output,0,24*sizeof(int));
    config.gridDim=dim3(size,1,1);
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=size;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;
    config.attrs=&attr;config.numAttrs=1;
    int active=-1;
    cudaError_t occ=cudaOccupancyMaxActiveClusters(&active,cluster_probe,&config);
    cudaError_t launch=cudaLaunchKernelEx(&config,cluster_probe,output);
    cudaError_t sync=launch==cudaSuccess?cudaDeviceSynchronize():launch;
    int host[24]={};
    if(sync==cudaSuccess)cudaMemcpy(host,output,sizeof(host),cudaMemcpyDeviceToHost);
    bool exact=sync==cudaSuccess;
    for(int rank=0;rank<size;rank++)exact&=host[rank*3]==rank&&host[rank*3+1]==size&&host[rank*3+2]==100+(rank+1)%size;
    std::printf("%s{\"size\":%d,\"active_clusters\":%d,\"occupancy_status\":%d,\"launch_status\":%d,\"sync_status\":%d,\"cross_block_shared_read_exact\":%s}",first?"":",",size,active,int(occ),int(launch),int(sync),exact?"true":"false");
    first=false;cudaFree(output);cudaGetLastError();
  }
  std::printf("]}\n");
  return 0;
}
