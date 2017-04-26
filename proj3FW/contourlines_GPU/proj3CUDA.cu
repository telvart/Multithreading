
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include "proj3CUDA.h"

__global__ void hello(){
  printf("Hello, CUDA!\n");
}

void run()
{

}

__global__ void countKernel(float* vertexes, int numVerticies, int* expectedEdges, int level)
{
  //printf("%d", *expectedEdges);
  //*expectedEdges += 1;
  atomicAdd(expectedEdges, 4);
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  printf("x:%d y:%d\n", x,y);

  //TODO: FOR VERTEXES ADDRESS; POSITION vertex[i][j] of dimensions [N][M] is i*M + j


  // for(int i=0; i<numVerticies; i++){
  //   printf("%.1f ", vertexes[i]);
  // }

  //printf("\n");
}

__global__ void computeKernel(int& i)
{
  i++;
}

void queryDevice(){

  int numDevices;
  cudaGetDeviceCount(&numDevices);
  for(int i=0; i<numDevices; i++){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout<<"\nDevice Index: "<<i
             <<"\nDevice Name: "<<prop.name
             <<"\nTotal Global Memory (bytes): "<<prop.totalGlobalMem
             <<"\nWarp Size: "<<prop.warpSize
             <<"\nMax Threads Per Block: "<<prop.maxThreadsPerBlock
             <<"\nMax Thread dimension (x,y,z): ("<<prop.maxThreadsDim[0]<<","<<prop.maxThreadsDim[1]<<","<<prop.maxThreadsDim[2]<<")"
             <<"\nMax Grid Size (x,y,z): "<<prop.maxGridSize[0]<<","<<prop.maxGridSize[1]<<","<<prop.maxGridSize[2]
             <<"\nNumber of Processors: "<<prop.multiProcessorCount
             <<"\nCompute Capability: "<<prop.major<<"\n";
  }


}

int launchCountKernel(float* h_vertexes, int numVerticies)
{

  float* d_vertexes;
  int* temp = 0;
  int h_theCount;
  int* d_theCount;

  int size = numVerticies * sizeof(float);

  cudaMalloc(&d_vertexes, size);
  cudaMalloc(&d_theCount, sizeof(int));

  cudaMemcpy(d_vertexes, h_vertexes, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_theCount, &temp, sizeof(int), cudaMemcpyHostToDevice);


  // for(int i=0; i<numVerticies; i++){
  //   printf("%.1f ", h_vertexes[i]);
  // }

  dim3 threadsBlock(3,3);
  countKernel<<<1,threadsBlock>>>(d_vertexes, numVerticies, d_theCount, 0);
  cudaThreadSynchronize();
  cudaMemcpy(&h_theCount, d_theCount, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_vertexes);
  cudaFree(d_theCount);
  //std::cout<<h_theCount<<" ";

  return 2;
  //return h_theCount;




}
int launchComputeKernel()
{
  int k=0;
  computeKernel<<<1,1>>>(k);
  cudaThreadSynchronize();
  return k;
}
