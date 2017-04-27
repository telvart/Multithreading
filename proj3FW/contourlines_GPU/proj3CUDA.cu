
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include "proj3CUDA.h"

__global__ void hello(){
  printf("Hello, CUDA!\n");
}

void run()
{

}

__global__ void countKernel(float* vertexes, int numRows, int numCols, int* expectedCount, float level)
{
  //printf("%d", *expectedEdges);
  //*expectedEdges += 1;

   int x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int y = (blockIdx.y * blockDim.y) + threadIdx.y;
   int i = y * gridDim.x * blockDim.x + x;
   int myRow = floorf(i/numCols);
   int myCol = i%numCols;

   if( (i < (numRows*numCols)) && (myRow < numCols-1) && (myCol < numRows -1))
   {
     int localCount = 0;
     //int tLeft = i;
     float tLeft = vertexes[i];
     //int tRight = i+1;
     float tRight = vertexes[i+1];
     //int bLeft = i + numRows;
     float bLeft = vertexes[i+numRows];
     //int bRight = i + numRows + 1;
     float bRight = vertexes[i + numRows + 1];

     int numAbove = 0;
     int numBelow = 0;

     if(tLeft > level) {numAbove++;}
     if(tRight > level) {numAbove++;}
     if(bLeft > level) {numAbove++;}
     if(bRight > level) {numAbove++;}

     if(tLeft < level) {numBelow++;}
     if(tRight < level) {numBelow++;}
     if(bLeft < level) {numBelow++;}
     if(bRight < level) {numBelow++;}

     if(numAbove == 4 || numBelow == 4) {localCount = 0;}
     if(numAbove == 3 || numBelow == 3) {localCount = 1;}
     if(numAbove == numBelow) {localCount = 2;}

     //printf("My values are: TL = %f, TR = %f, BL = %f, BR = %f\n", tLeft, tRight, bLeft, bRight);
     //printf("My indexes are: TL = %i, TR = %i, BL = %i, BR = %i\n", tLeft, tRight, bLeft, bRight);
     //vertical cases
    //  if((level > tLeft && level < tRight) && (level > bLeft && level < bRight)){
    //    localCount++;
    //  }
    //  //horizontal cases
    //  if ((level > tLeft && level < bLeft) && (level > tRight && level < bRight)){
    //    localCount++;
    //  }
     //top to right

     //left to bottom

     //left to top

     //bottom to right
     atomicAdd(expectedCount, localCount);
    // printf("IM A THREAD\n");
   }

  // printf("x:%d y:%d\n", x,y);

  //NOTE: FOR VERTEXES ADDRESS; POSITION vertex[i][j] of dimensions [N][M] is i*M + j


  // for(int i=0; i<numRows * numCols; i++){
  //   printf("%.1f ", vertexes[i]);
  // }

  //printf("\n");
}

__global__ void computeKernel()
{

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
             <<"\nRegisters Per Block: "<<prop.regsPerBlock
             <<"\nMax Threads Per Block: "<<prop.maxThreadsPerBlock
             <<"\nMax Thread dimension (x,y,z): ("<<prop.maxThreadsDim[0]<<","<<prop.maxThreadsDim[1]<<","<<prop.maxThreadsDim[2]<<")"
             <<"\nMax Grid Size (x,y,z): "<<prop.maxGridSize[0]<<","<<prop.maxGridSize[1]<<","<<prop.maxGridSize[2]
             <<"\nNumber of Processors: "<<prop.multiProcessorCount
             <<"\nCompute Capability: "<<prop.major<<"\n";
  }


}

int launchCountKernel(float* h_vertexes, int numRows, int numCols, float level)
{
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  float* d_vertexes;
  int* temp = 0;
  int h_theCount;
  int* d_theCount;

  int size = (numRows * numCols) * sizeof(float);

  cudaMalloc(&d_vertexes, size);
  cudaMalloc(&d_theCount, sizeof(int)); // allocate GPU memory for the count, and vertex array

  cudaMemcpy(d_vertexes, h_vertexes, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_theCount, &temp, sizeof(int), cudaMemcpyHostToDevice); //transfer data to the GPU

  int warpSize = p.warpSize;
  int maxWarpsinBlock = p.maxThreadsPerBlock / warpSize;

  dim3 block(warpSize, maxWarpsinBlock);

  int gridX = (int)(ceil((numRows)/warpSize)) + 1;
  int gridY = (int)ceil((numCols)/maxWarpsinBlock) + 1;
  //std::cout<<"GridX: "<<gridX<<"\nGridY: "<<gridY<<"\n";
  dim3 gridDim(gridX, gridY);

  // printf("Block Dimension (x,y,z): (%i, %i, %i)\n", gridDim.x, gridDim.y, gridDim.z);
  // printf("Thread Dim (x,y,z): (%i, %i, %i)\n", block.x, block.y, block.z);
  // printf("Block Size: %i\n\n", block.x*block.y*block.z);

  //std::cout<<"WarpSize: "<<warpSize<<"\nMaxWarps: "<<maxWarpsinBlock<<"\n";

  // for(int i=0; i<numVerticies; i++){
  //   printf("%.1f ", h_vertexes[i]);
  // }

  //dim3 threadsBlock(3,3);
  countKernel<<<gridDim,block>>>(d_vertexes, numRows, numCols, d_theCount, level);
  //countKernel<<<1,1>>>(d_vertexes, numRows, numCols, d_theCount, level);
  cudaThreadSynchronize();
  cudaMemcpy(&h_theCount, d_theCount, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_vertexes);
  cudaFree(d_theCount);
  std::cout<<h_theCount<<"\n";

  return 2;
  //return h_theCount;


}

int launchComputeKernel()
{
  int k=0;
  computeKernel<<<1,1>>>();
  cudaThreadSynchronize();
  return k;
}
