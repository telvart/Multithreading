
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include "proj3CUDA.h"
#include "ContourGenerator.h"

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

__global__ void countKernel(float* vertexes, int numRows, int numCols, int* expectedCount, float level)
{
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
    // printf("My indexes are: TL = %i, TR = %i, BL = %i, BR = %i\n", tLeft, tRight, bLeft, bRight);
     //vertical cases
    //  if((level > tLeft && level < tRight) && (level > bLeft && level < bRight)){
    //    localCount++;
    //  }
    //  //horizontal cases
    //  if ((level > tLeft && level < bLeft) && (level > tRight && level < bRight)){
    //    localCount++;
    //  }
     atomicAdd(expectedCount, localCount);
   }
}

__global__ void computeKernel(float* vertexes, int numRows, int numCols, int level, int* locCount, int* edgeCount, vec2* edgeBuf)
{

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int i = y * gridDim.x * blockDim.x + x;
  int myRow = floorf(i/numCols);
  int myCol = i%numCols;


  if( (i < (numRows*numCols)) && (myRow < numCols-1) && (myCol < numRows -1))
  {
      float tLeft = vertexes[i];
      //int tRight = i+1;
      float tRight = vertexes[i+1];
      //int bLeft = i + numRows;
      float bLeft = vertexes[i+numRows];
      //int bRight = i + numRows + 1;
      float bRight = vertexes[i + numRows + 1];

      int myLoc = atomicAdd(locCount, 4);
      //atomicAdd(edgeCount, 1);
      *edgeCount = 3;
      edgeBuf[0][0] = 0;
      edgeBuf[0][1] = 0;
      edgeBuf[1][0] = 3;
      edgeBuf[1][1] = 3;

      edgeBuf[2][0] = 0;
      edgeBuf[2][1] = 3;
      edgeBuf[3][0] = 3;
      edgeBuf[3][1] = 0;

      edgeBuf[4][0] = 1.5;
      edgeBuf[4][1] = 0;
      edgeBuf[5][0] = 1.5;
      edgeBuf[5][1] = 3;

      //printf("X: %.1f Y: %.1f\n", edgeBuf[0][0], edgeBuf[0][1]);
    // for(int i=0; i<numRows*numCols; i++)
    // {
    //   printf("%.1f ", vertexes[i]);
    // }
    // printf("\n");
  }
}



int launchComputeKernel(float* vertexes, int numRows, int numCols,
   float level, int expectedEdges, vec2* buf)
{

  //std::cout<<buf[0][0]<<" "<<buf[0][1];
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  float* d_vertexes;
  int* temp = 0;
  int h_edgeCount;
  int* d_edgeCount;
  int* d_locCounter;
  vec2* d_buffer;
  int size = (numRows * numCols) * sizeof(float);

  cudaMalloc(&d_vertexes, size);
  cudaMalloc(&d_edgeCount, sizeof(int));
  cudaMalloc(&d_locCounter, sizeof(int)); //allocate device memory for vertexes and counters
  cudaMalloc(&d_buffer, expectedEdges * sizeof(vec2));

  cudaMemcpy(d_vertexes, vertexes, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgeCount, &temp, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_locCounter, &temp, sizeof(int), cudaMemcpyHostToDevice); //
  cudaMemcpy(d_buffer, buf, expectedEdges * sizeof(vec2), cudaMemcpyHostToDevice);

  int warpSize = p.warpSize;
  int maxWarpsinBlock = p.maxThreadsPerBlock / warpSize; //determine #warps in one block

  int gridX = (int)(ceil((numRows)/warpSize)) + 1;
  int gridY = (int)ceil((numCols)/maxWarpsinBlock) + 1;

  dim3 block(warpSize, maxWarpsinBlock);
  dim3 gridDim(gridX, gridY); //launching dimensions

  computeKernel<<<gridDim, block>>>(d_vertexes, numRows, numCols, level, d_locCounter, d_edgeCount, d_buffer);
  cudaThreadSynchronize();

  cudaMemcpy(&h_edgeCount, d_edgeCount, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(buf, d_buffer, expectedEdges * sizeof(vec2), cudaMemcpyDeviceToHost);

  cudaFree(d_vertexes);
  cudaFree(d_edgeCount);
  cudaFree(d_locCounter);
  cudaFree(d_buffer);

  std::cout<<h_edgeCount<<" \n";

  return h_edgeCount;
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
  int maxWarpsinBlock = p.maxThreadsPerBlock / warpSize; //determine #warps in one block

  int gridX = (int)(ceil((numRows)/warpSize)) + 1;
  int gridY = (int)ceil((numCols)/maxWarpsinBlock) + 1;

  dim3 block(warpSize, maxWarpsinBlock);
  dim3 gridDim(gridX, gridY); //launching dimensions

  // printf("Block Dimension (x,y,z): (%i, %i, %i)\n", gridDim.x, gridDim.y, gridDim.z);
  // printf("Thread Dim (x,y,z): (%i, %i, %i)\n", block.x, block.y, block.z);
  // printf("Block Size: %i\n\n", block.x*block.y*block.z);

  //std::cout<<"WarpSize: "<<warpSize<<"\nMaxWarps: "<<maxWarpsinBlock<<"\n";

  countKernel<<<gridDim,block>>>(d_vertexes, numRows, numCols, d_theCount, level);
  cudaThreadSynchronize();
  cudaMemcpy(&h_theCount, d_theCount, sizeof(int), cudaMemcpyDeviceToHost); //bring back the estimated count

  cudaFree(d_vertexes);
  cudaFree(d_theCount); //free device memory
//  std::cout<<h_theCount<<"\n";

  return h_theCount;
}
