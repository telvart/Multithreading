
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
     float tLeft = vertexes[i];
     float tRight = vertexes[i+1];
     float bLeft = vertexes[i+numRows];
     float bRight = vertexes[i + numRows + 1];

     int localCount = 0;
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
     //atomicAdd(expectedCount, 2);
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
      float tRight = vertexes[i+1];
      float bLeft = vertexes[i+numRows];
      float bRight = vertexes[i + numRows + 1];

      // if(numAbove == 1 || numBelow == 1 || numAbove == numBelow)
      // {
        int myLoc;
        int numPoints = 0;
        vec2 oneEdge[2];
        //vec2 point1, point2;
        //float ratio;
         //if(numAbove != numBelow){
          // if(numAbove == 1 || numBelow == 1){
       float ratio;
       if((tLeft > 0 && tRight > 0) && ((level > tLeft && level < tRight) || (level < tLeft && level > tRight))){
         oneEdge[numPoints][0] = myCol + ((fabsf(level - tLeft)) / (fabsf(tLeft - tRight)));
         oneEdge[numPoints][1] = myRow + 1;
         numPoints++;
         //printf("Point (%.1f, %.1f)\n", oneEdge[numPoints-1][0], oneEdge[numPoints-1][1]);
       }
       if((tLeft > 0 && bLeft > 0) && ((level < tLeft && level > bLeft) || (level > tLeft && level < bLeft))){
         oneEdge[numPoints][0] = myCol;
         oneEdge[numPoints][1] = myRow + ((fabsf(level - tLeft)) / (fabsf(tLeft - bLeft)));
         numPoints++;
      //   printf("Point (%.1f, %.1f)\n", oneEdge[numPoints-1][0], oneEdge[numPoints-1][1]);
       }
       if((bLeft > 0 && bRight > 0) && ((level > bLeft && level < bRight) || (level < bLeft && level > bRight))){
         oneEdge[numPoints][0] = myCol + ((fabsf(level - bLeft)) / (fabsf(bLeft - bRight)));
         oneEdge[numPoints][1] = myRow;
         numPoints++;
        // printf("Point (%.1f, %.1f)\n", oneEdge[numPoints-1][0], oneEdge[numPoints-1][1]);
       }
       if((tRight > 0 && bRight > 0) && ((level < tRight && level > bRight) || (level > tRight && level < bRight))){
         oneEdge[numPoints][0] = myCol+1;
         oneEdge[numPoints][1] = myRow + ((fabsf(level - bRight)) / (fabsf(bRight - tRight)));
         numPoints++;
        // printf("Point (%.1f, %.1f)\n", oneEdge[numPoints-1][0], oneEdge[numPoints-1][1]);
       }


       if(numPoints == 2){

         //NOTE :'( sad day
         myLoc = atomicAdd(locCount, 2);
         atomicAdd(edgeCount, 1);
         edgeBuf[myLoc][0] = oneEdge[0][0];
         edgeBuf[myLoc][1] = oneEdge[0][1];
         edgeBuf[myLoc+1][0] = oneEdge[1][0];
         edgeBuf[myLoc+1][1] = oneEdge[1][1];

        // printf("Point1: (%.1f, %.1f) -- Point2: (%.1f, %.1f)\n", oneEdge[0][0], oneEdge[0][1], oneEdge[1][0], oneEdge[1][1]);
       }
       else if(numPoints == 4){

         //NOTE ambiguous case
        // printf("4\n");
       }

        //     if(tLeft > level){
        //       ratio = ((level - bLeft) / (tLeft - bLeft));
        //       point1[0] = (float)myCol;
        //       point1[1] = (float)myRow - ratio + 1;
        //       ratio = ((level - tRight) / (tLeft - tRight));
        //       point2[0] = (float)myCol + ratio;
        //       point2[1] = (float)myRow + 1;
        //     }
        //     else if(tRight > level){
        //       ratio = ((level - bRight) / (tRight - bRight));
        //       point1[0] = (float)myCol + 1;
        //       point1[1] = (float)myRow - ratio + 1;
        //       ratio = ((level - tLeft) / (tRight - tLeft));
        //       point2[0] = (float)myCol - ratio + 1;
        //       point2[1] = (float)myRow + 1;
        //     }
        //     else if (bLeft > level) {
        //       ratio = ((level - tLeft) / (bLeft - tLeft));
        //       point1[0] = (float)myCol;
        //       point1[1] = (float)myRow + ratio;
        //       ratio = ((level - bRight) / (bLeft - bRight));
        //       point2[0] = (float)myCol + ratio;
        //       point2[1] = (float)myRow;
        //     }
        //     else if (bRight > level){
        //       ratio = ((level - tRight) / (bRight - tRight));
        //       point1[0] = (float)myCol + 1;
        //       point1[1] = (float)myRow + ratio;
        //       ratio = ((level - bLeft) / (bRight - bLeft));
        //       point2[0] = (float)myCol - ratio + 1;
        //       point2[1] = (float)myRow;
        //     }
        //     //printf("Ratio: %.1f\n", ratio);
        //   //  printf("Point1: (%.1f, %.1f) -- Point2: (%.1f, %.1f)\n", point1[0], point1[1], point2[0], point1[1]);
        //     //printf("Point2: (%.1f, %.1f)\n", point2[0], point1[2]);
        //     myLoc = atomicAdd(locCount, 2);
        //     atomicAdd(edgeCount, 1);
        //
        //     edgeBuf[myLoc][0] = point1[0];
        //     edgeBuf[myLoc][1] = point1[1];
        //     edgeBuf[myLoc+1][0] = point2[0];
        //     edgeBuf[myLoc+1][1] = point2[1];
        //   }
        //   else if(numBelow == 1){
        //     if(tLeft < level){
        //       ratio = ((level - tLeft) / (tRight - tLeft));
        //       point1[0] = (float)myCol + ratio;
        //       point1[1] = (float)myRow + 1;
        //       ratio = ((level - tLeft) / (bLeft - tLeft));
        //       point2[0] = (float)myCol;
        //       point2[1] = (float)myRow - ratio + 1;
        //     }
        //     else if(tRight < level){
        //       ratio = ((level - tRight) / (tRight - tLeft));
        //       point1[0] = (float)myCol - ratio + 1;
        //       point1[1] = (float)myRow + 1;
        //       ratio = ((level - tRight) / (bRight - tRight));
        //       point2[0] = (float)myCol + 1;
        //       point2[1] = (float)myRow - ratio + 1;
        //     }
        //     else if(bLeft < level){
        //       ratio = ((level - bLeft) / (tLeft - bLeft));
        //       point1[0] = (float)myCol;
        //       point1[1] = (float)myRow + ratio;
        //       ratio = ((level - bLeft) / (bRight - bLeft));
        //       point2[0] = (float)myCol + ratio;
        //       point2[1] = (float)myRow;
        //     }
        //     else if(bRight < level){
        //       ratio = ((level - bRight) / (bLeft - bRight));
        //       point1[0] = (float)myCol - ratio + 1;
        //       point1[1] = (float)myRow;
        //       ratio = ((level - bRight) / (tRight - bRight));
        //       point2[0] = (float)myCol + 1;
        //       point2[1] = (float)myRow + ratio;
        //     }
        //
        //     //printf("Point1: (%.1f, %.1f) -- Point2: (%.1f, %.1f)\n", point1[0], point1[1], point2[0], point1[1]);
        //     //if(point1[1] == 60.0){printf("FLAG\n");}
        //
        //     myLoc = atomicAdd(locCount, 2);
        //     atomicAdd(edgeCount, 1);
        //
        //     edgeBuf[myLoc][0] = point1[0];
        //     edgeBuf[myLoc][1] = point1[1];
        //     edgeBuf[myLoc+1][0] = point2[0];
        //     edgeBuf[myLoc+1][1] = point2[1];
        //   }
        //
         //}
        // else
        // {
        //   // myLoc = atomicAdd(locCount, 4);
        //   // atomicAdd(edgeCount, 2);
        //   // edgeBuf[myLoc][0] = (float)myCol;
        //   // edgeBuf[myLoc][1] = (float)myRow;
        //   // edgeBuf[myLoc+1][0] = (float)myCol+1;
        //   // edgeBuf[myLoc+1][1] = (float)myRow+1;
        //   //
        //   // edgeBuf[myLoc+2][0] = (float)myCol;
        //   // edgeBuf[myLoc+2][1] = (float)myRow+1;
        //   // edgeBuf[myLoc+3][0] = (float)myCol+1;
        //   // edgeBuf[myLoc+3][1] = (float)myRow;
        // }


        /*
        TL myCol, myRow+1
        TR myCol+1, myRow+1
        BL myCol, myRow
        BR myCol+1, myRow
        */

        //printf("TL = (%i, %i) TR = (%i, %i) BL = (%i, %i) BR = (%i, %i)\n",
        // myCol, myRow+1, myCol+1, myRow+1, myCol, myRow, myCol+1, myRow);

        //printf("My x: %i, My Y: %i\n", myRow, myCol);
        //*edgeCount = 2;

        //NOTE MARK VALUES AS EITHER INSIDE OR OUTSIDE OF THE CONTOUR

      //}
      //printf("Edge added: (%i, %i) -> (%i, %i)\n", myCol, myRow, myCol+1, myRow+1);
  }
}

int launchComputeKernel(float* vertexes, int numRows, int numCols,
   float level, int expectedEdges, vec2* buf)
{

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
  cudaMalloc(&d_locCounter, sizeof(int));
  cudaMalloc(&d_buffer, expectedEdges * 2 * sizeof(vec2)); //allocate device memory for vertexes and counters

  cudaMemcpy(d_vertexes, vertexes, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgeCount, &temp, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_locCounter, &temp, sizeof(int), cudaMemcpyHostToDevice); //initialize the device data
  //cudaMemcpy(d_buffer, buf, expectedEdges * 2 * sizeof(vec2), cudaMemcpyHostToDevice);

  int warpSize = p.warpSize;
  int maxWarpsinBlock = p.maxThreadsPerBlock / warpSize; //determine #warps in one block

  int gridX = (int)(ceil((numCols)/warpSize)) + 1;
  int gridY = (int)ceil((numRows)/maxWarpsinBlock) + 1;

  dim3 block(warpSize, maxWarpsinBlock);
  dim3 gridDim(gridX, gridY); //launching dimensions

  computeKernel<<<gridDim, block>>>(d_vertexes, numRows, numCols, level, d_locCounter, d_edgeCount, d_buffer);
  cudaThreadSynchronize();

  cudaMemcpy(&h_edgeCount, d_edgeCount, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(buf, d_buffer, (h_edgeCount * 2) * sizeof(vec2), cudaMemcpyDeviceToHost); // copy back the endpoints and edge count

  cudaFree(d_vertexes);
  cudaFree(d_edgeCount);
  cudaFree(d_locCounter);
  cudaFree(d_buffer); // free device memory

  //std::cout<<h_edgeCount<<" \n";

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

  int gridX = (int)(ceil((numCols)/warpSize)) + 1;
  int gridY = (int)ceil((numRows)/maxWarpsinBlock) + 1;

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
