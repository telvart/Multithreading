
#include <stdio.h>
#include <cuda.h>
#include "proj3CUDA.h"

__global__ void hello(){
  printf("Hello, CUDA!\n");
}

void run()
{
  hello<<<1,1>>>();
  cudaThreadSynchronize();
}
