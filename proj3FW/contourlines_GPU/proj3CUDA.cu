
#include <stdio.h>
#include <cuda.h>

__global__ void hello(){
  printf("Hello, CUDA!\n");
}

void run()
{
  hello<<<1,10>>>();
  cudaThreadSynchronize();
}
