
#include "ContourGenerator.h"

typedef float vec2[2];

int launchCountKernel(float* h_vertexes, int numRows, int numCols, float level);
int launchComputeKernel(float* vertexes, int numRows, int numCols,
   float level, int expectedEdges, vec2* buf);
void queryDevice();
