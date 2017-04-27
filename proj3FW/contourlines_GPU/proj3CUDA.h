void run();

int launchCountKernel(float* h_vertexes, int numRows, int numCols, float level);
int launchComputeKernel();
void queryDevice();
