// ContourGenerator.c++ - Code to read a scalar data field and produce
// contours at requested levels.

#include "ContourGenerator.h"

const char* readSource(const char* clFileName);

ContourGenerator::ContourGenerator(std::istream& inp) :
	vertexValues(nullptr)
{
	inp >> nRowsOfVertices >> nColsOfVertices;
	std::string scalarDataFieldFileName;
	inp >> scalarDataFieldFileName;
	std::ifstream scalarFieldFile(scalarDataFieldFileName.c_str());
	if (scalarFieldFile.good())
	{
		readData(scalarFieldFile);
		scalarFieldFile.close();
	}
	else
	{
		std::cerr << "Could not open " << scalarDataFieldFileName
		          << " for reading.\n";
		nRowsOfVertices = nColsOfVertices = 0;
	}
}

ContourGenerator::~ContourGenerator()
{
	if (vertexValues != nullptr)
		delete [] vertexValues;
	// Delete any GPU structures (buffers) associated with this model as well!
}

int ContourGenerator::computeContourEdgesFor(float level, vec2*& lines)
{
	// Fire a kernel to determine expected number of edges at the given "level'
	int numExpectedEdges = launchCountKernel(vertexValues, nRowsOfVertices, nColsOfVertices, level);

	//std::cout<<"Expected Edges: "<<numExpectedEdges<<"\n";
	// Create space for the line end points on the device
	int numExpectedPoints = 2 * numExpectedEdges; // each edge is: (x,y), (x,y)

	vec2* buffer = new vec2[numExpectedPoints];
//	std::cout<<buffer[0][0]<<" "<<buffer[0][1]<<"\n";
//	std::cout<<buffer[1][0]<<" "<<buffer[1][1]<<"\n";

	// Fire a kernel to compute the edge end points (determimes "numActualEdges")
	int numActualEdges = launchComputeKernel(vertexValues, nRowsOfVertices, nColsOfVertices, level, numExpectedEdges, buffer);
	int numActualPoints = 2 * numActualEdges; // each edge is: (x,y), (x,y)

	// Get the point coords back, storing them into "lines"
	lines = buffer;//new vec2[numActualPoints];
	// Use CUDA or OpenCL code to retrieve the points, placing them into "lines".
	// As a placeholder for now, we will just make an "X" over the area:
	// lines[0][0] = 0.0;
	// lines[0][1] = 0.0;
	// lines[1][0] = nColsOfVertices - 1.0;
	// lines[1][1] = nRowsOfVertices - 1.0;
	//
	// lines[2][0] = 0.0;
	// lines[2][1] = nRowsOfVertices - 1.0;
	// lines[3][0] = nColsOfVertices - 1.0;
	// lines[3][1] = 0.0;

	//run();
	//queryDevice();

	//std::cout<<vertexValues[(row * nColsOfVertices) + col];

	//launchCountKernel(vertexValues, nRowsOfVertices * nColsOfVertices);
	//std::cout<<launchCountKernel();

	// After the line end points have been returned from the device, delete the
	// device buffer to prevent a memory leak.
	// ... do it here ...

	// return number of coordinate pairs in "lines":
	return numActualPoints;
}

void ContourGenerator::readData(std::ifstream& scalarFieldFile)
{
	vertexValues = new float[nRowsOfVertices * nColsOfVertices];
	int numRead = 0, numMissing = 0;
	float val;
	float minVal = 1.0, maxVal = -1.0;
	scalarFieldFile.read(reinterpret_cast<char*>(&val),sizeof(float));
	while (!scalarFieldFile.eof())
	{
		vertexValues[numRead++] = val;
		if (val == -9999)
			numMissing++;
		else if (minVal > maxVal)
			minVal = maxVal = val;
		else
		{
			if (val < minVal)
				minVal = val;
			else if (val > maxVal)
				maxVal = val;
		}
		scalarFieldFile.read(reinterpret_cast<char*>(&val),sizeof(float));
	}
	std::cout << "read " << numRead << " values; numMissing: " << numMissing
	          << "; range of values: " << minVal << " <= val <= " << maxVal << '\n';
}
