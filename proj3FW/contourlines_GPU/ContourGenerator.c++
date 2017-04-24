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
	//use vertexValues to compute the data

	// Fire a kernel to determine expected number of edges at the given "level'
	int numExpectedEdges = 3;

	// Create space for the line end points on the device
	int numExpectedPoints = 2 * numExpectedEdges; // each edge is: (x,y), (x,y)

	// Fire a kernel to compute the edge end points (determimes "numActualEdges")
	int numActualEdges = 3;
	int numActualPoints = 2 * numActualEdges; // each edge is: (x,y), (x,y)

	// Get the point coords back, storing them into "lines"
	lines = new vec2[numActualPoints];
	// Use CUDA or OpenCL code to retrieve the points, placing them into "lines".
	// As a placeholder for now, we will just make an "X" over the area:
	lines[0][0] = 0.0;
	lines[0][1] = 0.0;
	lines[1][0] = nColsOfVertices - 1.0;
	lines[1][1] = nRowsOfVertices - 1.0;

	lines[2][0] = 0.0;
  lines[2][1] = nRowsOfVertices - 1.0;
	lines[3][0] = nColsOfVertices - 1.0;
  lines[3][1] = 0.0;

	lines[4][0] = (nColsOfVertices - 1) / (double)2;
	lines[4][1] = 0.0;
	lines[5][0] = (nColsOfVertices - 1) / (double)2;
	lines[5][1] = nRowsOfVertices - 1;


	for(int i=0; i<nRowsOfVertices * nColsOfVertices; i++)
	{
		std::cout<<vertexValues[i]<<" ";
	}
	std::cout<<"\n\n";

	// for(int i=0; i<nRowsOfVertices; i++){
	// 	for(int j=0; j<nColsOfVertices; j++){
	// 		std::cout<<vertexValues[i + j]<<" ";
	// 	}
	// 	std::cout<<"\n";
	// }
	// std::cout<<"\n";


	// std::cout<<"Columns: "<<nColsOfVertices
	// 				 <<"\nRows:  "<<nRowsOfVertices<<"\n";

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
