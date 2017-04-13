// ContourGenerator.h - Code to read a scalar data field and produce
// contours at requested levels.

#ifndef CONTOURGENERATOR_H
#define CONTOURGENERATOR_H

#include <iostream>
#include <fstream>

typedef float vec2[2];

class ContourGenerator
{
public:
	ContourGenerator(std::istream& inp);
	virtual ~ContourGenerator();

    // Fires a GPU kernel to compute edges for the given level and return them
    int computeContourEdgesFor(float level, vec2*& lines);

	size_t getNumberCols() const { return nColsOfVertices; }
	size_t getNumberRows() const { return nRowsOfVertices; }

private:
	void readData(std::ifstream& scalarFieldFile);

	size_t nRowsOfVertices, nColsOfVertices;
	float* vertexValues;
};

#endif
