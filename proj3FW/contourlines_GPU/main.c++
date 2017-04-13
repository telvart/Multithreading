// main.c++: main program for contouring data

#include <iostream>
#include <fstream>
#include <string.h>

#include "ContourLineController.h"
#include "ContourGenerator.h"
#include "GL_LINES_Renderer.h"

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << " descriptionFileName\n";
		exit(1);
	}
	std::ifstream inp(argv[1]);
	if (inp.fail())
	{
		std::cerr << "Could not open " << argv[1] << " for reading.\n";
		exit(1);
	}

	int nLevels;
	inp >> nLevels;
	float* levels = new float[nLevels];
	for (int i=0 ; i<nLevels ; i++)
		inp >> levels[i];

	ContourGenerator* cg = new ContourGenerator(inp);

	// The class that will manage the graphics rendering:
	GL_LINES_Renderer::specifyGridDimensions(cg->getNumberRows(), cg->getNumberCols());

	ContourLineController c("Contour Lines");
	c.setContourGenerator(cg, levels, nLevels);

	double bb[6];
	c.getOverallMCBoundingBox(bb);
	ModelView::setMCRegionOfInterest(bb);

	c.run();

	return 0;
}
