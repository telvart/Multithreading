// GL_LINES_Renderer.h

#ifndef GL_LINES_RENDERER_H
#define GL_LINES_RENDERER_H

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

//#include <GL/glext.h>
#include <string>

#include "ModelView.h"
#include "ShaderIF.h"

typedef float vec2[2];
typedef float rgba[4];

class GL_LINES_Renderer : public ModelView
{
public:
	GL_LINES_Renderer(vec2* vertices, int numVertices, double* mm);
	virtual ~GL_LINES_Renderer();

	// xyzLimits: {wcXmin, wcXmax, wcYmin, wcYmax, wcZmin, wcZmax}
	void getMCBoundingBox(double* xyzLimitsF) const;
	void render();

	static void setNumColors(int nColors);
	static void setUseFixedColor(bool b);
	static void specifyGridDimensions(int nRows, int nCols)
		{ nGridRows = nRows; nGridCols = nCols; }
	static void toggleDrawGrid() { drawGrid = !drawGrid; }
private:
	int	numLineEndPoints, serialNum;

	GLuint vao, vertexBuffer;

	double minMax[6];

	void commonConstruct(double* mm);
	void defineDataToGPU(vec2* lineEndPoints);

	static void fetchGLSLVariableLocations();
	static void read(std::istream& is, rgba colorOut);
	static void readColorRamps(const std::string& fileName);

	static ShaderIF* shaderIF;
	static GLuint shaderProgram;
	static GLint pvaLoc_mcPosition;
	static GLint ppuLoc_color;
	static GLint ppuLoc_scaleTrans;
	static int serialNumber;
	static int serialNumberThatDrawsGrid;

	static int numColors, maxNumColors;
	static rgba** colors;
	static bool useFixedColor;
	static rgba fixedColor;
	static rgba backgroundColor;
	static rgba gridColor;

	// For grid drawing
	static GLuint vaoGrid, vboGrid;
	static int nGridRows, nGridCols;
	static void defineGrid();
	static void doDrawGrid();
	static bool drawGrid;
};

#endif
