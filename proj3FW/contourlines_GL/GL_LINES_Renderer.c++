// GL_LINES_Renderer.c++

#include <iostream>
#include <fstream>

#include "GL_LINES_Renderer.h"

ShaderIF* GL_LINES_Renderer::shaderIF = nullptr;
GLuint GL_LINES_Renderer::shaderProgram = 0;

GLint GL_LINES_Renderer::pvaLoc_mcPosition = -1;

GLint GL_LINES_Renderer::ppuLoc_color = -1;
GLint GL_LINES_Renderer::ppuLoc_scaleTrans = -1;

int GL_LINES_Renderer::serialNumber = 0;
int GL_LINES_Renderer::serialNumberThatDrawsGrid = -1;

int GL_LINES_Renderer::numColors = 0;
int GL_LINES_Renderer::maxNumColors = 0;
rgba** GL_LINES_Renderer::colors = nullptr;
bool GL_LINES_Renderer::useFixedColor = false;
rgba GL_LINES_Renderer::fixedColor = { 0.0, 0.0, 0.0, 1.0 };
rgba GL_LINES_Renderer::backgroundColor = { 1.0, 1.0, 1.0, 1.0 };
rgba GL_LINES_Renderer::gridColor = { 0.5, 0.5, 0.5, 1.0 };
bool GL_LINES_Renderer::drawGrid = false;
int GL_LINES_Renderer::nGridRows = 0;
int GL_LINES_Renderer::nGridCols = 0;
GLuint GL_LINES_Renderer::vaoGrid = 0;
GLuint GL_LINES_Renderer::vboGrid = 0;

GL_LINES_Renderer::GL_LINES_Renderer(vec2* vertices, int numVertices, double* mm) :
	numLineEndPoints(numVertices)
{
	if (numLineEndPoints == 0)
		return;
	commonConstruct(mm);

	defineDataToGPU(vertices);
}

GL_LINES_Renderer::~GL_LINES_Renderer()
{
	if (vertexBuffer > 0)
		glDeleteBuffers(1, &vertexBuffer);
	if (vao > 0)
		glDeleteVertexArrays(1, &vao);
	vertexBuffer = vao = 0;
	if (serialNum == serialNumberThatDrawsGrid)
		// someone else has to do it!
		serialNumberThatDrawsGrid = -1;
}

void GL_LINES_Renderer::commonConstruct(double* mm)
{
	for (int i=0 ; i<6 ; i++)
		minMax[i] = mm[i];

	if (shaderProgram == 0)
	{
		shaderIF = new ShaderIF("shaders/Lines.vsh", "shaders/Lines.fsh");
		shaderProgram = shaderIF->getShaderPgmID();
		fetchGLSLVariableLocations();
		readColorRamps("ramps/divergingColorRamps.txt");
	}
	serialNum = serialNumber++;
	if (serialNumberThatDrawsGrid < 0)
		serialNumberThatDrawsGrid = serialNum;
}

void GL_LINES_Renderer::defineDataToGPU(vec2* lineEndPoints)
{
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vertexBuffer);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	int nBytes = numLineEndPoints * sizeof(vec2);
	glBufferData(GL_ARRAY_BUFFER, nBytes, lineEndPoints, GL_STATIC_DRAW);
	glVertexAttribPointer(pvaLoc_mcPosition, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(pvaLoc_mcPosition);
}

void GL_LINES_Renderer::defineGrid()
{
	glGenVertexArrays(1, &vaoGrid);
	glGenBuffers(1, &vboGrid);

	int numEndPoints = 2 * (nGridRows + nGridCols);
	vec2* pts = new vec2[numEndPoints];
	int loc = 0;
	float y = 0.0;
	for (int i=0 ; i<nGridRows ; i++)
	{
		pts[loc][0] = 0.0; pts[loc][1] = y;
		pts[loc+1][0] = nGridCols - 1.0; pts[loc+1][1] = y;
		y += 1.0; loc += 2;
	}
	float x = 0.0;
	for (int i=0 ; i<nGridCols ; i++)
	{
		pts[loc][0] = x; pts[loc][1] = 0.0;
		pts[loc+1][0] = x; pts[loc+1][1] = nGridRows - 1.0;
		x += 1.0; loc += 2;
	}

	glBindVertexArray(vaoGrid);

	glBindBuffer(GL_ARRAY_BUFFER, vboGrid);
	int nBytes = numEndPoints * sizeof(vec2);
	glBufferData(GL_ARRAY_BUFFER, nBytes, pts, GL_STATIC_DRAW);
	glVertexAttribPointer(pvaLoc_mcPosition, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(pvaLoc_mcPosition);

	delete [] pts;
}

void GL_LINES_Renderer::doDrawGrid()
{
	if (vaoGrid == 0)
		defineGrid();
	glBindVertexArray(vaoGrid);
	glUniform4fv(ppuLoc_color, 1, gridColor);
	glDrawArrays(GL_LINES, 0, 2 * (nGridRows + nGridCols));
}

void GL_LINES_Renderer::fetchGLSLVariableLocations()
{
	if (shaderProgram > 0)
	{
		pvaLoc_mcPosition = shaderIF->pvaLoc("mcPosition");

		ppuLoc_color = shaderIF->ppuLoc("color");
		ppuLoc_scaleTrans = shaderIF->ppuLoc("scaleTrans");
	}
}

// xyzLimits: {wcXmin, wcXmax, wcYmin, wcYmax, wcZmin, wcZmax}
void GL_LINES_Renderer::getMCBoundingBox(double* xyzLimits) const
{
	for (int i=0 ; i<6 ; i++)
		xyzLimits[i] = minMax[i];
}

void GL_LINES_Renderer::read(std::istream& is, rgba colorOut)
{
	int oneColor;
	for (int k=0 ; k<4 ; k++)
	{
		is >> oneColor;
		colorOut[k] = static_cast<float>(oneColor)/255.0f;
	}
}

void GL_LINES_Renderer::readColorRamps(const std::string& fileName)
{
	std::ifstream inp(fileName.c_str());
	if (inp.fail())
	{
		std::cerr << "Could not read color ramp file: " << fileName << '\n';
		return;
	}
	read(inp, backgroundColor);
	read(inp, fixedColor);
	read(inp, gridColor);
	int mnc;
	inp >> mnc;
	if (mnc <= 0)
	{
		std::cerr << "Too few colors in color ramp file\n";
		return;
	}
	maxNumColors = mnc;
	colors = new rgba*[maxNumColors+1];
	colors[0] = nullptr; // we don't use this spot
	for (int i=1 ; i<=maxNumColors ; i++)
	{
		colors[i] = new rgba[i];
		for (int j=0 ; j<i ; j++)
			read(inp, colors[i][j]);
	}
}

void GL_LINES_Renderer::render()
{
	if (numLineEndPoints == 0)
		return;
	// save the current GLSL program in use
	GLint pgm;
	glGetIntegerv(GL_CURRENT_PROGRAM, &pgm);

	// draw the triangles using our vertex and fragment shaders
	glUseProgram(shaderProgram);

	// define the mapping from WC to -1..+1 Logical Device Space:
	float scaleTrans[4];
	compute2DScaleTrans(scaleTrans);
	glUniform4fv(ppuLoc_scaleTrans, 1, scaleTrans);

	if (useFixedColor || (serialNum == serialNumberThatDrawsGrid))
	{
		if (drawGrid)
			doDrawGrid();
	}
	if (useFixedColor || (colors == nullptr) || (colors[numColors] == nullptr))
		glUniform4fv(ppuLoc_color, 1, fixedColor);
	else
		glUniform4fv(ppuLoc_color, 1, colors[numColors][serialNum%numColors]);

	glBindVertexArray(vao);
	glDrawArrays(GL_LINES, 0, numLineEndPoints);

	// restore the previous program
	glUseProgram(pgm);
}

void GL_LINES_Renderer::setNumColors(int nColors)
{
	if (nColors > 0)
		numColors = nColors;
	if (numColors > maxNumColors)
		numColors = maxNumColors;
}

void GL_LINES_Renderer::setUseFixedColor(bool b)
{
	useFixedColor = b;
	glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);
}
