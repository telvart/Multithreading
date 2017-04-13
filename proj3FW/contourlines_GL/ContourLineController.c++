// ContourLineController.c++

#include <iostream>

#include "ContourLineController.h"
#include "ContourGenerator.h"
#include "GL_LINES_Renderer.h"

typedef float vec2[2];

bool ContourLineController::openGLSharingEnabled = true;

ContourLineController::ContourLineController(const std::string& title) :
		GLFWController(title),
		contourGenerator(nullptr), levels(nullptr), numLevels(0),
		originalLevels(nullptr), originalLevelsDisplayed(true),
		// for interactive contouring:
		currentLevel(0.0), deltaLevel(0.0), nSteps(40)
{
	reportVersions(std::cout);
}

ContourLineController::~ContourLineController()
{
	if (levels != nullptr)
		delete [] levels;
	if (contourGenerator != nullptr)
		delete contourGenerator;
}

void ContourLineController::computeCurrentLevel()
{
	removeAllModels(true);
	generateLineSet(currentLevel);
	redraw();
}

GL_LINES_Renderer* ContourLineController::generateLineSet(float level)
{
	GL_LINES_Renderer* retModel = nullptr;
	double mm[] = {
		0.0, static_cast<double>(contourGenerator->getNumberCols() - 1),
		0.0, static_cast<double>(contourGenerator->getNumberRows() - 1),
		-1.0, 1.0 };
	vec2* lineEndPoints = nullptr;
	int nLineEndPoints = contourGenerator->computeContourEdgesFor(level, lineEndPoints);
	if (nLineEndPoints > 0)
	{
		retModel = new GL_LINES_Renderer(lineEndPoints, nLineEndPoints, mm);
		addModel(retModel);
		delete [] lineEndPoints;
	}
	return retModel;
}

void ContourLineController::generateLineSet()
{
	if ((levels != nullptr) && (numLevels > 0) && (contourGenerator != nullptr))
	{
		for (int i=0 ; i<numLevels ; i++)
			originalLevels[i] = generateLineSet(levels[i]);
		GL_LINES_Renderer::setNumColors(numLevels);
	}
}

void ContourLineController::handleAsciiChar(unsigned char key, int x, int y)
{
	const char ESC = 27;
	if (key == ESC)
	{
		if (contourGenerator != nullptr)
		{
			delete contourGenerator;
			contourGenerator = nullptr;
		}
		GLFWController::handleAsciiChar(key, x, y);
	}
	else if (key == 'g')
		GL_LINES_Renderer::toggleDrawGrid();
	else if (key == ' ')
	{
		if (originalLevelsDisplayed)
		{
			GL_LINES_Renderer::setUseFixedColor(true);
			removeAllModels(false);
			computeCurrentLevel();
		}
		else
		{
			GL_LINES_Renderer::setUseFixedColor(false);
			for (int i=0 ; i<numLevels ; i++)
				if (originalLevels[i] != nullptr)
					addModel(originalLevels[i]);
		}
		originalLevelsDisplayed = !originalLevelsDisplayed;
	}
	else
		GLFWController::handleAsciiChar(key, x, y);
	redraw();
}

void ContourLineController::handleSpecialKey(Controller::SpecialKey key, int x, int y, int mods)
{
	if (key == Controller::LEFT_ARROW)
	{
		if (originalLevelsDisplayed)
			return;
		currentLevel -= deltaLevel;
		if (currentLevel < levels[0])
			currentLevel = levels[0];
		else
			computeCurrentLevel();
	}
	else if (key == Controller::RIGHT_ARROW)
	{
		if (originalLevelsDisplayed)
			return;
		currentLevel += deltaLevel;
		if (currentLevel > levels[numLevels-1])
			currentLevel = levels[numLevels-1];
		else
			computeCurrentLevel();
	}
	else
		GLFWController::handleSpecialKey(key, x, y, mods);
	redraw();
}

void ContourLineController::printKeyboardKeyList()
{
	std::cout << "ContourLineController:\n";
	std::cout << "                  g: Toggle drawing the grid\n";
	std::cout << "                ' ': (space bar) toggle interactive contouring\n";
	std::cout << "     Left Arrow Key: if interactive contouring, decrease the current contour level\n";
	std::cout << "    Right Arrow Key: if interactive contouring, increase the current contour level\n";
}

void ContourLineController::setContourGenerator(ContourGenerator* cg,
        float* levelsIn, int nLevels)
{
	contourGenerator = cg;
	if ((levelsIn != nullptr) && (nLevels > 0))
	{
		numLevels = nLevels;
		levels = new float[numLevels];
		originalLevels = new GL_LINES_Renderer*[numLevels];
		for (int i=0 ; i<numLevels ; i++)
		{
			levels[i] = levelsIn[i];
			originalLevels[i] = nullptr;
		}
		generateLineSet();

		// for interactive contouring:
		if (numLevels == 1)
			deltaLevel = 0;
		else
			deltaLevel = (levels[numLevels-1] - levels[0])/(nSteps-1);
		currentLevel = levels[0];
		GL_LINES_Renderer::setUseFixedColor(false);
	}
}
