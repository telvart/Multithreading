// ContourLineController.h

#ifndef CONTOURLINECONTROLLER_H
#define CONTOURLINECONTROLLER_H

#include "GLFWController.h"
#include "GL_LINES_Renderer.h"

class ContourGenerator;

class ContourLineController : public GLFWController
{
public:
	ContourLineController(const std::string& title);
	virtual ~ContourLineController();

	void printKeyboardKeyList();
	void setContourGenerator(ContourGenerator* cg,
		float* levelsIn, int nLevels);

	static bool getOpenGLSharingEnabled() { return openGLSharingEnabled; }
	static void setOpenGLSharingEnabled(bool b) { openGLSharingEnabled = b; }

protected:
	void handleAsciiChar(unsigned char key, int x, int y);
	void handleSpecialKey(Controller::SpecialKey command, int x, int y, int mods);
private:
	ContourGenerator* contourGenerator;
	float* levels;
	int numLevels;
	GL_LINES_Renderer** originalLevels;
	bool originalLevelsDisplayed;

	static bool openGLSharingEnabled;

	// for interactive contouring:
	float currentLevel, deltaLevel;
	int nSteps;

	std::vector<ModelView*> savedModels;
	
	void computeCurrentLevel();
	GL_LINES_Renderer* generateLineSet(float level);
	void generateLineSet();
};

#endif
