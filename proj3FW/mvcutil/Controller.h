// Controller.h - An Abstract base Class for a Controller (in Model-View-Controller sense)

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

class ModelView;

// Bit fields for Controller-related requests and masks.
// 1. Characteristics of the Rendering Context
#define MVC_USE_ACCUM_BIT 1
#define MVC_USE_ALPHA_BIT 2
#define MVC_USE_DEPTH_BIT 4
#define MVC_USE_STENCIL_BIT 8
#define MVC_USE_STEREO_BIT 16
// 2. Modifier masks in event callbacks
#define MVC_SHIFT_BIT 1
#define MVC_CTRL_BIT 2
#define MVC_ALT_BIT 4

class Controller
{
public:
	Controller();
	virtual ~Controller();

	enum MouseButton
	{
		LEFT_BUTTON, MIDDLE_BUTTON, RIGHT_BUTTON
	};

	enum SpecialKey
	{
		LEFT_ARROW, RIGHT_ARROW, UP_ARROW, DOWN_ARROW, HOME
	};

	// 1. MANAGING THE LIST OF MODELS
	void addModel(ModelView* m, int pos=32767);
	int getNumModels() const { return models.size(); }
	ModelView* getModel(int which) const;
	// The following method returns the ModelView removed from the
	// list (e.g., so the caller can issue a "delete" on it)
	ModelView* removeModel(int which);
	void removeModel(ModelView* m);
	// if (do_delete) each model will have "delete" applied to it as
	// it is removed from the list
	void removeAllModels(bool do_delete);

	// 2. OTHER METHODS
	void getOverallMCBoundingBox(double* xyzLimits) const;
	double getViewportAspectRatio() const; // height/width
	virtual void redraw() const {} // to force a display update
	void reportVersions(std::ostream& os) const;
	virtual void run() = 0;
	virtual void setWindowTitle(const std::string& title) = 0;

	// 3. CLASS METHODS
	static bool checkForErrors(std::ostream& os, const std::string& context);
	static Controller* getCurrentController() { return curController; }

protected:
	Controller(const Controller& c) {}
	std::vector<ModelView*> models;

	// Virtual methods for event handling (any can be overridden)
	// --> Keyboard-related event handling
	virtual void handleAsciiChar(unsigned char theChar, int x, int y);
	virtual void handleFunctionKey(int whichFunctionKey, int x, int y, int mods){}
	virtual void handleSpecialKey(SpecialKey key, int x, int y, int mods){}

	// --> Mouse-related event handling
	virtual void handleMouseButton(MouseButton button, bool pressed,
		int x, int y, int mods);
	virtual void handleMouseMotion(int x, int y);
	virtual void handleScroll(bool up);

	// --> Window-related event handling
	virtual void handleReshape(int width, int height);

	// --> Miscellaneous
	virtual void endProgram() { exit(0); }
	virtual void reportWindowInterfaceVersion(std::ostream& os) const = 0;
	void screenXYToLDS(int x, int y, double& ldsX, double& ldsY);
	void setClearFlags(int rcFlags);
	virtual void swapBuffers() const = 0;

	// Data the Controller uses to track the overall MC box bounding all models.
	double overallMCBoundingBox[6];
	int glClearFlags;

	static Controller* curController;
	static int newWindowWidth, newWindowHeight;
	static std::string titleString(const std::string& str);

private:

	// mouse state
	bool mouseMotionIsRotate, mouseMotionIsTranslate;
	double scaleFraction;
	int screenBaseX, screenBaseY;

	void updateMCBoundingBox(ModelView* m);
};

#endif
