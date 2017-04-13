// ModelView.h - an Abstract Base Class for a combined Model and View for OpenGL

#ifndef MODELVIEW_H
#define MODELVIEW_H

#include <string>

#ifdef __APPLE_CC__
#include "GLFW/glfw3.h"
#else
#include <GL/gl.h>
#endif

#include "AffPoint.h"
#include "AffVector.h"
#include "Controller.h"
#include "Matrix4x4.h"
#include "ProjectionType.h"

class ModelView
{
public:
	ModelView();
	virtual ~ModelView();

	// TIP: If you don't wish a particular ModelView instance to be included in
	//      bounding box calculations, return the box with xmin > xmax.
	virtual void getMCBoundingBox(double* xyzLimits) const = 0;
	// Some events should be processed by exactly one ModelView instance
	// (typically on behalf of the class as a whole), others need to be
	// handled by each individual ModelView instance. Therefore, in the
	// following three "handle" methods, if the ModelView instance returns
	// "true", it indicates that the Controller should continue sending
	// this event to other instances. If the return value is "false", it
	// tells the Controller not to send the event notice to any other
	// ModelView instance.
	virtual bool handleCommand(unsigned char key, double ldsX, double ldsY);
	virtual bool handleCommand(int whichFunctionKey,
		double ldsX, double ldsY, int mods);
	virtual bool handleCommand(Controller::SpecialKey key,
		double ldsX, double ldsY, int mods) { return true; }
	virtual void render() = 0;

	// Viewing controls common to 2D and 3D:
	static void setAspectRatioPreservationEnabled(bool b)
		{ aspectRatioPreservationEnabled = b; }
	static void setMCRegionOfInterest(double xyz[]);

	// Additional viewing API strictly for 3D:
	static void setECZminZmax(double zMinIn, double zMaxIn);
	static void setEyeCenterUp(cryph::AffPoint E, cryph::AffPoint C, cryph::AffVector up);
	static void setObliqueProjectionDirection(const cryph::AffVector& dir);
	static void setProjection(ProjectionType pType);
	static void setProjectionPlaneZ(double zppIn);
	static int getStereoImplementation() { return stereoImplementation; }
	static void setStereoImplementation(int i) { stereoImplementation = i; }
	static double getStereoSeparation() { return stereoSeparation; }
	static void setStereoSeparation(double distBetweenEyes) { stereoSeparation = distBetweenEyes; }

	// Viewing interfaces for dynamic view manipulation
	static void addToGlobalPan(double dxInLDS, double dyInLDS, double dzInLDS);
	static void addToGlobalRotationDegrees(double rx, double ry, double rz);
	static void resetGlobalDynamic(); // rotation and pan
	static void resetGlobalZoom();
	static void scaleGlobalZoom(double multiplier); // dynamic_zoomScale *= multiplier

	// EXTRA public methods in my private version:
	virtual bool handleCommand(unsigned char key, int num, double ldsX, double ldsY);
	virtual void printKeyboardKeyList(bool firstCall) const;
	static void setDebug(bool b) { debug = b; }
	static void setFractionalDistEyeToCenterOfRotation(double f); //0=>about eye; 1=>about center
protected:

	// NOTE: Subclasses will generally use EITHER compute2DScaleTrans OR getMatrices:
	// Method "compute2DScaleTrans" is intended for 2D scenes:
	static void compute2DScaleTrans(float* sclTrans);
	// Method "getMatrices" is intended for 3D scenes:
	static void getMatrices(cryph::Matrix4x4& mc_ec_fullOut,
	                        cryph::Matrix4x4& ec_ldsOut);

	// following here for now to help with tuning stereo controls:
	static double getCurDistEyeCenter() { return curEC.distEyeCenter; }

	static void getECDeltas(double& dx, double& dy, double& dz)
		{ dx = ecDeltaX; dy = ecDeltaY; dz = ecDeltaZ; }

	static void linearMap(double fromMin, double fromMax,
		double toMin, double toMax, double& scale, double& trans);
	static void matchAspectRatio(double& xmin, double& xmax,
		double& ymin, double& ymax, double vAR);
	// "pp": "per-primitive"; "pv": "per-vertex"
	static GLint ppUniformLocation(GLuint glslProgram, const std::string& name);
	static GLint pvAttribLocation(GLuint glslProgram, const std::string& name);

	// Class variables used to save information passed into the 3D Viewing API:
	static double ecZmin, ecZmax, ecZpp;
	static ProjectionType projType;
	static cryph::AffVector obliqueProjectionDir;
	static double mcRegionOfInterest[6];
	static bool aspectRatioPreservationEnabled;

	// Class variables used to save information for dynamic viewing:
	static double dynamic_zoomScale;
	static cryph::Matrix4x4 dynamic_view; // mouse-based dynamic 3D rotations and pan

	// EXTRA protected methods and variables in my private version:
	GLenum polygonMode;
private: // ModelView in SampleProgramSet3 does not have a "private" section.
	// The main differences here (i.e., why these declarations are here) is due to:
	// (1) I convert MC deltas from mcRegionOfInterest to actual EC deltas; see
	//     class method set_ecDeltas() and class variables ecDeltaX, ecDeltaY,
	//     and ecDeltaZ.
	// (2) The "experimental" stuff below on translating pan to changes in eye,
	//     center, up specs.

	struct MCtoECSpec
	{
		cryph::AffPoint eye, center;
		cryph::AffVector up;
		double distEyeCenter; // Computed as: eye.distanceTo(center)

		MCtoECSpec() : eye(0,0,4), center(0,0,0), up(0,1,0),
			distEyeCenter(4)
		{}
	};
	static MCtoECSpec curEC, origEC;

	struct ECtoLDSSpec
	{
		double ecZmin, ecZmax, ecZpp;
		ProjectionType projType;
		cryph::AffVector obliqueProjectionDir;
		bool aspectRatioPreservationEnabled;
	};
	static ECtoLDSSpec ecToLDSSpec;

	static double ecDeltaX, ecDeltaY, ecDeltaZ;
	// for stereo
	static int stereoImplementation;
	static double stereoSeparation, stereoSeparationInc;
	static bool scrollIsZoom;

	static double fractionOfDistEyeCenterToCenterOfRotation;
	static cryph::AffVector mcPanVector2D;
	// matrix variables and utilities
	static cryph::Matrix4x4 lookAtMatrix;
	static cryph::Matrix4x4 mc_ec_full; // = rotation * lookAtMatrix
	static cryph::Matrix4x4 ec_lds; // wv * proj

	static void scrollHandler(bool up);
	static void set_ecDeltas();
	static void setProjectionTransformation();
	// end matrix utilities

	static bool debug;

	// EXPERIMENTAL: Translating dynamic rotations to eye-center-up
	static bool translateDynamicRotationToEyeUp;
	static bool translateDynamicPanToEyeCenter;
	static cryph::Matrix4x4 originalVOM;
	static bool haveOriginalVOM;
	// END: EXPERIMENTAL
};

#endif
