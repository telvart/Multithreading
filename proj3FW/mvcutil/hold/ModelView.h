// ModelView.h - an Abstract Base Class for a combined Model and View for OpenGL
//
// This definition defines class ModelView as an Abstract Base Class by
// factoring out a minimal set of methods that can be used by a Controller to
// manage a collection of abstract models. The public methods here are those
// required by the Controller; the protected methods implement common operations
// likely required by all concrete subclasses, hence they can be implemented
// once in this abstract base class. Specifically:
// 1. Methods "getMCBoundingBox" and "render" are pure virtual.
// 2. The various polymorphic versions of "handleCommand" are given
//    default (empty) implementations. Subclasses need only override if
//    they actually want to handle these types of events.
// 3. PRIMARILY FOR 2D APPLICATIONS:
//    The protected class method "compute2DScaleTrans" is designed to be used
//    in 2D applications. ModelView subclasses can use this method to fetch the
//    scale and translation terms in x and y that implement a basic
//    window-viewport transformation. The implementation uses the current
//    mcRegionOfInterest to compute the scale/translation terms that map points
//    inside that window into the -1 <= [x,y] <= +1 logical viewport range, preserving
//    aspect ratios, if enabled. Subclasses will generally simply pass these scale
//    and translation terms to their vertex shaders.
// 4. FOR 3D APPLICATIONS:
//    See the protected method "getMatrices".

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

	// General methods common to 2D and 3D applications
	// TIP: If you don't wish a particular ModelView instance to be included in
	//      bounding box calculations, return the box with xmin > xmax.
	virtual void getMCBoundingBox(double* xyzLimits) const = 0; // return 3D box
	virtual bool handleCommand(unsigned char anASCIIChar, double ldsX, double ldsY)
		{ return true; }
	virtual bool handleCommand(int aFunctionKey,
		double ldsX, double ldsY, int mods) { return true; }
	virtual bool handleCommand(Controller::SpecialKey aSpecialKey,
		double ldsX, double ldsY, int mods) { return true; }
	virtual void render() = 0;

	// Viewing methods common to 2D and 3D applications:
	static void setAspectRatioPreservationEnabled(bool b)
		{ aspectRatioPreservationEnabled = b; }
	static void setMCRegionOfInterest(double xyz[]);

	// Additional viewing methods strictly for 3D:
	static void setECZminZmax(double zMinIn, double zMaxIn);
	static void setEyeCenterUp(cryph::AffPoint E, cryph::AffPoint C, cryph::AffVector up);
	static void setObliqueProjectionDirection(const cryph::AffVector& dir);
	static void setProjection(ProjectionType pType);
	static void setProjectionPlaneZ(double ecZppIn);

	// Viewing methods for dynamic view manipulation for either 2D or 3D. These
	// methods are called by the Controller when appropriate events (e.g., mouse
	// click and drag, mouse scroll wheeel, etc.) are detected.
	static void addToGlobalPan(double dxInLDS, double dyInLDS, double dzInLDS);
	static void addToGlobalRotationDegrees(double rx, double ry, double rz);
	static void resetGlobalDynamic(); // rotation and pan
	static void resetGlobalZoom();
	static void scaleGlobalZoom(double multiplier); // dynamic_zoomScale *= multiplier

protected:

	// Method "compute2DScaleTrans" maps MC to LDS and is intended
	// strictly for 2D scenes:
	static void compute2DScaleTrans(float* scaleTransF);
	// Method "getMatrices" is intended for 3D scenes:
	static void getMatrices(cryph::Matrix4x4& mc_ec_fullOut,
	                        cryph::Matrix4x4& ec_ldsOut);

	// The following two methods are used internally when implementing
	// compute2DScaleTrans. (They are also used by the ModelView3D subclass.)
	static void linearMap(double fromMin, double fromMax,
		double toMin, double toMax, double& scale, double& trans);
	static void matchAspectRatio(double& xmin, double& xmax,
		double& ymin, double& ymax, double vAR);

	// Class variables to store viewing information used in BOTH 2D and 3D:
	static bool aspectRatioPreservationEnabled;
	static double mcRegionOfInterest[6];
	static double dynamic_zoomScale;

	// Class variables to store viewing information strictly for 2D:
	static double lds2DPanX, lds2DPanY;

	// Class variables to store viewing information strictly for 3D:
	static cryph::AffPoint eye, center;
	static cryph::AffVector up;
	static double distEyeCenter;
	static ProjectionType projType;
	static cryph::AffVector obliqueProjectionDir;
	// The following "last_xxx" are set in getMatrices and used in addToGlobalPan.
	// (See ModelView_Additions.c++)
	static double last_ecXmin, last_ecXmax, last_ecYmin, last_ecYmax;
	static double ecZmin, ecZmax, ecZpp;
	static cryph::Matrix4x4 dynamic_view; // dynamic 3D rotation/pan
};

#endif
