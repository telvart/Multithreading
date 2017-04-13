// ModelView.c++ - an Abstract Base Class for a combined Model and View for OpenGL

#include <iostream>

#include "ModelView.h"
#include "Controller.h"

// Class variables to store viewing information used in BOTH 2D and 3D:
bool ModelView::aspectRatioPreservationEnabled = true;
double ModelView::mcRegionOfInterest[6] = { -1.0, 1.0, -1.0, 1.0, -1.0, 1.0 };
double ModelView::dynamic_zoomScale = 1.0; // dynamic zoom

// Class variables to store viewing information strictly for 2D:
double ModelView::lds2DPanX = 0.0, ModelView::lds2DPanY = 0.0;

// Class variables to store viewing information strictly for 3D:
cryph::AffPoint ModelView::eye(0, 0, 2);
cryph::AffPoint ModelView::center(0, 0, 0);
cryph::AffVector ModelView::up(0, 1, 0);
double ModelView::distEyeCenter = 2.0;
ProjectionType ModelView::projType = PERSPECTIVE;
cryph::AffVector ModelView::obliqueProjectionDir(0.25, 0.5, 1.0);
double ModelView::last_ecXmin = -1.0;
double ModelView::last_ecXmax =  1.0;
double ModelView::last_ecYmin = -1.0;
double ModelView::last_ecYmax =  1.0;
double ModelView::ecZmin = -2.0;
double ModelView::ecZmax = -0.01; // for perspective, must be strictly < 0
double ModelView::ecZpp = -1.0; // for perspective, must be strictly < 0
cryph::Matrix4x4 ModelView::dynamic_view; // dynamic 3D rotation/pan

ModelView::ModelView()
{
}

ModelView::~ModelView()
{
}

/* The following two methods will need to be implemented in the
 * "ModelView_Additions.c++" you will see in various sibling directories
 * (e.g., MandM, project2, etc.)

void ModelView::addToGlobalPan(double dxInLDS, double dyInLDS, double dzInLDS)
void ModelView::addToGlobalRotationDegrees(double rx, double ry, double rz)

*/

// compute2DScaleTrans determines the current model coordinate region of
// interest and then uses linearMap to determine how to map coordinates
// in the region of interest to their proper location in Logical Device
// Space. (Returns float[] because glUniform currently favors float[].)
void ModelView::compute2DScaleTrans(float* scaleTransF) // CLASS METHOD
{
	double xmin = mcRegionOfInterest[0] - lds2DPanX;
	double xmax = mcRegionOfInterest[1] - lds2DPanX;
	double ymin = mcRegionOfInterest[2] - lds2DPanY;
	double ymax = mcRegionOfInterest[3] - lds2DPanY;

	if (aspectRatioPreservationEnabled)
	{
		// preserve aspect ratio. Make "region of interest" wider or taller to
		// match the Controller's viewport aspect ratio.
		double vAR = Controller::getCurrentController()->getViewportAspectRatio();
		matchAspectRatio(xmin, xmax, ymin, ymax, vAR);
	}

	double scaleTrans[4];
	linearMap(xmin, xmax, -dynamic_zoomScale, dynamic_zoomScale,
		scaleTrans[0], scaleTrans[1]);
	linearMap(ymin, ymax, -dynamic_zoomScale, dynamic_zoomScale,
		scaleTrans[2], scaleTrans[3]);
	for (int i=0 ; i<4 ; i++)
		scaleTransF[i] = static_cast<float>(scaleTrans[i]);
}

/* The following method will need to be implemented in the
 * "ModelView_Additions.c++" you will see in various sibling directories
 * (e.g., MandM, project2, etc.)

void ModelView::getMatrices(cryph::Matrix4x4& mc_ec, cryph::Matrix4x4& ec_lds)

*/

// linearMap determines the scale and translate parameters needed in
// order to map a value, f (fromMin <= f <= fromMax) to its corresponding
// value, t (toMin <= t <= toMax). Specifically: t = scale*f + trans.
void ModelView::linearMap(double fromMin, double fromMax, double toMin, double toMax,
						  double& scale, double& trans) // CLASS METHOD
{
	scale = (toMax - toMin) / (fromMax - fromMin);
	trans = toMin - scale*fromMin;
}

void ModelView::matchAspectRatio(double& xmin, double& xmax,
	double& ymin, double& ymax, double vAR)
{
	double wHeight = ymax - ymin;
	double wWidth = xmax - xmin;
	double wAR = wHeight / wWidth;
	if (wAR > vAR)
	{
		// make window wider
		wWidth = wHeight / vAR;
		double xmid = 0.5 * (xmin + xmax);
		xmin = xmid - 0.5*wWidth;
		xmax = xmid + 0.5*wWidth;
	}
	else
	{
		// make window taller
		wHeight = wWidth * vAR;
		double ymid = 0.5 * (ymin + ymax);
		ymin = ymid - 0.5*wHeight;
		ymax = ymid + 0.5*wHeight;
	}
}

/* The following method will need to be implemented in the
 * "ModelView_Additions.c++" you will see in various sibling directories
 * (e.g., MandM, project2, etc.)

void ModelView::scaleGlobalZoom(double multiplier)

*/

void ModelView::setECZminZmax(double zMinIn, double zMaxIn)
{
	ecZmin = zMinIn;
	ecZmax = zMaxIn;
}

void ModelView::setEyeCenterUp(cryph::AffPoint E, cryph::AffPoint C, cryph::AffVector Up)
{
	eye = E;
	center = C;
	up = Up;
	distEyeCenter = E.distanceTo(C);
}

void ModelView::setMCRegionOfInterest(double xyz[6])
{
	for (int i=0 ; i<6 ; i++)
		mcRegionOfInterest[i] = xyz[i];
}

void ModelView::setProjection(ProjectionType pType)
{
	projType = pType;
}

void ModelView::setProjectionPlaneZ(double ecZppIn)
{
	ecZpp = ecZppIn;
}
