// ModelView.c++ - an Abstract Base Class for a combined Model and View for OpenGL

#include <iostream>

#include "ModelView.h"
#include "Controller.h"

// init EYE(0,0,4), CENTER(0,0,0), UP(0,1,0)
ModelView::MCtoECSpec ModelView::curEC, ModelView::origEC;

double ModelView::mcRegionOfInterest[6] = { -1.0, 1.0, -1.0, 1.0, -1.0, 1.0 };
bool ModelView::aspectRatioPreservationEnabled = true;
ProjectionType ModelView::projType = PERSPECTIVE;
cryph::AffVector ModelView::obliqueProjectionDir(0.25, 0.5, 1.0);
double ModelView::ecZmin = -2.0;
double ModelView::ecZmax = -0.01; // for perspective, must be strictly < 0
double ModelView::ecZpp = -1.0; // for perspective, must be strictly < 0

// We record eye coordinate deltas for use in defining projection
// matrices and converting LDS offsets to eye (and hence model) deltas.
double ModelView::ecDeltaX = 2.0, ModelView::ecDeltaY = 2.0, ModelView::ecDeltaZ = 2.0;

// For Stereo:
// Implementations mean:
// 1: two different mc_ec matrices; common center; common ec_lds matrix
// 2: two different mc_ec matrices; common line of sight; common ec_lds matrix
// 3: one common mc_ec matrix; two different ec_lds matrices (the "correct" approach)
int ModelView::stereoImplementation = 2;
static std::string stereoImpLabel[] = {
	"two MC eyes; common center", "two parallel lines of sight",
    "common mc_ec; two ec_lds matrices"
};
double ModelView::stereoSeparation = 0.0;
double ModelView::stereoSeparationInc = 0.0;
bool ModelView::scrollIsZoom = true;

// dynamic zoom is built into the window-viewport mapping portion of the
// ec_lds matrix by expanding/contracting the window
double ModelView::dynamic_zoomScale = 1.0;

// Dynamic view panning: A common interface from the Controller is used to
// drive either a 2D pan in compute2DScaleTrans, or a 3D one in getMatrices.
// For the former, "mcPanVector2D" is computed in model coordinates in addToGlobalPan
// below. 3D view panning can be done in one of (at least) two ways:
// (1) A 3D pan vector is computed in model coordinates and then added to
//     the eye and center.
// (2) Accumulate pan translations into the dynamic_view matrix.
cryph::AffVector ModelView::mcPanVector2D(0,0,0);
// Default for following is "rotate about center of attention"
double ModelView::fractionOfDistEyeCenterToCenterOfRotation = 1.0;
// Matrices
cryph::Matrix4x4 ModelView::dynamic_view;
cryph::Matrix4x4 ModelView::lookAtMatrix;
cryph::Matrix4x4 ModelView::mc_ec_full;
cryph::Matrix4x4 ModelView::ec_lds;

// EXPERIMENTAL:
// 1) Translating dynamic rotations to eye-center-up
// 2) Translating Pan to movements of eye-center
// TODO: How do these two options interact? Must at least one
//       be false, or can I have them both be true?
bool ModelView::translateDynamicRotationToEyeUp = false;
bool ModelView::translateDynamicPanToEyeCenter = false;
cryph::Matrix4x4 ModelView::originalVOM;
bool ModelView::haveOriginalVOM = false;
// END: EXPERIMENTAL

bool ModelView::debug = false;

ModelView::ModelView() : polygonMode(GL_FILL)
{
}

ModelView::~ModelView()
{
}

void ModelView::addToGlobalPan(double dxInLDS, double dyInLDS, double dzInLDS)
{
	// In case this is a 2D application/ModelView instance:
    float sclTrans[4];
	compute2DScaleTrans(sclTrans);
	cryph::AffVector inc(dxInLDS/sclTrans[0], dyInLDS/sclTrans[2], 0.0);
	mcPanVector2D += inc;

	// Now in case this is a 3D application:

	// map the deltas from their (-1,+1) range to distances in eye coordinates:
	double dxInEC = 0.5 * dxInLDS * ecDeltaX;
	double dyInEC = 0.5 * dyInLDS * ecDeltaY;
	double dzInEC = 0.5 * dzInLDS * ecDeltaZ;

	if (ModelView::translateDynamicPanToEyeCenter)
	{
		double m[16];
		mc_ec_full.extractColMajor(m);
		// map the eye coordinate pan vector to a model coordinate pan vector:

		cryph::AffVector mcPanVector3D =
		//                         EC x axis represented in MC:
		                  dxInEC * cryph::AffVector(m[0], m[4], m[8]) +
		//                         EC y axis represented in MC: 
		                  dyInEC * cryph::AffVector(m[1], m[5], m[9]) +
		//                         EC z axis represented in MC:
		                  dzInEC * cryph::AffVector(m[2], m[6], m[10]);
		curEC.eye -= mcPanVector3D;
		curEC.center -= mcPanVector3D;
	}
	else // conventional implementation:
	{
		// For 3D, we build and concatenate a translation matrix:
		cryph::Matrix4x4 trans = cryph::Matrix4x4::translation(
			cryph::AffVector(dxInEC, dyInEC, dzInEC));
		dynamic_view = trans * dynamic_view;
	}
}

void ModelView::addToGlobalRotationDegrees(double rx, double ry, double rz)
{
	cryph::Matrix4x4 rxM = cryph::Matrix4x4::xRotationDegrees(rx);
	cryph::Matrix4x4 ryM = cryph::Matrix4x4::yRotationDegrees(ry);
	cryph::Matrix4x4 rzM = cryph::Matrix4x4::zRotationDegrees(rz);
	dynamic_view = rxM * ryM * rzM * dynamic_view;
}

void ModelView::compute2DScaleTrans(float* scaleTransF) // USED FOR 2D SCENES
{
    double xmin = mcRegionOfInterest[0]-mcPanVector2D[0];
	double xmax = mcRegionOfInterest[1]-mcPanVector2D[0];
    double ymin = mcRegionOfInterest[2]-mcPanVector2D[1];
	double ymax = mcRegionOfInterest[3]-mcPanVector2D[1];

	if (aspectRatioPreservationEnabled)
	{
    	// preserve aspect ratio. Make "region of interest" wider or taller to
    	// match the Controller's viewport aspect ratio.
    	double vAR = Controller::getCurrentController()->getViewportAspectRatio();
		matchAspectRatio(xmin, xmax, ymin, ymax, vAR);
	}
	
    double scaleTrans[4];
	double ldsD = ModelView::dynamic_zoomScale;
    linearMap(xmin, xmax, -ldsD, ldsD, scaleTrans[0], scaleTrans[1]);
    linearMap(ymin, ymax, -ldsD, ldsD, scaleTrans[2], scaleTrans[3]);
    for (int i=0 ; i<4 ; i++)
        scaleTransF[i] = static_cast<float>(scaleTrans[i]);
}

void ModelView::getMatrices(
	cryph::Matrix4x4& mc_ec_fullOut, cryph::Matrix4x4& ec_ldsOut)
{
	double preTransDist = fractionOfDistEyeCenterToCenterOfRotation * curEC.distEyeCenter;
	cryph::Matrix4x4 preTrans = cryph::Matrix4x4::translation(
		cryph::AffVector(0.0, 0.0, preTransDist));
	cryph::Matrix4x4 postTrans = cryph::Matrix4x4::translation(
		cryph::AffVector(0.0, 0.0, -preTransDist));
	cryph::Matrix4x4 post_dynamic_pre = postTrans * dynamic_view * preTrans;

	lookAtMatrix = cryph::Matrix4x4::lookAt(curEC.eye, curEC.center, curEC.up);
	if (debug)
		std::cout << "\nlookAtMatrix = " << lookAtMatrix << '\n';
	if (ModelView::translateDynamicRotationToEyeUp)
	{
		if (!ModelView::haveOriginalVOM)
		{
			originalVOM = cryph::Matrix4x4::lookAt(
						curEC.eye, curEC.center, curEC.up);
			haveOriginalVOM = true;
		}
		cryph::Matrix4x4 vom = post_dynamic_pre * originalVOM;
		double m[16];
		vom.extractColMajor(m);
		cryph::AffVector wHat(m[2],m[6],m[10]);
		curEC.eye = curEC.center + curEC.distEyeCenter*wHat;
		curEC.up = cryph::AffVector(m[1],m[5],m[9]);
		mc_ec_full = lookAtMatrix;
	}
	else
		mc_ec_full = post_dynamic_pre * lookAtMatrix;

	setProjectionTransformation();

	// Return the two final matrices
	mc_ec_fullOut = mc_ec_full;
	ec_ldsOut = ec_lds;
}

bool ModelView::handleCommand(unsigned char key, double ldsX, double ldsY)
{
	return true;
}

static GLenum pm[3] = { GL_POINT, GL_LINE, GL_FILL };

bool ModelView::handleCommand(unsigned char key, int num, double ldsX, double ldsY)
{
	if ((key == 'p') && (num >= 0) && (num < 3))
	{
		polygonMode = pm[num];
		glPolygonMode(GL_FRONT_AND_BACK, polygonMode);
		return false;
	}
	return true;
}

bool ModelView::handleCommand(int whichFunctionKey,
	double ldsX, double ldsY, int mods)
{
	bool retValue = false;
	double d, dx, dy, dz, radius, deppInMCEC;
	double fovX, fovY;
	float r, g, b;
	switch (whichFunctionKey)
	{
		case 1: case 2: case 3:
			setStereoImplementation(whichFunctionKey);
			std::cout << "Now using stereo implementation "
			          << whichFunctionKey << " ("
			          << stereoImpLabel[whichFunctionKey-1] << ")\n";
			break;
		case 4:
			dx = mcRegionOfInterest[1] - mcRegionOfInterest[0];
			dy = mcRegionOfInterest[3] - mcRegionOfInterest[2];
			dz = mcRegionOfInterest[5] - mcRegionOfInterest[4];
			radius = sqrt(dx*dx + dy*dy + dz*dz);
			std::cout << "\n\n------------------------------\n";
			std::cout << mcRegionOfInterest[0] << " <= mcROIx <= " << mcRegionOfInterest[1] << "; dx = " << dx << '\n';
			std::cout << mcRegionOfInterest[2] << " <= mcROIy <= " << mcRegionOfInterest[3] << "; dy = " << dy << '\n';
			std::cout << mcRegionOfInterest[4] << " <= mcROIz <= " << mcRegionOfInterest[5] << "; dz = " << dz << '\n';
			std::cout << "Radius of circumscribing sphere = " << radius << "\n\n";
			deppInMCEC = -ecZpp;
			fovX = 2.0 * atan2(ecDeltaX/2.0, deppInMCEC) * 180.0/M_PI;
			fovY = 2.0 * atan2(ecDeltaY/2.0, deppInMCEC) * 180.0/M_PI;
			std::cout << "   Eye to projection plane distance = " << deppInMCEC << '\n';
			std::cout << "Eye to center of attention distance = " << ModelView::getCurDistEyeCenter() << '\n';
			std::cout << "fovX = " << fovX << "; fovY = " << fovY << '\n';
			std::cout << ecZmin << " <= ecZ <= " << ecZmax << '\n';
			std::cout << "Using stereo implementation: " << stereoImplementation
			          << " (" << stereoImpLabel[stereoImplementation-1] << ")\n\n";
			d = 0.05 * deppInMCEC;
			std::cout << "Stereo separation = " << ModelView::getStereoSeparation()
			          << "; enter new stereo separation (" << d << "?): ";
			std::cin >> d;
			setStereoSeparation(d);
			std::cout << "ecZpp = " << ecZpp << "; enter new ecZpp: ";
			std::cin >> d;
			setProjectionPlaneZ(d);
			std::cout << "------------------------------\n";
			break;
		case 6:
			std::cout << "Enter background color RGB: ";
			std::cin >> r >> g >> b;
			glClearColor(r, g, b, 1.0);
	}
	return retValue;
}

void ModelView::linearMap(double fromMin, double fromMax, double toMin, double toMax,
	double& scale, double& trans)
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

GLint ModelView::ppUniformLocation(GLuint glslProgram, const std::string& name)
{
	GLint loc = glGetUniformLocation(glslProgram, name.c_str());
	if (loc < 0)
		std::cerr << "Could not locate per-primitive uniform: '" << name << "'\n";
	return loc;
}

void ModelView::printKeyboardKeyList(bool firstCall) const
{
	if (!firstCall)
		return;

	std::cout << "ModelView:\n";
	std::cout << "\tp@0, p@1, p@2: GL_POINT, GL_LINE, GL_FILL\n";
	std::cout << "\n\tFunction Keys:\n";
	std::cout << "\tF1: Set stereo implementation 1 (" << stereoImpLabel[0] << ")\n";
	std::cout << "\tF2: Set stereo implementation 2 (" << stereoImpLabel[1] << ")\n";
	std::cout << "\tF3: Set stereo implementation 3 (" << stereoImpLabel[2] << ")\n";
	std::cout << "\tF4: Set stereo separation and zpp in MC/EC units\n";
	std::cout << "\tF5: Toggle scroll wheel (zoom vs. stereo separation)\n";
	std::cout << "\tF6: Set background color\n";
}

GLint ModelView::pvAttribLocation(GLuint glslProgram, const std::string& name)
{
	GLint loc = glGetAttribLocation(glslProgram, name.c_str());
	if (loc < 0)
		std::cerr << "Could not locate per-vertex attribute: '" << name << "'\n";
	return loc;
}

void ModelView::resetGlobalDynamic()
{
	mcPanVector2D = cryph::AffVector(0,0,0);
	dynamic_view = cryph::Matrix4x4::IdentityMatrix;
	curEC = origEC;
}

void ModelView::resetGlobalZoom()
{
	dynamic_zoomScale = 1.0;
}	

void ModelView::scaleGlobalZoom(double multiplier)
{
	if (multiplier > 0.0)
		dynamic_zoomScale *= multiplier;
}

void ModelView::scrollHandler(bool up)
{
	if (up)
		stereoSeparation += stereoSeparationInc;
	else
	{
		double t = stereoSeparation - stereoSeparationInc;
		if (t > 0.0)
			stereoSeparation = t;
	}
	std::cout << "Stereo separation: " << stereoSeparation << '\n';
}

void ModelView::set_ecDeltas()
{
	// Get MC deltas from region of interest:
	double dx = mcRegionOfInterest[1] - mcRegionOfInterest[0];
	double dy = mcRegionOfInterest[3] - mcRegionOfInterest[2];
	double dz = mcRegionOfInterest[5] - mcRegionOfInterest[4];

	// Convert MC deltas to EC deltas (5 steps):
	// 1. Form representative MC vectors spanning the region of interest
	cryph::AffVector vecs[7] = {
		cryph::AffVector(dx, 0, 0), cryph::AffVector(0, dy, 0),
		cryph::AffVector(0, 0, dz), cryph::AffVector(dx, dy, 0),
		cryph::AffVector(dx, 0, dz), cryph::AffVector(0, dy, dz),
		cryph::AffVector(dx, dy, dz)
	};

	// 2. Get the three unit EC system vectors as measured in MC
	double m[16];
	lookAtMatrix.extractRowMajor(m);
	cryph::AffVector uvw[3] = {
		cryph::AffVector(m[0], m[1], m[2]),
		cryph::AffVector(m[4], m[5], m[6]),
		cryph::AffVector(m[8], m[9], m[10])
	};

	// 3. In each EC direction, the EC delta will be the maximum absolute
	//    value of the length of the MC vector in the EC direction. (The
	//    x and y EC directions are really the main ones of interest, We
	//    compute all 3, even though the z delta will later be adjusted
	//    as noted below.)
	double ecd[3];
	for (int i=0 ; i<3 ; i++)
	{
		ecd[i] = 0.0;
		for (int j=0 ; j<7 ; j++)
		{
			double d = fabs(uvw[i].dot(vecs[j]));
			if (d > ecd[i])
				ecd[i] = d;
		}
	}

	// 4. Use current interactive global zoom while setting basic EC deltas:
	ecDeltaX = ecd[0] / dynamic_zoomScale;
	ecDeltaY = ecd[1] / dynamic_zoomScale;
	// ecDeltaZ will be reset in setProjectionTransformation based on the
	// explicit ecZmin and ecZmax as established via ModelView::setECZminZmax.
	// Nevertheless, for now we set:
	ecDeltaZ = ecd[2];

	// 5. Adjust the x and y ecDeltas, if desired, for aspect ratio preservation:
	if (aspectRatioPreservationEnabled)
	{
		double wAR = ecDeltaY / ecDeltaX; // height/width
		double vAR = Controller::getCurrentController()->getViewportAspectRatio();
		if (vAR > wAR)
			ecDeltaY = vAR * ecDeltaX;
		else
			ecDeltaX = ecDeltaY / vAR;
	}
}

void ModelView::setECZminZmax(double zMinIn, double zMaxIn)
{
	ecZmin = zMinIn;
	ecZmax = zMaxIn;
}

void ModelView::setEyeCenterUp(cryph::AffPoint E, cryph::AffPoint C, cryph::AffVector up)
{
	cryph::AffVector v, w;
	if (cryph::Matrix4x4::getECvw(E, C, up, v, w))
	{
		origEC.eye = E;
		origEC.center = C;
		origEC.up = up;
		origEC.distEyeCenter = origEC.eye.distanceTo(origEC.center);
		resetGlobalDynamic();
		resetGlobalZoom();
	}
}

// In following: 0: about eye; 1: about center of attention
void ModelView::setFractionalDistEyeToCenterOfRotation(double f)
{
	fractionOfDistEyeCenterToCenterOfRotation = f;
}

void ModelView::setMCRegionOfInterest(double xyz[])
{
	for (int i=0 ; i<6 ; i++)
		mcRegionOfInterest[i] = xyz[i];
}

void ModelView::setObliqueProjectionDirection(const cryph::AffVector& dir)
{
	ModelView::obliqueProjectionDir = dir;
}

void ModelView::setProjection(ProjectionType pType)
{
	projType = pType;
}

void ModelView::setProjectionPlaneZ(double ecZppIn)
{
	ecZpp = ecZppIn;
}

// Following is very roughly the 3D counterpart of the 2D "compute2DScaleTrans"
// IT IMPLICITLY ASSUMES THAT THE LINE OF SIGHT PASSES THROUGH THE CENTER
// OF THE SCENE.
void ModelView::setProjectionTransformation()
{
	// Convert MC deltas (from mcRegionOfInterest) to EC deltas, adjust them
	// to match the aspect ratio of the viewport, and record them in ecDelta[XYZ].
	// WARNING: The implementation in set_ecDeltas is ridiculously (and
	//          probably unjustifiably) general and complex. Probably should
	//          just do like the version in SampleProgramSet3/mvcutil.
	set_ecDeltas();

	// Create the parameters for the desired type of projection

	double halfDx = 0.5 * ecDeltaX;
	double halfDy = 0.5 * ecDeltaY;
	if (projType == PERSPECTIVE)
	{
		if (debug)
			std::cout << "PERSPECTIVE: ecZpp = " << ecZpp << ", ";
		double h = 0.0;
		ec_lds = cryph::Matrix4x4::perspective(ecZpp, -halfDx, halfDx, -halfDy, halfDy, ecZmin, ecZmax, h);
	}
	else if (projType == ORTHOGONAL)
	{
		if (debug)
			std::cout << "ORTHOGONAL: ";
		ec_lds = cryph::Matrix4x4::orthogonal(-halfDx, halfDx, -halfDy, halfDy, ecZmin, ecZmax);
	}
	else // (projType == OBLIQUE)
	{
		if (debug)
			std::cout << "OBLIQUE: ecZpp = " << ecZpp << ", ";
		ec_lds = cryph::Matrix4x4::oblique(ecZpp, -halfDx, halfDx, -halfDy, halfDy, ecZmin, ecZmax,
			ModelView::obliqueProjectionDir);
	}
	if (debug)
	{
		std::cout << "xyz min/max = " << (-halfDx) << ", " << halfDx << ", " << (-halfDy) << ", " << halfDy << ", "
		          << ecZmin << ", " << ecZmax;
		if (projType == OBLIQUE)
			std::cout << ", oblique projection dir = " << ModelView::obliqueProjectionDir;
		std::cout << '\n';
		std::cout << "ec_lds = " << ec_lds << "\n";
	}
	ecDeltaZ = ecZmax - ecZmin;
}
