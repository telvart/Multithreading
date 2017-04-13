// BasicShape.c++ - Interface for common shapes in OpenGL

#include <math.h>

#include "BasicShape.h"

static const double BasicDistanceTol = 1.0e-4;
static const double BasicUnitTol = 1.0e-6;

bool BasicShape::alwaysGeneratePerVertexNormals = false;

BasicShape::BasicShape() :
	pointCoords(nullptr), normals(nullptr), textureCoords(nullptr), nPoints(0),
	drawArraysCallData(nullptr), nDrawArraysCalls(0),
	indexLists(nullptr), nIndexLists(0), nIndexListsConstructed(0)
{
	// indicate min/max limits not yet initialized by setting min > max:
	xyzMinMax[0] = xyzMinMax[2] = xyzMinMax[4] = 1.0;
	xyzMinMax[1] = xyzMinMax[3] = xyzMinMax[5] = -1.0;
}

BasicShape::~BasicShape()
{
	if (pointCoords != nullptr)
		delete [] pointCoords;
	if (normals != nullptr)
		delete [] normals;
	if (textureCoords != nullptr)
		delete [] textureCoords;
	if (drawArraysCallData != nullptr)
	{
		delete [] drawArraysCallData;
		nDrawArraysCalls = 0;
	}
	if (indexLists != nullptr)
	{
		for (int i=0 ; i<nIndexLists ; i++)
			if (indexLists[i].indices != nullptr)
				delete[] indexLists[i].indices;
		delete [] indexLists;
		nIndexLists = nIndexListsConstructed = 0;
	}
}

// private methods to do the actual construction
void BasicShape::addCapsFixedNormal(Caps capSpec, int nPointsInCap,
	const cryph::AffVector& axis, int nPointsAlongAxis)
{
	if (capSpec == CAP_AT_NEITHER)
		return;
	if ((capSpec == CAP_AT_BOTTOM) || (capSpec == CAP_AT_BOTH))
	{
		indexLists[nIndexListsConstructed].mode = GL_TRIANGLE_FAN;
		indexLists[nIndexListsConstructed].thisStartsFace = -1;
		indexLists[nIndexListsConstructed].indices = new int[nPointsInCap];
		indexLists[nIndexListsConstructed].nIndices = nPointsInCap;
		indexLists[nIndexListsConstructed].usePerVertexTexCoords = false;
		indexLists[nIndexListsConstructed].usePerVertexNormals = false;
		indexLists[nIndexListsConstructed].useFixedNormal = -axis;
		// bottom cap:
		for (int i=0 ; i<nPointsInCap ; i++)
			indexLists[nIndexListsConstructed].indices[i] = i*nPointsAlongAxis;
		nIndexListsConstructed++;
	}
	if ((capSpec == CAP_AT_TOP) || (capSpec == CAP_AT_BOTH))
	{
		indexLists[nIndexListsConstructed].mode = GL_TRIANGLE_FAN;
		indexLists[nIndexListsConstructed].thisStartsFace = -1;
		indexLists[nIndexListsConstructed].indices = new int[nPointsInCap];
		indexLists[nIndexListsConstructed].nIndices = nPointsInCap;
		indexLists[nIndexListsConstructed].usePerVertexTexCoords = false;
		indexLists[nIndexListsConstructed].usePerVertexNormals = false;
		indexLists[nIndexListsConstructed].useFixedNormal = axis;
		// top cap:
		for (int i=0 ; i<nPointsInCap ; i++)
			indexLists[nIndexListsConstructed].indices[i] = (i+1)*nPointsAlongAxis - 1;
		nIndexListsConstructed++;
	}
}

void BasicShape::addCapsPerVertexNormals(Caps capSpec, int nPointsAlongAxis,
	int firstAtBottom, int firstAtTop, int nPointsInOneCap)
{
	if (capSpec == CAP_AT_NEITHER)
		return;
	if (nPointsAlongAxis == 2)
		// already generated the caps
		return;
	if ((capSpec == CAP_AT_BOTTOM) || (capSpec == CAP_AT_BOTH))
		nDrawArraysCalls = 1;
	if ((capSpec == CAP_AT_TOP) || (capSpec == CAP_AT_BOTH))
		nDrawArraysCalls++;
	drawArraysCallData = new DrawArraysCallData[nDrawArraysCalls];
	addDrawArraysDataForConeCylCaps(0, firstAtBottom, firstAtTop, nPointsInOneCap);
}

void BasicShape::addDrawArraysDataForConeCylCaps(int nextIndex,
	int firstAtBottom, int firstAtTop, int nPointsInOneCap)
{
	if (firstAtBottom >= 0)
	{
		drawArraysCallData[nextIndex].mode = GL_TRIANGLE_FAN;
		drawArraysCallData[nextIndex].thisStartsFace = -1;
		drawArraysCallData[nextIndex].offset = firstAtBottom/3;
		drawArraysCallData[nextIndex].nPointsInCall = nPointsInOneCap;
		drawArraysCallData[nextIndex].usePerVertexTexCoords = false;
		drawArraysCallData[nextIndex].usePerVertexNormals = true;
		nextIndex++;
	}
	if (firstAtTop >= 0)
	{
		drawArraysCallData[nextIndex].mode = GL_TRIANGLE_FAN;
		drawArraysCallData[nextIndex].thisStartsFace = -1;
		drawArraysCallData[nextIndex].offset = firstAtTop/3;
		drawArraysCallData[nextIndex].nPointsInCall = nPointsInOneCap;
		drawArraysCallData[nextIndex].usePerVertexTexCoords = false;
		drawArraysCallData[nextIndex].usePerVertexNormals = true;
	}
}

void BasicShape::allocateIndexListsForConeCyl(int nPointsAlongAxis, Caps capSpec)
{
	nIndexLists = nIndexListsConstructed = 0;
	if (nPointsAlongAxis > 2)
		nIndexLists = nPointsAlongAxis - 1;
	if (!alwaysGeneratePerVertexNormals)
	{
		if (capSpec == CAP_AT_BOTH)
			nIndexLists += 2;
		else if (capSpec != CAP_AT_NEITHER)
			nIndexLists++;
	}
	if (nIndexLists > 0)
		indexLists = new IndexListData[nIndexLists];
}

void BasicShape::allocateIndexListsForSphere(int nPointsAlongAxis)
{
	nIndexListsConstructed = 0;
	nIndexLists = nPointsAlongAxis - 1;
	indexLists = new IndexListData[nIndexLists];
}

void BasicShape::finishBlockDataUsingFixedNormals(
	const cryph::AffVector& uvNormal, const cryph::AffVector& uwNormal,
	const cryph::AffVector& vwNormal, const cryph::AffPoint* vertices)
{
	// copy the vertex data to buffer to be sent to GPU	
	nPoints = 8;
	pointCoords = new float[nPoints * 3]; // "*3": (x, y, z)
	int offset = 0;
	for (int i=0 ; i<8 ; i++, offset+=3)
		vertices[i].aCoords(pointCoords, offset);

	// first three faces are drawn using glDrawArrays
	nDrawArraysCalls = 3;
	drawArraysCallData = new DrawArraysCallData[nDrawArraysCalls];
	for (int i=0 ; i<3 ; i++)
	{
		drawArraysCallData[i].mode = GL_TRIANGLE_STRIP;
		drawArraysCallData[i].thisStartsFace = i;
		drawArraysCallData[i].offset = 2 * i;
		drawArraysCallData[i].nPointsInCall = 4;
		drawArraysCallData[i].usePerVertexTexCoords = false;
		drawArraysCallData[i].usePerVertexNormals = false;
	}
	drawArraysCallData[0].useFixedNormal = uwNormal;
	drawArraysCallData[1].useFixedNormal = vwNormal;
	drawArraysCallData[2].useFixedNormal = -uwNormal;

	// remaining three faces defined using index lists
	nIndexLists = nIndexListsConstructed = 3;
	indexLists = new IndexListData[nIndexLists];
	int startIndex = 0;
	IndexListData& ild0 = indexLists[0];
	ild0.mode = GL_TRIANGLE_STRIP;
	ild0.thisStartsFace = 3;
	ild0.nIndices = 4;
	ild0.indices = new int[4];
	ild0.indices[0] = 6;
	ild0.indices[1] = 7;
	ild0.indices[2] = 0;
	ild0.indices[3] = 1;
	ild0.usePerVertexTexCoords = false;
	ild0.usePerVertexNormals = false;
	ild0.useFixedNormal = -vwNormal;

	// "top" and "bottom" faces:
	for (int i=0 ; i<2 ; i++)
	{
		IndexListData& ild = indexLists[1+i];
		ild.mode = GL_TRIANGLE_STRIP;
		ild.thisStartsFace = 4+i;
		ild.nIndices = 4;
		ild.indices = new int[4];
		ild.indices[0] = i;
		ild.indices[1] = i+2;
		ild.indices[2] = i+6;
		ild.indices[3] = i+4;
		ild.usePerVertexTexCoords = false;
		ild.usePerVertexNormals = false;
	}
	indexLists[1].useFixedNormal = -uvNormal;
	indexLists[2].useFixedNormal = uvNormal;
}

void BasicShape::finishBlockDataUsingPerVertexNormals(
	const cryph::AffVector& uvNormal, const cryph::AffVector& uwNormal,
	const cryph::AffVector& vwNormal, const cryph::AffPoint* vertices)
{
	std::cerr << "\n\nWARNING: 'always use per vertex normals' has been set, and a Block is\n"
	          << "being constructed. Face identification for the first four faces of the\n"
	          << "block cannot be sent to a registered 'prepareForFace' function when this\n"
	          << "block is rendered.\n\n\n";
	// copy the vertex data to buffer to be sent to GPU	
	nPoints = 24; // store each point once for each face on which it lies
	pointCoords = new float[nPoints * 3]; // "*3": (x, y, z)
	normals = new float[nPoints * 3];
	// first four faces around perimeter
	int firstVertexLoc = 0;
	int offset = 0;
	cryph::AffVector currentNormal;
	for (int face=0 ; face<4 ; face++, firstVertexLoc+=2)
	{
		switch (face)
		{
			case 0: currentNormal = uwNormal; break;
			case 1: currentNormal = vwNormal; break;
			case 2: currentNormal = -uwNormal; break;
			default: currentNormal = -vwNormal;
		}
		for (int i=0 ; i<4 ; i++, offset+=3)
		{
			int loc = (firstVertexLoc + i) % 8;
			vertices[loc].aCoords(pointCoords, offset);
			currentNormal.vComponents(normals, offset);
		}
	}
	// top/bottom faces:
	currentNormal = uvNormal;
	for (int i=1 ; i<8 ; i+=2, offset+=3)
	{
			vertices[i].aCoords(pointCoords, offset);
			currentNormal.vComponents(normals, offset);
	}
	currentNormal = -uvNormal;
	for (int i=6 ; i>=0 ; i-=2, offset+=3)
	{
			vertices[i].aCoords(pointCoords, offset);
			currentNormal.vComponents(normals, offset);
	}

	// One glDrawArrays call for 4 perimeter faces; one each for top/bottom:
	nDrawArraysCalls = 3;
	drawArraysCallData = new DrawArraysCallData[nDrawArraysCalls];
	for (int i=0 ; i<3 ; i++)
	{
		if (i == 0)
		{
			drawArraysCallData[i].mode = GL_TRIANGLE_STRIP;
			// Since the first call draws all four perimeter faces,
			// we do not allow this to start a "single" face.
			drawArraysCallData[i].thisStartsFace = -1;
			drawArraysCallData[i].offset = 0;
			drawArraysCallData[i].nPointsInCall = 16;
		}
		else
		{
			drawArraysCallData[i].mode = GL_TRIANGLE_FAN;
			drawArraysCallData[i].thisStartsFace = 3+i;
			drawArraysCallData[i].offset = 16 + (i-1)*4;
			drawArraysCallData[i].nPointsInCall = 4;
		}
		drawArraysCallData[i].usePerVertexNormals = true;
		drawArraysCallData[i].usePerVertexTexCoords = false;
	}
}

// Accessors for use during actual drawing during display callbacks
// May have data for one or more glDrawArrays calls:
int BasicShape::getDrawArraysData(int i, GLenum& mode,
		int& thisStartsFace, int& offset,
		bool& canUsePerVertexTexCoords,
		bool& canUseNormalArray, cryph::AffVector& fixedN) const
{
	if ((i >= 0) && (i < nDrawArraysCalls))
	{
		mode = drawArraysCallData[i].mode;
		thisStartsFace = drawArraysCallData[i].thisStartsFace;
		offset = drawArraysCallData[i].offset;
		canUsePerVertexTexCoords = drawArraysCallData[i].usePerVertexTexCoords;
		canUseNormalArray = drawArraysCallData[i].usePerVertexNormals;
		fixedN = drawArraysCallData[i].useFixedNormal;
		return drawArraysCallData[i].nPointsInCall;
	}
	return 0;
}

const void* BasicShape::getIndexList(int i, GLenum& mode,
	int& thisStartsFace, int& nInList, GLenum& type,
	bool& canUsePerVertexTexCoords, bool& canUsePerVertexNormals,
	cryph::AffVector& fixedNormal) const
{
	if ((indexLists != nullptr) && (i < nIndexLists))
	{
		mode = indexLists[i].mode;
		thisStartsFace = indexLists[i].thisStartsFace;
		nInList = indexLists[i].nIndices;
		type = GL_UNSIGNED_INT;
		canUsePerVertexTexCoords = indexLists[i].usePerVertexTexCoords;
		canUsePerVertexNormals = indexLists[i].usePerVertexNormals;
		fixedNormal = indexLists[i].useFixedNormal;
		return indexLists[i].indices;
	}
	// bad request or no index lists
	return nullptr;
}

void BasicShape::getMCBoundingBox(double* xyzLimits) const
{
	for (int i=0 ; i<6 ; i++)
		xyzLimits[i] = xyzMinMax[i];
}

// Factory methods for creation of shapes:
// Texture coordinates are generated only if BOTH sMax>sMin AND tMax>tMin;
// See header file for details on 's' and 't' parametric directions.

// For Block, we create the eight vertex points, but since each face has a
// unique normal, we feed it back as six primitives, one for each face.
BasicShape* BasicShape::makeBlock(
		const cryph::AffPoint& llCorner,
		const cryph::AffVector& uEdge, double uLength,
		const cryph::AffVector& vEdge, double vLength,
		const cryph::AffVector& wEdge, double wLength)
{
	// validate and pre-process data:
	cryph::AffVector u, v, w;
	double length = uEdge.normalizeToCopy(u);
	if (length < BasicDistanceTol)
		return nullptr;
	if (uLength <= 0.0)
		uLength = length;
	length = vEdge.normalizeToCopy(v);
	if (length < BasicDistanceTol)
		return nullptr;
	if (fabs(fabs(u.dot(v)) - 1.0) < BasicDistanceTol)
		// u and v are parallel
		return nullptr;
	if (vLength <= 0.0)
		vLength = length;
	length = wEdge.normalizeToCopy(w);
	cryph::AffVector uCROSSv = u.cross(v);
	if (uCROSSv.dot(w) < BasicUnitTol)
		// w is either 0 or in uv plane
		w = uCROSSv;
	if (wLength <= 0.0)
		wLength = length;
	if (wLength == 0.0)
		wLength = 1.0;
	BasicShape* bs = new BasicShape();
	bs->makeBlockData(llCorner, u, uLength, v, vLength, w, wLength);
	return bs;
}

void BasicShape::makeBlockData(
		const cryph::AffPoint& llCorner,
		const cryph::AffVector& u, double uLength,
		const cryph::AffVector& v, double vLength,
		const cryph::AffVector& w, double wLength)
{
	// Normal vectors
	cryph::AffVector uvNormal = u.cross(v);
	cryph::AffVector uwNormal = u.cross(w);
	cryph::AffVector vwNormal = v.cross(w);
	// Vertices
	cryph::AffPoint vertices[8];
	makeEightBlockVertices(llCorner, u, uLength, v, vLength,
		w, wLength, vertices);

	if (alwaysGeneratePerVertexNormals)
		finishBlockDataUsingPerVertexNormals(uvNormal, uwNormal, vwNormal,
			vertices);
	else
		finishBlockDataUsingFixedNormals(uvNormal, uwNormal, vwNormal,
			vertices);
}

BasicShape* BasicShape::makeBoundedCone(
		const cryph::AffPoint& Pbottom, const cryph::AffPoint& Ptop,
		double radiusAtBottom, double radiusAtTop,
		int nPointsAroundSide, int nPointsAlongAxis,
		Caps capSpec, double sMin, double sMax, double tMin, double tMax,
		const cryph::AffVector& sZero)
{
	if ((nPointsAroundSide < 3) || (nPointsAlongAxis < 2) || (radiusAtBottom <= 0) || (radiusAtTop <= 0))
		return nullptr;
	cryph::AffVector axis = Ptop - Pbottom;
	double height = axis.normalize();
	if (height < BasicDistanceTol)
		return nullptr;
	BasicShape* bs = new BasicShape();
	bs->makeRuledSurfaceBetweenCircles(
			Pbottom, axis, height, radiusAtBottom, radiusAtTop,
			nPointsAroundSide, nPointsAlongAxis, capSpec, sMin, sMax, tMin, tMax, sZero);
	return bs;
}
	
BasicShape* BasicShape::makeBoundedCylinder(
		const cryph::AffPoint& Pbottom, const cryph::AffPoint& Ptop,
		double radius,
		int nPointsAroundSide, int nPointsAlongAxis,
		Caps capSpec,
		double sMin, double sMax, double tMin, double tMax,
		const cryph::AffVector& sZero)
{
	if ((nPointsAroundSide < 3) || (nPointsAlongAxis < 2) || (radius <= 0))
		return nullptr;
	cryph::AffVector axis = Ptop - Pbottom;
	double height = axis.normalize();
	if (height < BasicDistanceTol)
		return nullptr;
	BasicShape* bs = new BasicShape();
	bs->makeRuledSurfaceBetweenCircles(
			Pbottom, axis, height, radius, radius,
			nPointsAroundSide, nPointsAlongAxis, capSpec, sMin, sMax, tMin, tMax, sZero);
	return bs;
}

// Worker routines

// firstAtBottom and firstAtTop are only non-negative if we are generating
// per-vertex normals for the caps AND caps were requested.
void BasicShape::makeDrawArraysDataForConeCyl(int nPointsWithoutCaps,
	int firstAtBottom, int firstAtTop)
{
	nDrawArraysCalls = 1;
	if (firstAtBottom >= 0)
		nDrawArraysCalls = 2;
	if (firstAtTop >= 0)
		nDrawArraysCalls++;
	drawArraysCallData = new DrawArraysCallData[nDrawArraysCalls];
	drawArraysCallData[0].mode = GL_TRIANGLE_STRIP;
	drawArraysCallData[0].thisStartsFace = 0;
	drawArraysCallData[0].offset = 0;
	// all points are required for this glDrawArrays call:
	drawArraysCallData[0].nPointsInCall = nPointsWithoutCaps;
	drawArraysCallData[0].usePerVertexTexCoords = (textureCoords != nullptr);
	drawArraysCallData[0].usePerVertexNormals = true;
	// See if we have caps
	if (nDrawArraysCalls > 1)
	{
		int nPointsInOneCap = (nPoints - nPointsWithoutCaps)/(nDrawArraysCalls - 1);
		addDrawArraysDataForConeCylCaps(1, firstAtBottom, firstAtTop, nPointsInOneCap);
	}
}

void BasicShape::makeEightBlockVertices(
		const cryph::AffPoint& llCorner,
		const cryph::AffVector& u, double uLength,
		const cryph::AffVector& v, double vLength,
		const cryph::AffVector& w, double wLength,
		cryph::AffPoint* vertices)
{
	vertices[0] = llCorner;
	vertices[1] = llCorner + wLength * w;
	vertices[2] = llCorner + uLength * u;
	vertices[3] = vertices[1] + uLength * u;
	vertices[4] = vertices[2] + vLength * v;
	vertices[5] = vertices[3] + vLength * v;
	vertices[6] = vertices[4] - uLength * u;
	vertices[7] = vertices[5] - uLength * u;
	for (int i=0 ; i<8 ; i++)
		updateXYZMinMaxLimits(vertices[i]);
}

void BasicShape::makeIndexLists(int nPointsAroundSide, int nPointsAlongAxis)
{
	int commonIndexListLength = 2 * (nPointsAroundSide + 1); // '2' because two points on adjacent circular cross sections per spot around side
	int firstIndex = 0;
	for (int i=1 ; i<nPointsAlongAxis ; i++) // for each index list:
	{
		indexLists[nIndexListsConstructed].mode = GL_TRIANGLE_STRIP;
		indexLists[nIndexListsConstructed].indices = new int[commonIndexListLength];
		indexLists[nIndexListsConstructed].nIndices = commonIndexListLength;
		indexLists[nIndexListsConstructed].usePerVertexTexCoords = (textureCoords != nullptr);
		indexLists[nIndexListsConstructed].usePerVertexNormals = true;
		// bottom cap has the points with even numbered indices
		int delta = 0;
		for (int j=0 ; j<=2*nPointsAroundSide ; j+=2) // "<=" here so that the copied last column gets its own indices
		{
			// Since vertices are generated along a ruling, corresponding points between
			// two consecutive circular cross sections are one apart in the point array:
			indexLists[nIndexListsConstructed].indices[j  ] = firstIndex + delta;
			indexLists[nIndexListsConstructed].indices[j+1] = firstIndex + delta + 1;
			// then moving from one ruling to the next, we skip over "nPointsAlongAxis"
			// points to get to the next point on a given circular cross section:
			delta += nPointsAlongAxis;
		}
		// done with this index list
		nIndexListsConstructed++;
		// then to get to start of next index list (again, because vertices are generated
		// along a ruling):
		firstIndex++;
	}
}

void BasicShape::makeRuledSurfaceBetweenCircles(
		const cryph::AffPoint& Pbottom, cryph::AffVector& axis,
		double height, double radiusAtBottom, double radiusAtTop,
		int nPointsAroundSide, int nPointsAlongAxis,
		Caps capSpec, double sMin, double sMax, double tMin, double tMax,
		const cryph::AffVector& sZero)
{
	nPoints = nPointsAlongAxis * (nPointsAroundSide + 1); // "+1" so that lastCol==firstCol
	int nPointsWithoutCaps = nPoints;
	int nPointsInOneCap = nPointsAroundSide + 1;
	int addForCapAtBottom = -1, addForCapAtTop = -1;
	if (alwaysGeneratePerVertexNormals)
	{
		if ((capSpec == CAP_AT_BOTTOM) || (capSpec == CAP_AT_BOTH))
		{
			addForCapAtBottom = 3*nPoints;
			nPoints += nPointsInOneCap;
		}
		if ((capSpec == CAP_AT_TOP) || (capSpec == CAP_AT_BOTH))
		{
			addForCapAtTop = 3*nPoints;
			nPoints += nPointsInOneCap;
		}
	}
	int firstAtBottom = addForCapAtBottom, firstAtTop = addForCapAtTop;
	pointCoords = new float[nPoints * 3]; // "*3": (x, y, z)
	normals = new float[nPoints * 3]; // "*3": (dx, dy, dz)
	// height and radius deltas along the axis
	double dh = height / (nPointsAlongAxis - 1);
	double dr = (radiusAtTop - radiusAtBottom) / (nPointsAlongAxis - 1);
	// texture parameters, if generating texture coordinates
	double s, ds, dt;
	if ((sMax > sMin) && (tMax > tMin))
	{
		s = sMin;
		ds = (sMax - sMin) / nPointsAroundSide;
		dt = (tMax - tMin) / (nPointsAlongAxis - 1);
		textureCoords = new float[nPointsWithoutCaps * 2]; // "*2": (s, t)
	}
	cryph::AffVector u(sZero), v;
	cryph::AffVector::coordinateSystemFromUW(u, v, axis);
	cryph::AffPoint Ptop = Pbottom + height*axis;
	cryph::AffVector negAxis = -axis;
	double theta = 0.0, dTheta = 2.0*M_PI/nPointsAroundSide;
	int loc2 = 0, loc3 = 0;
	for (int i=0 ; i<nPointsAroundSide ; i++)
	{
		cryph::AffVector vecAwayFromAxis = cos(theta)*u + sin(theta)*v;
		theta += dTheta;
		// compute common normal along axis
		cryph::AffPoint pb = Pbottom + radiusAtBottom*vecAwayFromAxis;
		cryph::AffPoint pt = Ptop + radiusAtTop*vecAwayFromAxis;
		cryph::AffVector ruling = pt - pb;
		cryph::AffVector vPar, nHat;
		ruling.decompose(axis, vPar, nHat);
		if (nHat.normalize() < BasicUnitTol)
			// two radii must be (nearly) the same; normal is vecAwayFromAxis
			nHat = vecAwayFromAxis;
		// initialize parameters for points along this ruling
		double t = tMin, h = 0.0, r = radiusAtBottom;
		for (int j=0 ; j<nPointsAlongAxis ; j++)
		{
			cryph::AffPoint pnt = Pbottom + h*axis + r*vecAwayFromAxis;
			updateXYZMinMaxLimits(pnt);
			pnt.aCoords(pointCoords, loc3);
			nHat.vComponents(normals, loc3);
			loc3 += 3;
			if ((j == 0) && (addForCapAtBottom >= 0))
			{
				pnt.aCoords(pointCoords, addForCapAtBottom);
				negAxis.vComponents(normals, addForCapAtBottom);
				addForCapAtBottom += 3;
			}
			if ((j == (nPointsAlongAxis-1)) && (addForCapAtTop >= 0))
			{
				pnt.aCoords(pointCoords, addForCapAtTop);
				axis.vComponents(normals, addForCapAtTop);
				addForCapAtTop += 3;
			}
			if (textureCoords != nullptr)
			{
				textureCoords[loc2  ] = s;
				textureCoords[loc2+1] = t;
				loc2 += 2;
				t += dt;
			}
			h += dh; r += dr;
		}
		s += ds;
	}
	// make lastCol == firstCol
	for (int j=0 ; j<nPointsAlongAxis*3 ; j++) // '*3' because we are copying (x,y,z) at each point
	{
		pointCoords[loc3+j] = pointCoords[j];
		normals[loc3+j] = normals[j];
	}
	for (int j=0 ; j<3 ; j++)
	{
		if (addForCapAtBottom >= 0)
		{
			pointCoords[addForCapAtBottom+j] = pointCoords[firstAtBottom+j];
			normals[addForCapAtBottom+j] = normals[firstAtBottom+j];
		}
		if (addForCapAtTop >= 0)
		{
			pointCoords[addForCapAtTop+j] = pointCoords[firstAtTop+j];
			normals[addForCapAtTop+j] = normals[firstAtTop+j];
		}
	}
	if (textureCoords != nullptr)
	{
		for (int j=0 ; j<nPointsAlongAxis ; j++)
		{
			// all 's' values need to be sMax:
			textureCoords[loc2  ] = sMax;
			// all 't' values need to be a copy of first column:
			textureCoords[loc2+1] = textureCoords[2*j+1];
			loc2 += 2;
		}
	}
	allocateIndexListsForConeCyl(nPointsAlongAxis, capSpec);
	if (nPointsAlongAxis == 2)
		makeDrawArraysDataForConeCyl(nPointsWithoutCaps, firstAtBottom, firstAtTop);
	else
	{
		makeIndexLists(nPointsAroundSide, nPointsAlongAxis);
		indexLists[0].thisStartsFace = 0;
	}
	if (alwaysGeneratePerVertexNormals)
		addCapsPerVertexNormals(capSpec, nPointsAlongAxis, firstAtBottom,
			firstAtTop, nPointsInOneCap);
	else
		addCapsFixedNormal(capSpec, nPointsAroundSide, axis, nPointsAlongAxis);
}

BasicShape* BasicShape::makeSphere(const cryph::AffPoint& center, double radius,
									 int nPointsAroundSide, int nPointsAlongAxis,
									 double sMin, double sMax, double tMin, double tMax,
									 const cryph::AffVector& upAxis, const cryph::AffVector& sZero)
{
	if ((radius <= 0.0) || (nPointsAroundSide < 3) || (nPointsAlongAxis < 3))
		return nullptr;
	cryph::AffVector u(sZero), v, w(upAxis);
	cryph::AffVector::coordinateSystemFromUW(u, v, w);
	BasicShape* bs = new BasicShape();
	bs->makeSphere(center, u, v, upAxis, radius,
				   nPointsAroundSide, nPointsAlongAxis, sMin, sMax, tMin, tMax);
	return bs;
}

void BasicShape::makeSphere(const cryph::AffPoint& center,
			const cryph::AffVector& u, const cryph::AffVector& v, const cryph::AffVector& w,
			double radius, int nPointsAroundSide, int nPointsAlongAxis,
			double sMin, double sMax, double tMin, double tMax)
{
	nPoints = nPointsAlongAxis * (nPointsAroundSide + 1); // "+1" so that lastCol==firstCol
	pointCoords = new float[nPoints * 3]; // "*3": (x, y, z)
	normals = new float[nPoints * 3]; // "*3": (dx, dy, dz)
	// height and radius deltas along the axis
	const double maxPhi = 0.995*(M_PI/2.0); // don't go quite up to the pole
	const double minPhi = -maxPhi;
	double dPhi = (maxPhi - minPhi) / (nPointsAlongAxis - 1);
	// texture parameters, if generating texture coordinates
	double s, ds, dt;
	if ((sMax > sMin) && (tMax > tMin))
	{
		s = sMin;
		ds = (sMax - sMin) / nPointsAroundSide;
		dt = (tMax - tMin) / (nPointsAlongAxis - 1);
		textureCoords = new float[nPoints * 2]; // "*2": (s, t)
	}

	double theta = 0.0, dTheta = 2.0*M_PI/nPointsAroundSide;
	int loc2 = 0, loc3 = 0;
	for (int i=0 ; i<nPointsAroundSide ; i++)
	{
		// initialize parameters for points along this ruling
		double t = tMin;
		double phi = minPhi;
		for (int j=0 ; j<nPointsAlongAxis ; j++)
		{
			cryph::AffVector nHat = cos(phi)*(cos(theta)*u + sin(theta)*v) + sin(phi)*w;
			cryph::AffPoint pnt = center + radius*nHat;
			updateXYZMinMaxLimits(pnt);
			pnt.aCoords(pointCoords, loc3);
			nHat.vComponents(normals, loc3);
			loc3 += 3;
			if (textureCoords != nullptr)
			{
				textureCoords[loc2  ] = s;
				textureCoords[loc2+1] = t;
				loc2 += 2;
				t += dt;
			}
			phi += dPhi;
		}
		theta += dTheta;
		s += ds;
	}
	// make lastCol == firstCol
	for (int j=0 ; j<nPointsAlongAxis*3 ; j++) // '*3' because we are copying (x,y,z) at each point
	{
		pointCoords[loc3+j] = pointCoords[j];
		normals[loc3+j] = normals[j];
	}
	if (textureCoords != nullptr)
	{
		for (int j=0 ; j<nPointsAlongAxis ; j++)
		{
			// all 's' values need to be sMax:
			textureCoords[loc2  ] = sMax;
			// all 't' values need to be a copy of first column:
			textureCoords[loc2+1] = textureCoords[2*j+1];
			loc2 += 2;
		}
	}
	allocateIndexListsForSphere(nPointsAlongAxis);
	makeIndexLists(nPointsAroundSide, nPointsAlongAxis);
	indexLists[0].thisStartsFace = 0;
}

void BasicShape::setAlwaysGeneratePerVertexNormals(bool b)
{
	alwaysGeneratePerVertexNormals = b;
}

void BasicShape::updateOneCoordLimit(double c, double minMax[2])
{
	if (minMax[0] > minMax[1])
		// not yet initialized
		minMax[0] = minMax[1] = c;
	else
	{
		if (c < minMax[0])
			minMax[0] = c;
		else if (c > minMax[1])
			minMax[1] = c;
	}
}

void BasicShape::updateXYZMinMaxLimits(const cryph::AffPoint& p)
{
	updateOneCoordLimit(p[0], xyzMinMax);
	updateOneCoordLimit(p[1], &xyzMinMax[2]);
	updateOneCoordLimit(p[2], &xyzMinMax[4]);
}
