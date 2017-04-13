/** @file BasicShape.h
 *  Class definition for common 3D shapes Block, Cylinder, Cone, and Sphere.
 *  <p>BasicShape instances can only be created using one of the public factory
 *  methods. There are no public constructors for class BasicShape.</p>
 *  <p>The public <i>instance</i> methods of this class are intended to be used
 *  only by class BasicShapeRenderer, although it is  not inconceivable that they
 *  might be useful under certain conditions by other client classes.</p>
 *  @see BasicShapeRenderer
 */

// This is OPEN SOURCE software developed by James R. Miller (jrmiller@ku.edu)

#ifndef BASICSHAPE_H
#define BASICSHAPE_H

#ifdef __APPLE_CC__
#include "GLFW/glfw3.h"
#else
#include <GL/gl.h>
#endif

#include "AffPoint.h"
#include "AffVector.h"

class BasicShape
{
public:
	/** The destructor */
	virtual ~BasicShape();

	/** An enumerated type that allows specification of whether cylinders and
	 *  cones should have planar caps at the top and/or at the bottom. */
	enum Caps { CAP_AT_NEITHER, CAP_AT_BOTTOM, CAP_AT_TOP, CAP_AT_BOTH };

	/** A factory method to create a block (actually a 3D parallelepiped). The
	 *  uEdge, vEdge, and wEdge vectors determine the u, v, and w edge directions of
	 *  the parallelepiped. There is no requirement that they be perpendicular to one
	 *  another. They need only be linearly independent.
	 *  <p>Texture coordinates are not generated for block shapes. However
	 *  it will be of use to clients to know that the 8 vertices are
	 *  stored in the VBO in the order: (uMin, vMin, wMin), (uMin, vMin, wMax),
	 *  (uMax, vMin, wMin), (uMax, vMin, wMax), (uMax, vMax, wMin), (uMax, vMax, wMax),
	 *  (uMin, vMax, wMin), (uMin, vMax, wMax). Furthermore, the faces are numbered
	 *  and rendered as: [0:  the vMin face], [1: the uMax face], [2: the vMax face],
	 *  [3: uMin face], [4: wMin face], and [5: wMax face].
	 *  Faces 0-2 are drawn using glDrawArrays calls with a count of 4 and "first"
	 *  index = 0, 2, and 4, respectively;
	 *  faces 3-5 are drawn using glDrawElements with the following indices: [face 3:
	 *  6, 7, 0, 1], [face 4: 0, 2, 6, 4], and [face 5: 1, 3, 7, 5]. Using this information,
	 *  clients of BasicShapeRenderer can apply texture to any and all
	 *  block faces when the corresponding "prepareForFace" callback passed to
	 *  BasicShapeRenderer::drawShape is invoked during display callbacks. The easiest
	 *  way to do this is to use the utility BasicShapeRenderer::setTexCoordsForBlock,
	 *  passing that method the current face index.</p>
	 *  <p>If either the uEdge vector or the vEdge vector is the zero vector, OR if they
	 *  are parallel to one another, nullptr is returned. These are the only error conditions.
	 *  A block will be created as specified below in all other cases.</p>
	 *  @param llCorner the (x,y,z) coordinates of the corner of the block
	 *  @param uEdge a vector specifying the u direction of the block. If this is a zero
	 *         vector, nullptr is returned.
	 *  @param uLength if uLength > 0, this will be used as the length of the block in
	 *         the u direction. If uLength <= 0, the length of the uEdge vector will be used.
	 *  @param vEdge a vector specifying the v direction of the block. If this is a zero
	 *         vector or a scalar multiple of the uEdge vector, nullptr is returned.
	 *  @param vLength if vLength > 0, this will be used as the length of the block in
	 *         the v direction. If vLength <= 0, the length of the vEdge vector will be used.
	 *  @param wEdge a vector specifying the w direction of the block. If this is a zero
	 *         vector or in the plane spanned by (uEdge, vEdge), then a w direction of
	 *         uEdge x vEdge will be used for the w direction.
	 *  @param wLength if wLength > 0, this will be used as the length of the block in
	 *         the w direction. If wLength <= 0, the length of the wEdge vector will be used.
	 *         If BOTH wLength<0 and the w direction was computed from a cross product as
	 *         described above, then the wLength will be set to 1.0.
	 *  @return the BasicShape pointer, or nullptr if invalid data was provided
	 */
	static BasicShape* makeBlock(
		const cryph::AffPoint& llCorner,
		const cryph::AffVector& uEdge, double uLength,
		const cryph::AffVector& vEdge, double vLength,
		const cryph::AffVector& wEdge, double wLength);

	/** A factory method to create a piecewise linear approximation to a bounded cone.
	 *  Texture coordinates are generated only if (sMax>sMin) AND (tMax>tMin). If
	 *  texture coordinates are generated, the 's' texture direction is around the
	 *  circumference of the cone; the 't' texture direction is along a cone ruling.
	 *  If invalid input is given (negative radii, coincident points, or any other
	 *  invalid specification), nullptr is returned.
	 *  @param Pbottom the (x,y,z) coordinates for the bottom of the cone
	 *  @param Ptop the (x,y,z) coordinates for the top of the cone
	 *  @param radiusAtBottom the radius of the circular cross section at Pbottom
	 *  @param radiusAtTop the radius of the circular cross section at Ptop (if
	 *         radiusAtBottom==radiusAtTop, the BasicShape will be equivalent to
	 *         a cylinder)
	 *  @param nPointsAroundSide the number of points in the piecewise linear
	 *         approximation to the cone that will be generated around a circular
	 *         cross section. Must be at least 3.
	 *  @param nPointsAlongAxis the number of points in the piecewise linear
	 *         approximation to the cone that will be generated along a cone ruling.
	 *         This must be at least 2. Moreover, 2 is generally perfectly adequate.
	 *  @param capSpec specifies whether circular disks will be generated at the top
	 *         and/or bottom of the generated cone shape. Any created disks are created
	 *         as a GL_TRIANGLE_FAN, reusing the points on the top and/or bottom
	 *         piecewise linear cross section.
	 *  @param sMin the 's' texture coordinate associated with the sZero vector direction
	 *         described below
	 *  @param sMax the 's' texture coordinate associated with the final point generated
	 *         on the circular cross section.
	 *  @param tMin the 't' texture coordinate associated with the bottom of the cone
	 *  @param tMax the 't' texture coordinate associated with the top of the cone
	 *  @param sZero the component of the optional vector "sZero" that is perpendicular
	 *         to the axis of the cone (axis = Ptop - Pbottom) will define the direction
	 *         for texture s coordinate=0. It serves no other externally significant
	 *         purpose. If the component of "sZero" perpendicular to the axis is the zero
	 *         vector, an arbitrary vector perpendicular to the axis is used.
	 *  @return the BasicShape pointer, or nullptr if invalid data was provided
	 */
	static BasicShape* makeBoundedCone(
		const cryph::AffPoint& Pbottom, const cryph::AffPoint& Ptop,
		double radiusAtBottom, double radiusAtTop,
		int nPointsAroundSide, int nPointsAlongAxis,
		Caps capSpec=CAP_AT_BOTH,
		double sMin=0, double sMax=0, double tMin=0, double tMax=0,
		const cryph::AffVector& sZero=cryph::AffVector::xu);

	/** A factory method to create a piecewise linear approximation to a bounded cylinder.
	 *  Texture coordinates are generated only if (sMax>sMin) AND (tMax>tMin). If
	 *  texture coordinates are generated, the 's' texture direction is around the
	 *  circumference of the cylinder; the 't' texture direction is along a cylinder ruling.
	 *  If invalid input is given (negative radius, coincident points, or any other
	 *  invalid specification), nullptr is returned.
	 *  @param Pbottom the (x,y,z) coordinates for the bottom of the cylinder
	 *  @param Ptop the (x,y,z) coordinates for the top of the cylinder
	 *  @param radius the radius of the cylinder
	 *  @param nPointsAroundSide the number of points in the piecewise linear
	 *         approximation to the cylinder that will be generated around a circular
	 *         cross section. Must be at least 3.
	 *  @param nPointsAlongAxis the number of points in the piecewise linear
	 *         approximation to the cylinder that will be generated along a cylinder ruling.
	 *         This must be at least 2. Moreover, 2 is generally perfectly adequate.
	 *  @param capSpec specifies whether circular disks will be generated at the top
	 *         and/or bottom of the generated cylinder shape. Any created disks are created
	 *         as a GL_TRIANGLE_FAN, reusing the points on the top and/or bottom
	 *         piecewise linear cross section.
	 *  @param sMin the 's' texture coordinate associated with the sZero vector direction
	 *         described below
	 *  @param sMax the 's' texture coordinate associated with the final point generated
	 *         on the circular cross section.
	 *  @param tMin the 't' texture coordinate associated with the bottom of the cylinder
	 *  @param tMax the 't' texture coordinate associated with the top of the cylinder
	 *  @param sZero the component of the optional vector "sZero" that is perpendicular
	 *         to the axis of the cylinder (axis = Ptop - Pbottom) will define the direction
	 *         for texture s coordinate=0. It serves no other externally significant
	 *         purpose. If the component of "sZero" perpendicular to the axis is the zero
	 *         vector, an arbitrary vector perpendicular to the axis is used.
	 *  @return the BasicShape pointer, or nullptr if invalid data was provided
	 */
	static BasicShape* makeBoundedCylinder(
		const cryph::AffPoint& Pbottom, const cryph::AffPoint& Ptop,
		double radius,
		int nPointsAroundSide, int nPointsAlongAxis,
		Caps capSpec=CAP_AT_BOTH,
		double sMin=0, double sMax=0, double tMin=0, double tMax=0,
		const cryph::AffVector& sZero=cryph::AffVector::xu);

	/** A factory method to create a piecewise linear approximation to a sphere.
	 *  Texture coordinates are generated only if (sMax>sMin) AND (tMax>tMin). If
	 *  texture coordinates are generated, the 's' texture direction is determined by
	 *  the sZero vector described below, and the 't' texture direction is determined by
	 *  the upAxis vector described below.
	 *  If invalid input is given (negative radius or any other invalid specification),
	 *  nullptr is returned.
	 *  @param center the (x,y,z) coordinates of the center of the sphere
	 *  @param radius the radius of the sphere
	 *  @param nPointsAroundSide the number of points in the piecewise linear
	 *         approximation to the sphere that will be generated around a circular
	 *         cross section perpendicular to the upAxis described below. Must be at least 3.
	 *  @param nPointsAlongAxis the number of points in the piecewise linear
	 *         approximation to the sphere that will be generated along circular cross
	 *         sections in the upAxis direction. Must be at least 3. In general, an odd
	 *         number will work best because that forces points along the equator to be
	 *         generated.
	 *  @param sMin the 's' texture coordinate associated with the sZero vector direction
	 *         described below
	 *  @param sMax the 's' texture coordinate associated with the final point generated
	 *         on the circular cross section
	 *  @param tMin the 't' texture coordinate associated with the bottom of the sphere
	 *         as determined by the -upAxis vector
	 *  @param tMax the 't' texture coordinate associated with the top of the sphere
	 *         as determined by the +upAxis vector
	 *  @param upAxis determines the top/bottom directions for the sphere, hence where
	 *         the "north pole" and "south pole" are located.
	 *  @param sZero the component of the optional vector "sZero" that is perpendicular
	 *         to upAxis will define the direction for texture s coordinate=0. It serves
	 *         no other externally significant purpose. If the component of "sZero"
	 *         perpendicular to upAxis is the zero vector, an arbitrary vector
	 *         perpendicular to upAxis is used.
	 *  @return the BasicShape pointer, or nullptr if invalid data was provided
	 */
	static BasicShape* makeSphere(
		const cryph::AffPoint& center, double radius,
		int nPointsAroundSide, int nPointsAlongAxis,
		double sMin=0, double sMax=0, double tMin=0, double tMax=0,
		const cryph::AffVector& upAxis=cryph::AffVector::yu,
		const cryph::AffVector& sZero=cryph::AffVector::xu);

	/** Retrieve the MC bounding box for the BasicShape instance
	 *  @param xyzLimits the {xmin, xmax, ymin, ymax, zmin, zmax} bounds will be
	 *         returned in this parameter. It is assumed to be at least of length 6.
	 */
	void getMCBoundingBox(double* xyzLimits) const;

	/** An accessor method <i>intended to be used only by class BasicShapeRenderer</i>
	 *  @return the number of points (equivalently, the number of normals, if applicable
	 *          and the number of texture coordinates, if applicable) that must be sent
	 *          to the GPU in VBOs.
	 */
	int getNumPoints() const { return nPoints; }

	/** An accessor method <i>intended to be used only by class BasicShapeRenderer</i>
	 *  @return the actual array of 3D point coordinates that must be stored in the
	 *          point coordinate VBO.
	 */
	const float* getPointCoords() const { return pointCoords; }

	/** An accessor method <i>intended to be used only by class BasicShapeRenderer</i>
	 *  @return the actual array of 3D normal vectors that must be stored in the normal
	 *          vector VBO. nullptr is returned if there are no vectors to be stored there.)
	 */
	const float* getNormals() const { return normals; }

	/** An accessor method <i>intended to be used only by class BasicShapeRenderer</i>
	 *  @return the actual array of 2D texture coordinates that must be stored in the texture
	 *          coordinate VBO. nullptr is returned if there are no texture coordinates.)
	 */
	const float* getTextureCoords() const { return textureCoords; }

	/** An accessor method <i>intended to be used only by class BasicShapeRenderer</i>
	 *  @return the number of glDrawArrays calls that are needed to render this BasicShape
	 */
	int getNumDrawArraysCalls() const { return nDrawArraysCalls; }

	/** An accessor method <i>intended to be used only by class BasicShapeRenderer</i>
	 *  @param i the glDrawArrays data desired (0 <= i < getNumDrawArraysCalls())
	 *  @param mode the OpenGL draw mode to be used
	 *  @param thisStartsFace an indication of whether this glDrawArrays call is the first
	 *         on a face of the BasicShape. BasicShape instances created as cylinders,
	 *         cones, and spheres will only report starting face 0; BasicShape instances
	 *         created by makeBlock will report faces 0-5 as defined in BasicShape::makeBlock.
	 *         If the current glDrawArrays call does not start a face, thisStartsFace will
	 *         be returned as -1.
	 *         The intended use for this output is to allow a client to "prepare for a face"
	 *         by, for example, establishng texture mapping information. (See the
	 *         BasicShapeRenderer::drawShape method.)
	 *  @param offset the offset parameter to be passed to glDrawArrays
	 *  @param canUseVertexTexCoords true if and only if texture mapping is being used
	 *  @param canUseNormalArray true if and only if a normal vector VBO should be enabled;
	 *         otherwise the normal will be set by passing fixedN to glVertexAttrib
	 *  @param fixedN the normal vectior to be used if canUseNormalArray is false
	 *  @return the number of points to be passed to the glDrawArrays call
	 *  @see BasicShapeRenderer
	 */
	int getDrawArraysData(int i, GLenum& mode, int& thisStartsFace, int& offset,
			bool& canUsePerVertexTexCoords, bool& canUseNormalArray,
			cryph::AffVector& fixedN) const;

	/** An accessor method <i>intended to be used only by class BasicShapeRenderer</i>
	 *  @return the number of glDrawElements calls that are needed to render this BasicShape
	 */
	int getNumIndexLists() const { return nIndexLists; }

	/** An accessor method <i>intended to be used only by class BasicShapeRenderer</i>
	 *  @param i the glDrawElements data desired (0 <= i < getNumIndexLists())
	 *  @param mode the OpenGL draw mode to be used
	 *  @param thisStartsFace an indication of whether this glDrawElements call is the first
	 *         on a face of the BasicShape. BasicShape instances created as cylinders,
	 *         cones, and spheres will only report starting face 0; BasicShape instances
	 *         created by makeBlock will report faces 0-5 as defined in BasicShape::makeBlock.
	 *         If the current glDrawArrays call does not start a face, thisStartsFace will
	 *         be returned as -1.
	 *         The intended use for this output is to allow a client to "prepare for a face"
	 *         by, for example, establishng texture mapping information. (See the
	 *         BasicShapeRenderer::drawShape method.)
	 *  @param nInList the number of indices in the returned index list
	 *  @param type the type of the indices to be specified to glDrawElements
	 *  @param canUseVertexTexCoords true if and only if texture mapping is being used
	 *  @param canUseNormalArray true if and only if a normal vector VBO should be enabled;
	 *         otherwise the normal will be set by passing fixedN to glVertexAttrib
	 *  @param fixedN the normal vectior to be used if canUseNormalArray is false
	 *  @return the index list. At the discretion of the client (e.g., BasicShapeRenderer),
	 *          this will either be passed directly to glDrawElements, or stored in an EBO.
	 *  @see BasicShapeRenderer
	 */
	const void* getIndexList(int i, GLenum& mode, int& thisStartsFace, int& nInList,
			GLenum& type,
			bool& canUsePerVertexTexCoords, bool& canUsePerVertexNormals,
			cryph::AffVector& fixedNormal) const;

	/** An accessor method that returns whether the implementation is currently forcing
	 *  all normal vectors to be stored in VBOs. This is largely for experimental purposes.
	 *  <i>Use of this method is highly discouraged.</i>
	 */
	static bool getAlwaysGeneratePerVertexNormals()
		{ return alwaysGeneratePerVertexNormals; }

	/** A method that can be used to force BasicShape instances to always generate
	 *  per-vertex normals, regardless of whether they are actually needed.
	 *  This is largely for experimental purposes.
	 *  <i>Use of this method is highly discouraged.</i>
	 *  Among other things, it can negatively impact the ability to use the "thisStartsFace"
	 *  functionality described in getDrawArraysData and getIndexList.
	 */
	static void setAlwaysGeneratePerVertexNormals(bool b);

private:
	BasicShape(); // use public factory methods instead
	BasicShape(const BasicShape& s) {} // don't make copies

	// common "worker" routines for cones and cylinders:
	void addCapsFixedNormal(Caps capSpec, int nPointsInCap,
			const cryph::AffVector& axis, int nPointsAlongAxis);
	void addCapsPerVertexNormals(Caps capSpec, int nPointsAlongAxis,
			int firstAtBottom, int firstAtTop, int nPointsInOneCap);
	void addDrawArraysDataForConeCylCaps(int nextIndex,
			int firstAtBottom, int firstAtTop, int nPointsInOneCap);
	void allocateIndexListsForConeCyl(int nPointsAlongAxis, Caps capSpec);
	void allocateIndexListsForSphere(int nPointsAlongAxis);
	void makeBlockData(
		const cryph::AffPoint& llCorner,
		const cryph::AffVector& uEdge, double uLength,
		const cryph::AffVector& vEdge, double vLength,
		const cryph::AffVector& wEdge, double wLength);
	void finishBlockDataUsingFixedNormals(
		const cryph::AffVector& uvNormal, const cryph::AffVector& uwNormal,
		const cryph::AffVector& vwNormal, const cryph::AffPoint* vertices);
	void finishBlockDataUsingPerVertexNormals(
		const cryph::AffVector& uvNormal, const cryph::AffVector& uwNormal,
		const cryph::AffVector& vwNormal, const cryph::AffPoint* vertices);
	void makeDrawArraysDataForConeCyl(int nPointsWithoutCaps,
		int firstAtBottom, int firstAtTop); // if nPointsAlongAxis == 2
	void makeEightBlockVertices(
		const cryph::AffPoint& llCorner,
		const cryph::AffVector& u, double uLength,
		const cryph::AffVector& v, double vLength,
		const cryph::AffVector& w, double wLength,
		cryph::AffPoint* vertices);
	void makeIndexLists(int nPointsAroundSide, int nPointsAlongAxis);
	void makeRuledSurfaceBetweenCircles(
		const cryph::AffPoint& Pbottom, cryph::AffVector& axis,
		double height, double radiusAtBottom, double radiusAtTop,
		int nPointsAroundSide, int nPointsAlongAxis,
		Caps capSpec, double sMin, double sMax, double tMin, double tMax,
		const cryph::AffVector& sZero);
	void makeSphere(const cryph::AffPoint& center,
					const cryph::AffVector& u, const cryph::AffVector& v, const cryph::AffVector& w,
					double radius,
					int nPointsAroundSide, int nPointsAlongAxis,
					double sMin, double sMax, double tMin, double tMax);
	static void updateOneCoordLimit(double c, double minMax[2]);
	void updateXYZMinMaxLimits(const cryph::AffPoint& p);

	// geometry and per-vertex attributes
	float* pointCoords;
	float* normals;
	float* textureCoords;
	int nPoints;
	// always generate per-vertex normals, even for flat faces?
	static bool alwaysGeneratePerVertexNormals;
	// for bounding box
	double xyzMinMax[6];

	struct DrawArraysCallData
	{
		GLenum mode;
		int thisStartsFace; // >= 0 ==> this starts a new face on the original shape
		int offset;
		int nPointsInCall;
		bool usePerVertexTexCoords, usePerVertexNormals;
		cryph::AffVector useFixedNormal; // if !usePerVertexNormals
		DrawArraysCallData() : thisStartsFace(-1) {}
	};
	DrawArraysCallData* drawArraysCallData;
	int nDrawArraysCalls;

	struct IndexListData
	{
		GLenum mode;
		int thisStartsFace; // >= 0 ==> this starts a new face on the original shape
		int* indices;
		int nIndices;
		bool usePerVertexTexCoords, usePerVertexNormals;
		cryph::AffVector useFixedNormal; // if !usePerVertexNormals
		IndexListData(): thisStartsFace(-1) {}
	};
	IndexListData* indexLists;
	int nIndexLists, nIndexListsConstructed;
};

#endif
