/** @file ProjPoint.h
 *  Class definition for 4D Projective Space points.
 *  This is OPEN SOURCE software developed by James R. Miller (jrmiller@ku.edu)
 *  Original version: ca. 1996. See README_*.txt for more information.
 */

#ifndef PROJPOINT_H
#define PROJPOINT_H

#include "AffPoint.h"

namespace cryph
{

class ProjPoint
{
public:
	/** Default constructor creates ProjPoint (0,0,0,1) */
	ProjPoint();

	/** The copy constructor.
	 *  @param p the ProjPoint whose coordinates are used to initialize "this".
	 */
	ProjPoint(const ProjPoint& p);

	/** Construct a projective space point from an affine point an an explicit w.
	 *  The constructed point will be (w*p.x, w*p.y, w*p.z, w).
	 *  @param p an affine point'
	 *  @param w the w plane of projective space onto which p is to be projected.
	 */
	ProjPoint(const AffPoint& p, double w=1.0);

	/** Construct a projective space point using 4 double precision coordinates
     *  in the given array
	 *  @param p assumed to be a double precision array of length 4 holding the
	 *           coordinates of a point in projective space.
	 */
	ProjPoint(const double* p);

	/** Construct a projective space point using 4 single precision coordinates
     *  in the given array
	 *  @param p assumed to be a single precision array of length 4 holding the
	 *           coordinates of a point in projective space.
	 */
	ProjPoint(const float* p);

	/** Construct a projective space point from the given 4 coordinates. The
	 *  coordinates are assumed to be already embedded in projective space.
	 *  @param xx the affine x coordinate already multiplied by w
	 *  @param yy the affine y coordinate already multiplied by w
	 *  @param zz the affine z coordinate already multiplied by w
	 *  @param ww the w coordinate
	 */
	ProjPoint(double xx, double yy, double zz=0.0, double ww=1.0);

	/** The destructor. */
	virtual ~ProjPoint();

	/** The assignment operator
	 *  @param rhs the projective space point to be assigned to "this".
	 *  @return "this" ProjPoint
	 */
	ProjPoint operator=(const ProjPoint& rhs);

	/** The += operator
	 *  @param rhs the projective space point to be accumulated into "this".
	 *  @return "this" ProjPoint
	 */
	ProjPoint operator+=(const ProjPoint& rhs);

	/** The *= operator for scalar multiplication
	 *  @param f the scalar to be used to multiply all 4 coordinates of "this".
	 *  @return "this" ProjPoint
	 */
	ProjPoint operator*=(double f);

	/** The /= operator for scalar division
	 *  @param f the scalar to be used to divide into all 4 coordinates of "this".
	 *  @return "this" ProjPoint
	 */
	ProjPoint operator/=(double f);

	/** Read-only indexing
	 *  @param index 0 <= index <= 3 (0: x; 1: y: 2: z; 3: w)
	 *  @return the requested coordinate
	 *  @see AffPoint where indexing constants are defined
	 */
	double operator[](int index) const;

	/** Return the sum of "this" point and another, leaving "this" point
	 *  unchanged.
	 *  @param p2 the second point to be summed
	 *  @return the sum
	 */
	ProjPoint operator+(const ProjPoint& p2) const
				{ return ProjPoint(x+p2.x, y+p2.y, z+p2.z, w+p2.w); }

	/** Return the difference between "this" point and another, leaving "this"
	 *  point unchanged.
	 *  @param p2 the point to be subtracted from "this".
	 *  @return the difference
	 */
	ProjPoint operator-(const ProjPoint& p2) const
				{ return ProjPoint(x-p2.x, y-p2.y, z-p2.z, w-p2.w); }

	/** Return the scalar product of "this" ProjPoint and a scalar, leaving
	 *  "this" point unchanged.
	 *  @param f the scalar
	 *  @return the product: f*this
	 */
	ProjPoint operator*(double f) const
				{ return ProjPoint(f*x, f*y, f*z, f*w); }

	/** Return the quotient of "this" ProjPoint and a scalar, leaving "this"
	 *  ProjPoint unchanged.
	 *  @param f the scalar
	 *  @return the quotient: this/f
	 */
	ProjPoint operator/(double f) const
				{ return ProjPoint(x/f, y/f, z/f, w/f); }

	/** Return the affine coordinates of this ProjPoint in the given array
	 *  @param coords an array of doubles to hold the 3 affine coordinates (must
	 *                be at least length offset+3.
	 *  @param offset the starting location in coords where the 3 affine
	 *                coordinates are to be written.
	 *  @return the base address of the given "coords" array so that this
	 *          method call can be used as a parameter to a function expecting
	 *          an array of double.
	 */
	double*	aCoords(double coords[], int offset=0) const;

	/** Return the affine coordinates of this ProjPoint in the given array of
	 *  3-element values. (Mimics an OpenGL GLSL array of dvec3.)
	 *  @param coords an array of double[3] to hold the 3 affine coordinates
	 *                The first dimension must be at least offset+1.
	 *  @param offset the starting location in coords where the 3 affine
	 *                coordinates are to be written.
	 *  @return the base address of the given "coords" array so that this
	 *          method call can be used as a parameter to a function expecting
	 *          an array of "dvec3".
	 */
	double*	aCoords(double coords[][3], int offset=0) const;

	/** Return the affine coordinates of this ProjPoint in the given array
	 *  @param coords an array of float to hold the 3 affine coordinates (must
	 *                be at least length offset+3.
	 *  @param offset the starting location in coords where the 3 affine
	 *                coordinates are to be written.
	 *  @return the base address of the given "coords" array so that this
	 *          method call can be used as a parameter to a function expecting
	 *          an array of float.
	 */
	float* aCoords(float coords[], int offset=0) const;

	/** Return the affine coordinates of this ProjPoint in the given array of
	 *  3-element values. (Mimics an OpenGL GLSL array of vec3.)
	 *  @param coords an array of float[3] to hold the 3 affine coordinates
	 *                The first dimension must be at least offset+1.
	 *  @param offset the starting location in coords where the 3 affine
	 *                coordinates are to be written.
	 *  @return the base address of the given "coords" array so that this
	 *          method call can be used as a parameter to a function expecting
	 *          an array of "vec3".
	 */
	float* aCoords(float coords[][3], int offset=0) const;

	/** Fetch the affine coordinates of the point corresponding to this ProjPoint
	 *  instance into the given AffPoint instance.
	 *  @param aPnt the AfPoint into which the projected point is to be written.
	 */
	void aCoords(AffPoint& aPnt) const    // ... into an AffPoint
				{ aPnt = AffPoint(x/w, y/w, z/w); }

	/** Return an AffPoint holding the projected coordinates of this ProjPoint.
	 *  @return the projected AffPoint.
	 */
	AffPoint aCoords() const    // ... into an AffPoint
				{ return AffPoint(x/w, y/w, z/w); }

	/** Return the projective coordinates of this ProjPoint in the given array
	 *  @param coords an array of doubles to hold the 4 projective space
	 *                coordinates (must be at least length offset+4.
	 *  @param offset the starting location in coords where the 4 projective
	 *                space coordinates are to be written.
	 *  @return the base address of the given "coords" array so that this
	 *          method call can be used as a parameter to a function expecting
	 *          an array of double.
	 */
	double*	pCoords(double* coords, int offset=0) const;

	/** Return the projective space coordinates of this ProjPoint in the given
	 *  array of 4-element values. (Mimics an OpenGL GLSL array of dvec4.)
	 *  @param coords an array of double[4] to hold the 4 projective space
	 *                coordinates. The first dimension must be at least offset+1.
	 *  @param offset the starting location in coords where the 4 projective
	 *                space coordinates are to be written.
	 *  @return the base address of the given "coords" array so that this
	 *          method call can be used as a parameter to a function expecting
	 *          an array of "dvec4".
	 */
	double*	pCoords(double coords[][4], int offset=0) const;

	/** Return the projective coordinates of this ProjPoint in the given array
	 *  @param coords an array of float to hold the 4 projective space
	 *                coordinates (must be at least length offset+4.
	 *  @param offset the starting location in coords where the 4 projective
	 *                space coordinates are to be written.
	 *  @return the base address of the given "coords" array so that this
	 *          method call can be used as a parameter to a function expecting
	 *          an array of float.
	 */
	float* pCoords(float* coords, int offset=0) const;

	/** Return the projective space coordinates of this ProjPoint in the given
	 *  array of 4-element values. (Mimics an OpenGL GLSL array of vec4.)
	 *  @param coords an array of float[4] to hold the 4 projective space
	 *                coordinates. The first dimension must be at least offset+1.
	 *  @param offset the starting location in coords where the 4 projective
	 *                space coordinates are to be written.
	 *  @return the base address of the given "coords" array so that this
	 *          method call can be used as a parameter to a function expecting
	 *          an array of "vec4".
	 */
	float* pCoords(float coords[][4], int offset=0) const;

	/** A method inspired by OpenGL's GLSL's swizzle operation. If the i-th
	 *  character of xyzw is 'x', then the x coordinate of "this" point is
	 *  placed into the i-th component of "this" point. If instead it is 'X',
	 *  then negative of the x coordinate of "this" point is placed into the
	 *  i-th component of "this" point. Characters 'y', 'Y', 'z', 'Z', 'w',
	 *  'W' behave similarly. Any other character is simply ignored. Some
	 *  examples:<pre>
	 cryph::ProjPoint pp(4.0, 5.0, 6.0, 1.0);
	 pp.swizzle("Zxy "); // pp is now (-6.0, 4.0, 5.0, 1.0)
	 pp.swizzle("dzyw"); // pp is now (-6.0, 5.0, 4.0, 1.0)
	 pp.swizzle("xxyw"); // pp is now (-6.0, -6.0, 5.0, 1.0)</pre>
	 * @param xyz [INPUT ONLY] an array of char of length 4 interpreted as
	 *            explained and illustrated above
	 */
	void swizzle(char xyzw[4]);

	// Coordinates are public, but you should try to use direct access
	// to these only for simple alteration of coordinates. Otherwise
	// use public methods.
	double	x; /**< x coordinate */
	double	y; /**< y coordinate */
	double	z; /**< z coordinate */
	double	w; /**< w coordinate */
};

std::ostream&	operator<<(std::ostream& os, const ProjPoint& p);
std::istream&	operator>>(std::istream& is, ProjPoint& p);

static ProjPoint operator*(double f, const ProjPoint& p)
	{ return ProjPoint(f*p[X],f*p[Y],f*p[Z],f*p[W]); }
}

#endif
