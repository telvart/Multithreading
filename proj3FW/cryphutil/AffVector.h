/** @file AffVector.h
 *  Class definition for 3D vectors associated with an affine space.
 *  This is OPEN SOURCE software developed by James R. Miller (jrmiller@ku.edu)
 *  Original version: ca. 1996. See README_*.txt for more information.
 */

#ifndef AFFVECTOR_H
#define AFFVECTOR_H

#include <math.h>
#include <iostream>

namespace cryph
{
class AffPoint;

// indices for extracting components from vectors

const int DX = 0;
const int DY = 1;
const int DZ = 2;
const int DW = 3;

class AffVector
{
public:
	/** Default constructor creates a zero vector */
	AffVector();

	/** The copy constructor
	 *  @param v [INPUT ONLY] the vector whose components are used to initialize "this" vector
	 */
	AffVector(const AffVector& v);

	/** Construct a vector from a point (i.e., "this" vector = p - origin)
	 *  @param p [INPUT ONLY] the point
	 */
	AffVector(const AffPoint& p);

	/** Creates a vector with the given components
	 *  @param Dx [INPUT ONLY] the x component
	 *  @param Dy [INPUT ONLY] the y component
	 *  @param Dz [INPUT ONLY] the z component (defaults to 0.0)
	 */
	AffVector(double Dx, double Dy, double Dz=0.0);

	/** Construct from a double array
	 *  @param xyz [INPUT ONLY] an array assumed to be length 3 holding (dx, dy, dz)
	 */
	AffVector(const double xyz[]);

	/** Construct from a float array
	 *  @param xyz [INPUT ONLY] an array assumed to be length 3 holding (dx, dy, dz)
	 */
	AffVector(const float xyz[]);

	/** The destructor */
	virtual ~AffVector();

	/** Assigns the components of rhs to "this" vector
	 *  @param rhs [INPUT ONLY] a const reference to an AffVector
	 *  @return the value assigned
     */
	AffVector operator=(const AffVector& rhs);

	/** Adds the components of rhs to "this" vector
	 *  @param rhs [INPUT ONLY] a const reference to an AffVector
	 *  @return the result of incrementing AffVector
	 */
	AffVector operator+=(const AffVector& rhs);

	/** Subtracts the components of rhs from "this" vector
	 *  @param rhs [INPUT ONLY] a const reference to an AffVector
	 *  @return the result of decrementing the AffVector
	 */
	AffVector operator-=(const AffVector& rhs);

	/** Scales the components of "this" vector by f
	 *  @param f [INPUT ONLY] the scale factor
	 *  @return the value of the scaled AffVector
	 */
	AffVector operator*=(double f);

	/** Divides the components of "this" vector by f (i.e., scales it by 1/f)
	 *  @param f [INPUT ONLY] the divisor
	 *  @return the value of the divided AffVector
	 */
	AffVector operator/=(double f);

	/** Returns dx (if index = 0); dy If index = 1); dz (if index = 2)
	 *  The returned value cannot be used as an kl-value.
	 *  @return the selected vector component
	 */
	double operator[](int index) const;

	/** Adds "this" vector to v2 and returns the result as an AffVector
	 *  @param v2 [INPUT ONLY] the vector to add
	 *  @return the sum of "this" vector and v2
	 */
	AffVector	operator+(const AffVector& v2) const
				{ return AffVector(dx+v2.dx , dy+v2.dy , dz+v2.dz); }

	/** Subtracts v2 from "this" vector and returns the result as an AffVector
	 *  @param v2 [INPUT ONLY] the vector to subtract
	 *  @return the difference of the two vectors
	 */
	AffVector	operator-(const AffVector& v2) const
				{ return AffVector(dx-v2.dx , dy-v2.dy , dz-v2.dz); }

	/** Unary negation operator.
	 *  @return the negated value of "this" vector
	 */
	AffVector	operator-() const { return AffVector(-dx, -dy, -dz); }

	/** Scales "this" vector by f and returns the result as an AffVewctor
	 *  @param f [INPUT ONLY] the scale factor
	 *  @return the scaled AffVector
	 */
	AffVector	operator*(double f) const
				{ return AffVector(f*dx , f*dy , f*dz); }

	/** Divide "this" vector by f (i.e., scale by 1/f)
	 *  @param f [INPUT ONLY] the divisor
	 *  @return the AffVector divided by f
	 */
	AffVector	operator/(double f) const
				{ return AffVector(dx/f , dy/f , dz/f); }

	/** Generate a vector normal to "this" vector and return in formal parameter
	 *  @param normal [OUTPUT ONLY] the output normal vector computed
	 */
	void arbitraryNormal(AffVector& normal) const;

	/** Reset the (dx, dy, dz) components of this vector
	 *  @param dxx [INPUT ONLY] the new dx
	 *  @param dyy [INPUT ONLY] the new dy
	 *  @param dzz [INPUT ONLY] the new dz
	 */
	void assign(double dxx, double dyy, double dzz);

	/** Compute and return the cross product: "this" x rhs
	 *  @param rhs [INPUT ONLY] the second vector in the cross product
	 *  @return the computed cross product
	 */
	AffVector cross(const AffVector& rhs) const;

	/** Decompose an arbitrary vector into its components parallel to and
	 *  perpendicular to "this" vector.
	 *  @param arbitraryVector [INPUT ONLY] the vector to decompose
	 *  @param parallel [OUTPUT ONLY] the component of "arbitraryVector"
	 *                  parallel to "this" vector
	 *  @param perpendicular [OUTPUT ONLY] the component of "arbitraryVector"
	 *                       perpendicular to "this" vector
	 */
	void decompose(const AffVector& arbitraryVector,
					AffVector& parallel, AffVector& perpendicular) const;

	/** Compute and return the dot product of "this" vector with "rhs"
	 *  @param rhs [INPUT ONLY] the second vector in the dot product
	 *  @return the dot product of "this" vector and "rhs" vector
	 */
	double dot(const AffVector& rhs) const
				{ return dx*rhs.dx + dy*rhs.dy + dz*rhs.dz; }

	/** Compute and return the length of "this" vector */
	double length() const { return sqrt(lengthSquared()); }

	/** Compute and return the length squared of "this" vector */
	double lengthSquared() const { return dx*dx + dy*dy + dz*dz; }

	/** Determine the component of this vector with the largest absolute value
	 *  @param componentIndex [OUTPUT ONLY] the index (0, 1, or 2) of the
	 *                        component with the largest absolute value
	 *  @return the value of the component with the largest absolute value
	 */
	double maxAbsComponent(int& componentIndex) const;

	/** Determine the component of this vector with the smallest absolute value
	 *  @param componentIndex [OUTPUT ONLY] the index (0, 1, or 2) of the
	 *                        component with the smallest absolute value
	 *  @return the value of the component with the smallest absolute value
	 */
	double minAbsComponent(int& componentIndex) const;

	/** Normalize "this" vector
	 *  @return the length of the vector before it was normalized
	 */
	double normalize();

	/** Compute a unit vector in the direction of "this" vector, placing it in
	 *  "normalizedCopy". Leave "this" vector unchanged.
	 *  @param normalizedCopy [OUTPUT ONLY] the vector to hold the computed
	 *                        unit vector
	 *  @return the length of "this" vector"
	 */
	double normalizeToCopy(AffVector& normalizedCopy) const;

	/** Determine whether "this" vector is parallel to "v".
	 *  @param v [INPUT ONLY] the vector to be compared to "this" vector
	 *  @return true if the vectors are parallel; false otherwise
	 */
	bool parallelTo(const AffVector& v) const;

	/** Extract the components of "this" vector into an array of double,
	 *  starting at the given offset. Among other things, this method is useful
	 *  for interfacing with OpenGL, especially routines like glBufferData. For
	 *  example, a caller can allocate an array of an appropriate size, then
	 *  loop over an array of AffVector instances, invoking the vComponents
	 *  method as follows:<pre>
	    double* buf = new double[3*NUM_VECTORS];
	    for (int i=0 ; i<NUM_VECTORS ; i++)
	        affVectorArray[i].vComponents(buf, 3*i);
	    glBufferData(GL_ARRAY_BUFFER, 3*NUM_VECTORS*sizeof(double), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param components [OUTPUT ONLY] the array into which the (dx, dy, dz)
	 *                components of "this" vector are to be copied. Must be
	 *                at least (offset+3) positions long.
	 *  @param offset [INPUT ONLY] the starting location in components where the
	 *                components are to be placed (defaults to 0)
	 *  @return the given components array pointer so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array of
	 *          double
	 */
	double* vComponents(double* components, int offset=0) const;

	/** Extract the components of "this" vector into an array of "dvec3" (a
	 *  built-in GLSL data type essentially equivalent to "typedef double
	 *  dvec3[3];"), starting at the given offset. Among other things, this
	 *  method is useful for interfacing with OpenGL, especially routines like
	 *  glBufferData. For example, a caller can allocate an array of an
	 *  appropriate size, then loop over an array of AffVector instances,
	 *  invoking the vComponents method as follows:<pre>
	    dvec3* buf = new dvec3[3*NUM_VECTORS]; // assuming the "typedef" above
	    for (int i=0 ; i<NUM_VECTORS ; i++)
	        affVectorArray[i].vComponents(buf, i);
	    glBufferData(GL_ARRAY_BUFFER, NUM_VECTORS*sizeof(double), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param components [OUTPUT ONLY] the array into which the (dx, dy, dz)
	 *                components of "this" vector are to be copied. Must be
	 *                at least (offset+1) positions long.
	 *  @param offset [INPUT ONLY] the starting location in components where the
	 *                components are to be placed (defaults to 0)
	 *  @return the given components array pointer so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array of
	 *          double
	 */
	double* vComponents(double components[][3], int offset=0) const;

	/** Extract the components of "this" vector into an array of float,
	 *  starting at the given offset. Among other things, this method is useful
	 *  for interfacing with OpenGL, especially routines like glBufferData. For
	 *  example, a caller can allocate an array of an appropriate size, then
	 *  loop over an array of AffVector instances, invoking the vComponents
	 *  method as follows:<pre>
	    float* buf = new float[3*NUM_VECTORS];
	    for (int i=0 ; i<NUM_VECTORS ; i++)
	        affVectorArray[i].vComponents(buf, 3*i);
	    glBufferData(GL_ARRAY_BUFFER, 3*NUM_VECTORS*sizeof(float), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param components [OUTPUT ONLY] the array into which the (dx, dy, dz)
	 *                components of "this" vector are to be copied. Must be
	 *                at least (offset+3) positions long.
	 *  @param offset [INPUT ONLY] the starting location in components where the
	 *                components are to be placed (defaults to 0)
	 *  @return the given components array pointer so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array of
	 *          float
	 */
	float* vComponents(float* components, int offset=0) const;

	/** Extract the components of "this" vector into an array of "vec3" (a
	 *  built-in GLSL data type essentially equivalent to "typedef float
	 *  vec3[3];"), starting at the given offset. Among other things, this
	 *  method is useful for interfacing with OpenGL, especially routines like
	 *  glBufferData. For example, a caller can allocate an array of an
	 *  appropriate size, then loop over an array of AffVector instances,
	 *  invoking the vComponents method as follows:<pre>
	    vec3* buf = new vec3[3*NUM_VECTORS]; // assuming the "typedef" above
	    for (int i=0 ; i<NUM_VECTORS ; i++)
	        affVectorArray[i].vComponents(buf, i);
	    glBufferData(GL_ARRAY_BUFFER, NUM_VECTORS*sizeof(float), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param components [OUTPUT ONLY] the array into which the (dx, dy, dz)
	 *                components of "this" vector are to be copied. Must be
	 *                at least (offset+1) positions long.
	 *  @param offset [INPUT ONLY] the starting location in components where the
	 *                components are to be placed (defaults to 0)
	 *  @return the given components array pointer so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array of
	 *          float
	 */
	float* vComponents(float components[][3], int offset=0) const;

	/** Create a right-handed orthonormal coordinate system from the given
	 *  U and W vectors. If W is a zero vector, then U, V, and W will be set
	 *  to (1,0,0), (0,1,0), and (0,0,1), respectively. Otherwise, W is
	 *  normalized, and the component of U perpendicular to W is computed. If
	 *  that is zero (including the case that U itself is (0,0,0)), then an
	 *  arbitrary vector perpendicular to W is created, normalized, and assigned
	 *  to U. Finally, V &larr; W x U.
	 *  @param U [INPUT/OUTPUT] Used and modified as explained above.
	 *  @param V [OUTPUT ONLY] Computed as explained above.
	 *  @param W [INPUT/OUTPUT] Used and modified as explained above.
	 *  @see coordinateSystemFromVW
	 */
	static void coordinateSystemFromUW(AffVector& U, AffVector& V,
					AffVector& W);

	/** Create a right-handed orthonormal coordinate system from the given
	 *  V and W vectors. If W is a zero vector, then U, V, and W will be set
	 *  to (1,0,0), (0,1,0), and (0,0,1), respectively. Otherwise, W is
	 *  normalized, and the component of V perpendicular to W is computed. If
	 *  that is zero (including the case that V itself is (0,0,0)), then an
	 *  arbitrary vector perpendicular to W is created, normalized, and assigned
	 *  to V. Finally, U &larr; V x W.
	 *  @param U [OUTPUT ONLY] Computed as explained above.
	 *  @param V [INPUT/OUTPUT] Used and modified as explained above.
	 *  @param W [INPUT/OUTPUT] Used and modified as explained above.
	 *  @see coordinateSystemFromUW
	 */
	static void coordinateSystemFromVW(AffVector& U, AffVector& V,
					AffVector& W);

	/** Compute and return the cross product of the two given vectors
	 *  @param v1 [INPUT ONLY] The first vector of the cross product
	 *  @param v2 [INPUT ONLY] The second vector of the cross product
	 *  @return the cross product: v1 x v2
	 */
	static AffVector cross(const AffVector& v1, const AffVector& v2);

	/** Compute and return the dot product of the two given vectors
	 *  @param v1 [INPUT ONLY] The first vector of the dot product
	 *  @param v2 [INPUT ONLY] The second vector of the dot product
	 *  @return the dot product: v1 . v2
	 */
	static double dot(const AffVector& v1, const AffVector& v2)
					{ return v1.dx*v2.dx + v1.dy*v2.dy + v1.dz*v2.dz; }

	static const AffVector xu; /**< the vector (1,0,0) */
	static const AffVector yu; /**< the vector (0,1,0) */
	static const AffVector zu; /**< the vector (0,0,1) */
	static const AffVector zeroVector; /**< the vector (0,0,0) */

	// The vector components are public, but it is best to use the
	// other public methods whenever possible. Only use direct access
	// to these variables for ease of overwriting some specific
	// component.
	double	dx; /**< dx component */
	double	dy; /**< dy component */
	double	dz; /**< dz component */
};

std::ostream&	operator<<(std::ostream& os, const AffVector& v);
std::istream&	operator>>(std::istream& is, AffVector& v);

static AffVector operator*(double f, const AffVector& v)
	{ return AffVector(f*v[DX] , f*v[DY] , f*v[DZ]); }
}

#endif
