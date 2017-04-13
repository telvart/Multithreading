/** @file AffPoint.h
 *  Class definition for 3D points in an affine space.
 *  This is OPEN SOURCE software developed by James R. Miller (jrmiller@ku.edu)
 *  Original version: ca. 1996. See README_*.txt for more information.
 */

#ifndef AFFPOINT_H
#define AFFPOINT_H

#include <math.h>
#include <iostream>

#include "AffVector.h"

namespace cryph
{

// indices for extracting coordinates from points

const int X = 0;
const int Y = 1;
const int Z = 2;
const int W = 3;

class AffPoint
{
public:
	/** Default constructor creates a point at the origin.  */
	AffPoint();

	/** The copy constructor.
	 *  @param P [INPUT ONLY] the point whose coordinates are to initialize "this"
	 */
	AffPoint(const AffPoint& P);

	/** Construct a point from a vector (i.e., "this" point = origin + v)
	 *  @param v [INPUT ONLY] a vector
	 */
	AffPoint(const AffVector& v);

	/** Creates a point at the specified location.
	 *  @param xx [INPUT ONLY] the x coordinate
	 *  @param yy [INPUT ONLY] the y coordinate
	 *  @param zz [INPUT ONLY] the z coordinate (defaults to 0.0)
	 */
	AffPoint(double xx, double yy, double zz=0.0);

	/** Construct from a double array
	 *  @param P [INPUT ONLY] an array assumed to be of length 3 holding (x, y, z)
	 */
	AffPoint(const double P[]);

	/** Construct from a float array
	 *  @param P [INPUT ONLY] an array assumed to be of length 3 holding (x, y, z)
	 */
	AffPoint(const float P[]);

	/** The destructor */
	virtual ~AffPoint();

	/** Assigns the coordinates of rhs to "this" point
	 *  @param rhs [INPUT ONLY] a const reference to an AffPoint
	 *  @return the value assigned
	 */
	AffPoint operator=(const AffPoint& rhs);

	/** Adds the coordinates of rhs to "this" point
	 *  @param rhs [INPUT ONLY] a const reference to an AffVector
	 *  @return the result of incrementing the AffPoint
	 */
	AffPoint operator+=(const AffVector& rhs);

	/** Adds the coordinates of rhs to "this" point
	 *  @param rhs [INPUT ONLY] a const reference to an AffPoint
	 *  @return the result of incrementing the AffPoint
	 */
	AffPoint operator+=(const AffPoint& rhs);

	/** Subtracts the coordinates of rhs from "this" point
	 *  @param rhs [INPUT ONLY] a const reference to an AffVector
	 *  @return the result of decrementing the AffPoint
	 */
	AffPoint operator-=(const AffVector& rhs);

	/** Scales the coordinates of "this" point by f
	 *  @param f [INPUT ONLY] the scale factor
	 *  @return the value of the scaled AffPoint
	 */
	AffPoint operator*=(double f);

	/** Divides the coordinates of "this" point by f (i.e., scales by 1/f)
	 *  @param f [INPUT ONLY] the divisor
	 *  @return the value of the divided AffPoint
	 */
	AffPoint operator/=(double f);

	/** Returns x (if index = 0); y (if index is 1); z (if index is 2)
	 *  The returned value cannot be used as an l-value.
	 *  @param index [INPUT ONLY] the desired coordinate
	 *  @return the selected point coordinate
	 */
	double operator[](int index) const;

	/** Adds "this" point to p2 and returns the result as an AffPoint
	 *  @param p2 [INPUT ONLY] the point to add
	 *  @return the sum of "this" point and p2
	 */
	AffPoint operator+(const AffPoint& p2) const
				{ return AffPoint(x + p2.x, y + p2.y, z + p2.z); }

	/** Scales "this" point by f and returns the result as an AffPoint
	 *  @param f [INPUT ONLY] the scale factor
	 *  @return the scaled AffPoint
	 */
	AffPoint operator*(double f) const
				{ return AffPoint (f*x, f*y, f*z); }

	/** Divides "this" point by f (i.e., scales by 1/f) and returns the
	 *  result as an AffPoint
	 *  @param f [INPUT ONLY] the divisor
	 *  @return the AffPoint divided by f
	 */
	AffPoint operator/(double f) const
				{ return AffPoint (x/f, y/f, z/f); }

	/** Subtracts p2 from "this" point and returns the result as an AffVector
	 *  @param p2 [INPUT ONLY] the point to subtract
	 *  @return the AffVector resulting from subtracting p2 from "this" point
	 */
	AffVector operator-(const AffPoint& p2) const
				{ return AffVector(x-p2.x, y-p2.y, z-p2.z); }

	/** Adds the vector v2 to "this" point and returns the result as an AffPoint
	 *  @param v2 [INPUT ONLY] the vector to add
	 *  @return the AffPoint resulting from adding v2 to "this" point
	 */
	AffPoint operator+(const AffVector& v2) const
				{ return AffPoint(x+v2[DX], y+v2[DY], z+v2[DZ]); }

	/** Subtracts the vector v2 from "this" point and returns the result
	 *  as an AffPoint
	 *  @param v2 [INPUT ONLY] the vector to subtract
	 *  @return the AffPoint resulting from subtracting v2 from "this" point
	 */
	AffPoint operator-(const AffVector& v2) const
				{ return AffPoint(x-v2[DX], y-v2[DY], z-v2[DZ]); }

	/** Extract the coordinates of "this" point into an array of double,
	 *  starting at the given offset. Among other things, this method is useful
	 *  for interfacing with OpenGL, especially routines like glBufferData. For
	 *  example, a caller can allocate an array of an appropriate size, then
	 *  loop over an array of AffPoint instances, invoking the aCoords method
	 *  as follows:<pre>
	    double* buf = new double[3*NUM_POINTS];
	    for (int i=0 ; i<NUM_POINTS ; i++)
	        affPointArray[i].aCoords(buf, 3*i);
	    glBufferData(GL_ARRAY_BUFFER, 3*NUM_POINTS*sizeof(double), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param coords [OUTPUT ONLY] the array into which the (x, y, z)
	 *                coordinates of "this" point are to be copied. Must be
	 *                at least (offset+3) positions long.
	 *  @param offset [INPUT ONLY] the starting location in coords where the
	 *                coordinates are to be placed (defaults to 0)
	 *  @return the given coords array pointer so that the method call can be used
	 *          as a parameter to an OpenGL function expecting an array of
	 *          double
	 *  @see pCoords
	 */
	double* aCoords(double* coords, int offset=0) const;

	/** Extract the coordinates of "this" point into an array of "dvec3" (a
	 *  built-in GLSL data type essentially equivalent to "typedef double
	 *  dvec[3];"), starting at the given offset. Among other things, this
	 *  method is useful for interfacing with OpenGL, especially routines like
	 *  glBufferData. For example, a caller can allocate an array of an
	 *  appropriate size, then loop over an array of AffPoint instances,
	 *  invoking the aCoords method as follows:<pre>
	    dvec3* buf = new dvec3[NUM_POINTS]; // assuming the "typedef" above
	    for (int i=0 ; i<NUM_POINTS ; i++)
	        affPointArray[i].aCoords(buf, i);
	    glBufferData(GL_ARRAY_BUFFER, NUM_POINTS*sizeof(dvec3), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param coords [OUTPUT ONLY] the array into which the (x, y, z)
	 *                coordinates of "this" point are to be copied. Must be
	 *                at least (offset+1) positions long.
	 *  @param offset [INPUT ONLY] the starting location in coords where the
	 *                coordinates are to be placed (defaults to 0)
	 *  @return the given coords array pointer so that the method call can be used
	 *          as a parameter to an OpenGL function expecting an array of
	 *          double
	 *  @see pCoords
	 */
	double* aCoords(double coords[][3], int offset=0) const;

	/** Extract the coordinates of "this" point into an array of float,
	 *  starting at the given offset. Among other things, this method is useful
	 *  for interfacing with OpenGL, especially routines like glBufferData. For
	 *  example, a caller can allocate an array of an appropriate size, then
	 *  loop over an array of AffPoint instances, invoking the aCoords method
	 *  as follows:<pre>
	    float* buf = new float[3*NUM_POINTS];
	    for (int i=0 ; i<NUM_POINTS ; i++)
	        affPointArray[i].aCoords(buf, 3*i);
	    glBufferData(GL_ARRAY_BUFFER, 3*NUM_POINTS*sizeof(float), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param coords [OUTPUT ONLY] the array into which the (x, y, z)
	 *                coordinates of "this" point are to be copied. Must be
	 *                at least (offset+3) positions long.
	 *  @param offset [INPUT ONLY] the starting location in coords where the
	 *                coordinates are to be placed (defaults to 0)
	 *  @return the given coords array pointer so that the method call can be used
	 *          as a parameter to an OpenGL function expecting an array of float
	 *  @see pCoords
	 */
	float* aCoords(float* coords, int offset=0) const;

	/** Extract the coordinates of "this" point into an array of "vec3" (a
	 *  built-in GLSL data type essentially equivalent to
	 *  "typedef float vec[3];"), starting at the given offset. Among other
	 *  things, this method is useful for interfacing with OpenGL, especially
	 *  routines like glBufferData. For example, a caller can allocate an array
	 *  of an appropriate size, then loop over an array of AffPoint instances,
	 *  invoking the aCoords method as follows:<pre>
	    vec3* buf = new vec3[NUM_POINTS]; // assuming the "typedef" above
	    for (int i=0 ; i<NUM_POINTS ; i++)
	        affPointArray[i].aCoords(buf, i);
	    glBufferData(GL_ARRAY_BUFFER, NUM_POINTS*sizeof(vec3), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param coords [OUTPUT ONLY] the array into which the (x, y, z)
	 *                coordinates of "this" point are to be copied. Must be
	 *                at least (offset+1) positions long.
	 *  @param offset [INPUT ONLY] the starting location in coords where the
	 *                coordinates are to be placed (defaults to 0)
	 *  @return the given coords array pointer so that the method call can be used
	 *          as a parameter to an OpenGL function expecting an array of float
	 *  @see pCoords
	 */
	float* aCoords(float coords[][3], int offset=0) const;

	/** Reset the (x, y, z) coordinates of "this" point
	 *  @param xx [INPUT ONLY] the new x coordinate
	 *  @param yy [INPUT ONLY] the new y coordinate
	 *  @param zz [INPUT ONLY] the new z coordinate
	 */
	void assign(double xx, double yy, double zz);

	/** Compute the Barycentric coordinates (b1, b2, b3) of "this" point with
	 *  respect to the three given reference points, P1, P2, and P3. We assume
	 *  "this" point lies in the plane determined by the points P1, P2, and P3.
	 *  On return, b1, b2, and b3 will have been computed such that:
	 *  (i) b1 + b2 + b3 = 1, (ii) "this" point = b1*P1 + b2*P2 + b3*P3, and
	 *  (iii) none of "this" point, P1, P2, or P3 will have been changed.
	 *  Furthermore, if "this" point lies in the triangle P1-P2-P3, then
	 *  b1>=0, b2>=0, and b3>=0.
	 *  @param P1 [INPUT ONLY] the first reference point
	 *  @param P2 [INPUT ONLY] the second reference point
	 *  @param P3 [INPUT ONLY] the third reference point
	 *  @param b1 [OUTPUT ONLY] b1 the returned Barycentric coordinate (associated with P1)
	 *  @param b2 [OUTPUT ONLY] b2 the returned Barycentric coordinate (associated with P2)
	 *  @param b3 [OUTPUT ONLY] b3 the returned Barycentric coordinate (associated with P3)
	 *  @see fromBarycentricCoords
	 */
	void barycentricCoords( // in a plane
				// find the areal Barycentric coordinates
				// of "this" point with respect to:
				const AffPoint& P1, const AffPoint& P2, const AffPoint& P3,
				// returning them in:  (b1+b2+b3 = 1)
				double& b1, double& b2, double& b3) const;

	/** Compute the Barycentric coordinates (b1, b2) of "this" point with
	 *  respect to the two given reference points, P1 and P2. We assume
	 *  "this" point lies on the line determined by the points P1 and P2.
	 *  On return, b1 and b2 will have been computed such that:
	 *  (i) b1 + b2 = 1, (ii) "this" point = b1*P1 + b2*P2, and
	 *  (iii) none of "this" point, P1, or P2 will have been changed.
	 *  Furthermore, if ("this" point lies between P1 and P2 on the line, then
	 *  b1>=0 and b2>=0.
	 *  @param P1 [INPUT ONLY] the first reference point
	 *  @param P2 [INPUT ONLY] the second reference point
	 *  @param b1 [OUTPUT ONLY] the returned Barycentric coordinate (associated with P1)
	 *  @param b2 [OUTPUT ONLY] the returned Barycentric coordinate (associated with P2)
	 *  @see fromBarycentricCoords
	 */
	void barycentricCoords( // on a line
				// find the areal Barycentric coordinates of "this"
				// point with respect to:
				const AffPoint& P1, const AffPoint& P2,
				// returning them in:  (b1+b2 = 1)
				double& b1, double& b2) const;

	/** Determine whether "this" point is coincident with P within the current
	 *  coincidence tolerance
	 *  @param P [INPUT ONLY] the point to be compared with "this" point
	 *  @return true if and only if "this" point is coincident with P within
	 *          the current coincidence tolerance.
	 *  @see getCoincidenceTolerance
	 *  @see setCoincidenceTolerance
	 */
	bool coincidentWith(const AffPoint& p) const;

	/** Compute and return the distance from "this" point to the given line
	 *  @param B [INPUT ONLY] a point on the line
	 *  @param u [INPUT ONLY] a vector along the line
	 *  @return the distance from "this" point to the line(B, u)
	 */
	double distanceFromLine(const AffPoint& B, const AffVector& u) const;

	/** Compute the distance from "this" point to the origin
	 *  @return the distance
	 */
	double distanceFromOrigin() const;

	/** Compute and return the distance squared from "this" point to the given
	 *  line
	 *  @param B [INPUT ONLY] a point on the line
	 *  @param u [INPUT ONLY] a vector along the line
	 *  @return the square of the distance from "this" point to the line(B, u)
	 */
	double distanceSquaredFromLine(const AffPoint& B, const AffVector& u) const;

	/** Compute the square of the distance from "this" point to the origin
	 *  @return the square of the distance from "this" point to the origin
	 */
	double distanceSquaredFromOrigin() const;

	/** Compute the square of the distance from "this" point to the given point
	 *  @param P [INPUT ONLY] the point whose distance squared from "this" point we seek
	 *  @return the square of the distance from "this" point to the given point
	 */
	double distanceSquaredTo(const AffPoint& p) const;

	/** Compute the distance from "this" point to the given point
	 *  @param P [INPUT ONLY] the point whose distance from "this" point we seek
	 *  @return the distance from "this" point to the given point
	 */
	double distanceTo(const AffPoint& p) const;

	/** Move "this" point along the line through the origin until it lies
	 *  on the unit sphere centered at the origin.
	 */
	double normalize();

	/** Extract the coordinates of "this" point into an array of double,
	 *  starting at the given offset. While doing so, embed each point in
	 *  projective space using the given w coordinate. Among other things, this
	 *  method is useful for interfacing with OpenGL, especially routines like
	 *  glBufferData. For example, a caller can allocate an array of an
	 *  appropriate size, then loop over an array of AffPoint instances,
	 *  invoking the pCoords method as follows:<pre>
	    double* buf = new double[4*NUM_POINTS];
	    for (int i=0 ; i<NUM_POINTS ; i++)
	        affPointArray[i].pCoords(buf, 4*i);
	    glBufferData(GL_ARRAY_BUFFER, 4*NUM_POINTS*sizeof(double), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param coords [OUTPUT ONLY] the array into which the (x, y, z, w)
	 *                projective space coordinates of "this" point are to be
	 *                copied. Must be at least (offset+4) positions long.
	 *  @param offset [INPUT ONLY] the starting location in coords where the
	 *                coordinates are to be placed (defaults to 0)
	 *  @return the given coords array pointer so that the method call can be used
	 *          as a parameter to an OpenGL function expecting an array of
	 *          double
	 *  @see aCoords
	 */
	double* pCoords(double* coords, double w, int offset=0) const;

	/** Extract the coordinates of "this" point into an array of float,
	 *  starting at the given offset. While doing so, embed each point in
	 *  projective space using the given w coordinate. Among other things, this
	 *  method is useful for interfacing with OpenGL, especially routines like
	 *  glBufferData. For example, a caller can allocate an array of an
	 *  appropriate size, then loop over an array of AffPoint instances,
	 *  invoking the pCoords method as follows:<pre>
	    float* buf = new float[4*NUM_POINTS];
	    for (int i=0 ; i<NUM_POINTS ; i++)
	        affPointArray[i].pCoords(buf, 4*i);
	    glBufferData(GL_ARRAY_BUFFER, 4*NUM_POINTS*sizeof(float), buf, GL_STATIC_DRAW);
	    delete [] buf;</pre>
	 *  @param coords [OUTPUT ONLY] the array into which the (x, y, z, w)
	 *                projective space coordinates of "this" point are to be
	 *                copied. Must be at least (offset+4) positions long.
	 *  @param offset [INPUT ONLY] the starting location in coords where the
	 *                coordinates are to be placed (defaults to 0)
	 *  @return the given coords array pointer so that the method call can be used
	 *          as a parameter to an OpenGL function expecting an array of float
	 *  @see aCoords
	 */
	float* pCoords(float* coords, float w, int offset=0) const;

	/** A method inspired by OpenGL's GLSL's swizzle operation. If the i-th
	 *  character of xyz is 'x', then the x coordinate of "this" point is
	 *  placed into the i-th component of "this" point. If instead it is 'X',
	 *  then negative of the x coordinate of "this" point is placed into the
	 *  i-th component of "this" point. Characters 'y', 'Y', 'z', 'Z' behave
	 *  similarly. Any other character is simply ignored. Some examples:<pre>
	 cryph::AffPoint ap(1.0, 2.0, 3.0);
	 ap.swizzle("Zxy"); // ap is now (-3.0, 1.0, 2.0)
	 ap.swizzle("dzy"); // ap is now (-3.0, 2.0, 1.0)
	 ap.swizzle("xxy"); // ap is now (-3.0, -3.0, 2.0)</pre>
	 * @param xyz [INPUT ONLY] an array of char of length 3 interpreted as
	 *            explained and illustrated above
	 */
	void swizzle(char xyz[3]);

	/** Compute and return the cylindrical coordinates of "this" point
	 *  @param r [OUTPUT ONLY] the output computed r coordinate
	 *  @param theta [OUTPUT ONLY] the output computed theta coordinate
	 *  @param z [OUTPUT ONLY] the output computed z coordinate (same is this->z).
	 *  @see fromCylindrical
	 */
	void toCylindrical(double& r, double& theta, double& z) const;

	/** Compute and return the spherical coordinates of "this" point.
	 *  @param rho [OUTPUT ONLY] the output computed distance from the origin to "this" point
	 *  @param theta [OUTPUT ONLY] the output computed angle in the xy-plane from the positive
	 *               x-axis to the projection of the vector ("this" - origin)
	 *               onto the xy-plane. Value will be: -PI <= theta <= PI.
	 *  @param phi [OUTPUT ONLY] the output computed angle from the positive z-axis to the
	 *             vector ("this" - origin). Value will be: 0 <= phi <= PI.
	 *  @see fromSpherical
 	 */
	void toSpherical(double& rho, double& theta, double& phi) const;

	/** Compute and return the centroid of the given array of points
	 *  @param p [INPUT ONLY] an array of AffPoint instances
	 *  @param nPoints [INPUT ONLY] the number of AffPoint instances in the array, p
	 *  @return the computed centroid
	 */
	static AffPoint centroid(const AffPoint p[], int nPoints);

	/** Compute and return the point whose barycentric coordinates are
	 *  (b1, b2, b3) with respect to the reference points P1, P2, and P3
	 *  @param P1 [INPUT ONLY] the first input reference point
	 *  @param P2 [INPUT ONLY] the second input reference point
	 *  @param P3 [INPUT ONLY] the third input reference point
	 *  @param b1 [INPUT ONLY] the Barycentric coordinate for P1
	 *  @param b2 [INPUT ONLY] the Barycentric coordinate for P2
	 *  @param b3 [INPUT ONLY] the Barycentric coordinate for P3
	 *  @return the computed point
	 *  @see barycentricCoords
	 */
	static AffPoint fromBarycentricCoords( // in a plane
							const AffPoint& P1, const AffPoint& P2,
							const AffPoint& P3,
							double b1, double b2, double b3)
						{ return b1*P1 + b2*P2 + b3*P3; }

	/** Compute and return the point whose barycentric coordinates are
	 *  (b1, b2) with respect to the reference points P1 and P2
	 *  @param P1 [INPUT ONLY] the first input reference point
	 *  @param P2 [INPUT ONLY] the second input reference point
	 *  @param b1 [INPUT ONLY] the Barycentric coordinate for P1
	 *  @param b2 [INPUT ONLY] the Barycentric coordinate for P2
	 *  @return the computed point
	 *  @see barycentricCoords
	 */
	static AffPoint fromBarycentricCoords( // on a line
							const AffPoint& P1, const AffPoint& P2,
							double b1, double b2)
						{ return b1*P1 + b2*P2; }

	/** Compute and return the point with the given cylindrical coordinates
	 *  @param r [INPUT ONLY] the radial cylindrical coordinate
	 *  @param theta [INPUT ONLY] the theta cylindrical coordinate
	 *  @param z [INPUT ONLY] the z coordinate
	 *  @return the computed point
	 *  @see toCylindrical
	 */
	static AffPoint fromCylindrical(double r, double theta, double z)
						{ return AffPoint( r*cos(theta), r*sin(theta), z ); }

	/** Compute and return the point with the given spherical coordinates
	 *  @param rho [INPUT ONLY] the distance from the origin to the desired point
	 *  @param theta [INPUT ONLY] the desired theta
	 *  @param phi [INPUT ONLY] the desired phi
	 *  @see toSpherical
 	 */
	static AffPoint fromSpherical(double rho, double theta, double phi)
						{ return AffPoint(rho*sin(phi)*cos(theta),
										  rho*sin(phi)*sin(theta), rho*cos(phi) ); }

	/** Get the current point coincidence tolerance.
	 *  @return the current tolerance
	 *  @see setCoincidenceTolerance
	 *  @see coincidentWith
	 */
	static double getCoincidenceTolerance() { return AffPoint::sCoincidenceTol; }
	/** Find the point in the given buffer of points that is farthest away
	 *  away in the given direction. If that point is 'P', the function return
	 *  value is "(P-ref).dir". If there is exactly one such farthest point,
	 *  then the indices index1 and index2 will both contain the index of 'P'
	 *  in 'buf'. If several points tie, then index1 and index2 are the indices
	 *  of the first succession of adjacent points all at that offset. If there
	 *  are multiple sets of adjacent tieing points, only the indices of the
	 *  first are returned.
	 *  @param ref [INPUT ONLY] a reference point used to compute distances
	 *  @param dir [INPUT ONLY] a vector which, along with 'ref', define a
	 *             plane away from which distances are measured.
	 *  @param buf [INPUT ONLY] the array of points to be analyzed
	 *  @param bufSize [INPUT ONLY] the number of points in 'buf'
	 *  @param index1 [OUTPUT ONLY] as described above
	 *  @param index2 [OUTPUT ONLY] as described above
	 */
	static double maxOffsetInDirection(
					const AffPoint& ref, const AffVector& dir,
					const AffPoint buf[], int bufSize,
					int& index1, int& index2);

	/** Compute the unit vector uHat from (c-a). Then compute and return
	 *  ratio(a,b,c) as defined by Farin.
	 *  @param a [INPUT ONLY] first point
	 *  @param b [INPUT ONLY] middle point
	 *  @param c [INPUT ONLY] last point
	 *  @return fabs(uHat.(c-b))>tol ? uHat.(b-a) / uHat.(c-b) : 0.0
	 */
	static double ratio(const AffPoint& a, const AffPoint& b, const AffPoint& c);
	/** Set the current point coincidence tolerance.
	 *  @param tol [INPUT ONLY] the desired point coincidence tolerance
	 *  @see getCoincidenceTolerance
	 *  @see coincidentWith
	 */
	static void setCoincidenceTolerance(double tol);

    // ---------- Global constants

	// Special Points
	static const AffPoint		origin; /**< the point (0,0,0) */
	static const AffPoint		xAxisPoint; /**< the point (1,0,0) */
	static const AffPoint		yAxisPoint; /**< the point (0,1,0) */
	static const AffPoint		zAxisPoint; /**< the point (0,0,1) */

	// The coordinates are public, but it is usually best to use
	// the public methods, using public access to the following only
	// for convenient overwriting of the coordinates of a specific point.
	double	x; /**< x coordinate */
	double	y; /**< y coordinate */
	double	z; /**< z coordinate */

private:
	static double	sCoincidenceTol;
};

std::ostream&		operator<<(std::ostream& os, const AffPoint& p);
std::istream&		operator>>(std::istream& is, AffPoint& p);

static AffPoint operator*(double f, const AffPoint& p)
	{ return AffPoint(f*p[X], f*p[Y], f*p[Z]); }
}

#endif
