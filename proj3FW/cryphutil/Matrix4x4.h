/** @file Matrix4x4.h
 *  Class definition for creating, manipulating, and using 4x4 matrices
 *  @see Matrix3x3
 *  This is OPEN SOURCE software developed by James R. Miller (jrmiller@ku.edu)
 *  Original version: ca. 1996. See README_*.txt for more information.
 */

#ifndef MATRIX4x4_H
#define MATRIX4x4_H

#include <iostream>

#include "AffPoint.h"
#include "AffVector.h"
#include "Matrix3x3.h"

namespace cryph
{

class ProjPoint;
class ViewVolume;

class Matrix4x4
{
public:
	/** Default constructor creates a 4x4 identity matrix */
	Matrix4x4();

	/** The copy constructor
	 *  @param m the matrix whose elements are copied
	 */
	Matrix4x4(const Matrix4x4& m);

	/** Construct a 4x4 matrix directly from 16 values given in row-major order
	 *  @param mrc The element for row r, column c
	 */
	Matrix4x4(
		double m11, double m12, double m13, double m14,
		double m21, double m22, double m23, double m24,
		double m31, double m32, double m33, double m34,
		double m41, double m42, double m43, double m44);

	/** Construct a 4x4 transformation matrix from a 3x3 matrix that is
	 *  assumed to encode a 2D transformation in projective space.
	 *  The implementation slides the third row and third column over
	 *  to the fourth, and inserts a third row and third column from
	 *  a 4x4 Identity matrix.
	 *  @param M a 3x3 matrix that describes a 2D xform in projective space.
	 *           That is, it is assumed to do (x', y', w')^T = M*(x, y, 1)^T.
	 */
	Matrix4x4(const Matrix3x3& M);

	/** Construct a 4x4 matrix from an Affine transformation: (M, t),
	 *  where (x', y', z')^T = M*(x, y, z)^T + t.
	 *  The implementation places M in the upper left 3x3 submatrix of the
	 *  4x4 matrix, it places t in the first three rows of the fourth column,
	 *  and it assigns (0,0,0,1) to the 4th row.
	 *  @param M a 3x3 matrix encoding a linear transformation on 3D vectors
	 *  @param t a 3D vector
	 *  @see extractAffineMt
	 */
	Matrix4x4(const Matrix3x3& M, const AffVector& t);

	/** Construct a 4x4 matrix from a 3x3 matrix, M (encoding a linear
	 *  transformation on 3D vectors) and a fixed point. It first constructs
	 *  the translation vector, t, such that: M*FixedPoint + t = FixedPoint.
	 *  It then creates the matrix as if the constructor with prototype
	 *  Matrix4x4(Matrix3x3, AffVector) were called.
	 *  @param M a 3x3 matrix encoding a linear transformation on 3D vectors
	 *  @param FixedPoint a 3D affine point unaffected by the affine transformation
	 */
	Matrix4x4(const Matrix3x3& M, const AffPoint& FixedPoint);

	// From an affine matrix M and a point whose pre- and post-image are known
	/** Construct a 4x4 matrix from a 3x3 matrix, M (encoding a linear
	 *  transformation on 3D vectors) and a pair of points, the latter being
	 *  the point to which the former is mapped by the transformation to be
	 *  encoded in this newly constructed matrix. It first constructs
	 *  the translation vector, t, such that: M*PreImage + t = PostImage.
	 *  It then creates the matrix as if the constructor with prototype
	 *  Matrix4x4(Matrix3x3, AffVector) were called.
	 *  @param M a 3x3 matrix encoding a linear transformation on 3D vectors
	 *  @param PreImage a 3D affine point
	 *  @param PostImage the 3D affine point to which "PreImage" is mapped
	 */
	Matrix4x4(const Matrix3x3& M,
			const AffPoint& PreImage, const AffPoint& PostImage);

	/** The destructor */
	virtual ~Matrix4x4();

	/** Factory method to create a 4x4 matrix from a single precision array
	 *  of 16 elements assumed to hold the values of the matrix in column-major
	 *  order.
	 *  @param m an array of float of length 16
	 *  @return the constructed 4x4 matrix
	 *  @see extractColMajor
	 */
	static Matrix4x4 fromColMajor(const float* m);

	/** Factory method to create a 4x4 matrix from a double precision array
	 *  of 16 elements assumed to hold the values of the matrix in column-major
	 *  order.
	 *  @param m an array of double of length 16
	 *  @return the constructed 4x4 matrix
	 *  @see extractColMajor
	 */
	static Matrix4x4 fromColMajor(const double* m);

	/** Factory method to create a 4x4 matrix from a single precision array
	 *  of 16 elements assumed to hold the values of the matrix in row-major
	 *  order.
	 *  @param m an array of float of length 16
	 *  @see extractRowMajor
	 */
	static Matrix4x4 fromRowMajor(const float* m);

	/** Factory method to create a 4x4 matrix from a double precision array
	 *  of 16 elements assumed to hold the values of the matrix in row-major
	 *  order.
	 *  @param m an array of double of length 16
	 *  @return the constructed 4x4 matrix
	 *  @see extractRowMajor
	 */
	static Matrix4x4 fromRowMajor(const double* m);

	/** Factory method to create a 4x4 matrix that performs a rotation about
	 *  a general axis in 3D space.
	 *  @param B a 3D point on the desired rotation axis
	 *  @param axis a 3D vector along the rotation axis
	 *  @param angle the desired rotation angle in degrees
	 *  @return the constructed 4x4 matrix
	 *  @see extractAxisAngle
	 */
	static Matrix4x4 generalRotationDegrees(const AffPoint& B,
		const AffVector& axis, double angle);

	/** Factory method to create a 4x4 matrix that performs a rotation about
	 *  a general axis in 3D space.
	 *  @param B a 3D point on the desired rotation axis
	 *  @param axis a 3D vector along the rotation axis
	 *  @param angle the desired rotation angle in radians
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 generalRotationRadians(const AffPoint& B,
		const AffVector& axis, double angle);

	/** Factory method to create a 4x4 matrix that performs a rotation about
	 *  the x-axis.
	 *  @param angle the desired rotation angle in degrees
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 xRotationDegrees(double angle);

	/** Factory method to create a 4x4 matrix that performs a rotation about
	 *  the y-axis.
	 *  @param angle the desired rotation angle in degrees
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 yRotationDegrees(double angle);

	/** Factory method to create a 4x4 matrix that performs a rotation about
	 *  the z-axis.
	 *  @param angle the desired rotation angle in degrees
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 zRotationDegrees(double angle);

	/** Factory method to create a 4x4 matrix that performs a rotation about
	 *  the x-axis.
	 *  @param angle the desired rotation angle in radians
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 xRotationRadians(double angle);

	/** Factory method to create a 4x4 matrix that performs a rotation about
	 *  the y-axis.
	 *  @param angle the desired rotation angle in radians
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 yRotationRadians(double angle);

	/** Factory method to create a 4x4 matrix that performs a rotation about
	 *  the z-axis.
	 *  @param angle the desired rotation angle in radians
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 zRotationRadians(double angle);

	/** Factory method to create a 4x4 matrix that performs a translation
	 *  @param trans the desired translation vector
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 translation(const AffVector& trans);

	/** Factory method to create a 4x4 matrix that performs a non-uniform
	 *  scale transformation.
	 *  @param sx the scale factor along the x-axis
	 *  @param sy the scale factor along the y-axis
	 *  @param sz the scale factor along the z-axis
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 scale(double sx, double sy, double sz);

	/** Factory method to create a 4x4 matrix that maps model coordinates
	 *  into an eye coordinate system in which the x-axis is oriented left to
	 *  right in the field of view; the y-axis is oriented bottom to top, and
	 *  the z-axis is oriented from the scene back towards the eye.
	 *  @param eye the (x,y,z) model coordinate position of the viewer
	 *  @param center the (x,y,z) model coordinate position of the "center of
	 *                attention", typically a point in front of the eye near
	 *                the center of the scene to be viewed.
	 *  @param up a 3D vector to fix the final rotational degree of freedom. The
	 *            component of the "up" vector perpendicular to the line of
	 *            sight (the vector from "center" towards "eye") will become
	 *            the y-axis of the eye coordinate system.
	 *  @return the constructed 4x4 matrix
	 *  @see getECvw
	 */
	static Matrix4x4 lookAt(const cryph::AffPoint& eye,
							const cryph::AffPoint& center,
							const cryph::AffVector& up);

	/** Factory method to create a 4x4 matrix that performs an orthogonal
	 *  projection from an eye coordinate system into a logical device
	 *  space (LDS) in which the x, y, and z limits are -1 .. +1. This will
	 *  produce a left-handed LDS system as expected by OpenGL.
	 *  @param xmin the minimum x eye coordinate of the desired field of view
	 *  @param xmax the maximum x eye coordinate of the desired field of view
	 *  @param ymin the minimum y eye coordinate of the desired field of view
	 *  @param ymax the maximum y eye coordinate of the desired field of view
	 *  @param zmin the minimum z eye coordinate of the desired field of view
	 *  @param zmax the maximum z eye coordinate of the desired field of view
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 orthogonal(double xmin, double xmax, double ymin,
				double ymax, double zmin, double zmax);

	/** Factory method to create a 4x4 matrix that performs an oblique
	 *  projection from an eye coordinate system into a logical device
	 *  space (LDS) in which the x, y, and z limits are -1 .. +1. This will
	 *  incorporate a shear transformation that will produce an orthogonal
	 *  left-handed LDS system as expected by OpenGL.
	 *  @param zpp the eye coordinate z value that specifies the location of
	 *             the projection plane
	 *  @param xmin the minimum x eye coordinate of the desired field of view
	 *              defined on the projection plane
	 *  @param xmax the maximum x eye coordinate of the desired field of view
	 *              defined on the projection plane
	 *  @param ymin the minimum y eye coordinate of the desired field of view
	 *              defined on the projection plane
	 *  @param ymax the maximum y eye coordinate of the desired field of view
	 *              defined on the projection plane
	 *  @param zmin the minimum z eye coordinate of the desired field of view
	 *  @param zmax the maximum z eye coordinate of the desired field of view
	 *  @param projDir a vector defined in eye coordinates that specifies the
	 *                 common direction of projection. projDir.dz cannot be zero.
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 oblique(double zpp, double xmin, double xmax, double ymin,
				double ymax, double zmin, double zmax,
				const cryph::AffVector& projDir);

	/** Factory method to create a 4x4 matrix that performs a perspective
	 *  projection from an eye coordinate system into a logical device
	 *  space (LDS) in which the x, y, and z limits are -1 .. +1. This will
	 *  produce an orthogonal left-handed LDS system as expected by OpenGL.
	 *  @param zpp the eye coordinate z value that specifies the location of
	 *             the projection plane
	 *  @param xmin the minimum x eye coordinate of the desired field of view
	 *              defined on the projection plane
	 *  @param xmax the maximum x eye coordinate of the desired field of view
	 *              defined on the projection plane
	 *  @param ymin the minimum y eye coordinate of the desired field of view
	 *              defined on the projection plane
	 *  @param ymax the maximum y eye coordinate of the desired field of view
	 *              defined on the projection plane
	 *  @param zmin the minimum z eye coordinate of the desired field of view (must be negative)
	 *  @param zmax the maximum z eye coordinate of the desired field of view (must be negative AND strictly greater than zmin)
	 *  @param h the signed half-distance between the left and right eye, if generating a matrix for
	 *           a stereo display. If (h<0), this is for the left eye; if (h>0) this is for the
	 *           right eye. To generate a single perspective matrix for a monoscopic view, use the default h==0.
	 *           <p><b>NOTE</b>: if a stereo pair is being created by generating two different MC
	 *           to EC matrices, then a common perspective matrix should be used for both
	 *           the left and right eye cumulative matrices, and <b>h should be 0</b>.
	 *  @return the constructed 4x4 matrix
	 */
	static Matrix4x4 perspective(double zpp, double xmin, double xmax,
			double ymin, double ymax, double zmin, double zmax, double h=0.0);

	/** Determine whether a given (eye, center, up) specification uniquely
	 *  specifies a MC to EC transformation. This method is primarily designed
	 *  for internal use (i.e., as a part of the implementation of "lookAt"),
	 *  but it is potentially useful externally as well in order to ensure
	 *  that a given set of viewing parameters is valid.
	 *  @param eye the (x,y,z) model coordinate position of the viewer
	 *  @param center the (x,y,z) model coordinate position of the "center of
	 *                attention", typically a point in front of the eye near
	 *                the center of the scene to be viewed.
	 *  @param up a 3D vector to fix the final rotational degree of freedom. The
	 *            component of the "up" vector perpendicular to the line of
	 *            sight (the vector from "center" towards "eye") will become
	 *            the y-axis of the eye coordinate system.
	 *  @param v [OUTPUT ONLY] the unit vector as measured in MC of the EC y-axis
	 *  @param w [OUTPUT ONLY] the unit vector as measured in MC of the EC z-axis
	 *  @return true if and only if the (eye, center, up) parameters are valid
	 *               and the output "v" and "w" parameters have been computed.
	 *  @see lookAt
	 */
	static bool getECvw(const cryph::AffPoint& eye,
			const cryph::AffPoint& center, const cryph::AffVector& up,
			cryph::AffVector& v, cryph::AffVector& w);

	/** Assigns the matrix "rhs" to "this" matrix
	 *  @param rhs the matrix whose values are to be copied
	 *  @return the value assigned
	 */
	Matrix4x4 operator=(const Matrix4x4& rhs);

	/** Assign this = this*rhs
	 *  @param rhs the 4x4 matrix to be multiplied on the right of "this"
	 *  @return the updated matrix value
	 */
	Matrix4x4 operator*=(const Matrix4x4& rhs);

	/** Assign this = f*this. (That is, scale all elements of "this" matrix
	 *  by "f".)
	 *  @param f the scale factor
	 *  @return the updated matrix value
	 */
	Matrix4x4 operator*=(double f);

	/** Assign this = this + rhs
	 *  @param rhs the 4x4 matrix whose elements are to be added to "this"
	 *  @return the updated matrix value
	 */
	Matrix4x4 operator+=(const Matrix4x4& rhs);

	/** Multiply an affine point by "this" matrix: p' = this * p. The point
	 *  will be transformed using only the upper left 3x3 submatrix, M, and
	 *  the translation vector, t, taken from the first three rows of the
	 *  fourth column: p' = M*p + t.
	 *  @param p the 3D affine point to be transformed
	 *  @return the transformed affine point
	 */
	AffPoint operator*(const AffPoint& p) const;

	/** Multiply a point in projective space by "this" matrix: p' = this * p.
	 *  @param p the 4D projective space point to be transformed
	 *  @return the transformed projective space point
	 */
	ProjPoint operator*(const ProjPoint& p) const;

	/** Multiply a 3D vector by "this" matrix: v' = this * v. The vector
	 *  will be transformed using only the upper left 3x3 submatrix, M.
	 *  v' = M*v.
	 *  @param v the 3D vector to be transformed
	 *  @return the transformed vector
	 */
	AffVector operator*(const AffVector& v) const;

	/** Multiply "this" matrix on the right by "m2"
	 *  @param m2 the matrix to be multiplied by "this" matrix
	 *  @return the product: this*m2
	 */
	Matrix4x4 operator*(const Matrix4x4& m2) const;

	/** Add "this" matrix to matrix "m2"
	 *  @param m2 the matrix to be added to "this" matrix
	 *  @return the sum: this + m2
	 */
	Matrix4x4 operator+(const Matrix4x4& m2) const;

	/** Subtract m2 from "this" matrix
	 *  @param m2 the matrix to be subtracted from "this" matrix
	 *  @return the difference: this - m2
	 */
	Matrix4x4 operator-(const Matrix4x4& m2) const;

	/** Scale the given matrix by the given scale factor
	 *  @param f the scale factor
	 *  @param m the matrix to scale (will not be modified)
	 *  @return the scaled matrix
	 */
	friend Matrix4x4 operator*(double f, const Matrix4x4& m);

	friend std::ostream& operator<<(std::ostream& os, const Matrix4x4& m);
	friend std::istream& operator>>(std::istream& is, Matrix4x4& m);

	/** Compute the determinant of "this" matrix
	 * @return the determinant
	 */
	double determinant() const;

	/** Retrieve a specific element of the matrix
	 *  @param r the row containing the desired element
	 *  @param c the column containing the desired element
	 *  @return the element at [r][c]
	 *  @see setElementAt
	 */
	double elementAt(int r, int c) const;

	/** Extract the components of the matrix into the given single-precision
	 *  array in column-major order.
	 *  @param m [OUTPUT ONLY] an array of float of length 16
	 *  @return the base address of "m" so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array
	 *          of float.
	 *  @see fromColMajor
	 */
	float* extractColMajor(float m[16]) const;

	/** Extract the components of the matrix into the given double-precision
	 *  array in column-major order.
	 *  @param m [OUTPUT ONLY] an array of double of length 16
	 *  @return the base address of "m" so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array
	 *          of double.
	 *  @see fromColMajor
	 */
	double* extractColMajor(double m[16]) const;

	/** Extract the components of the matrix into the given single-precision
	 *  array in row-major order.
	 *  @param m [OUTPUT ONLY] an array of float of length 16
	 *  @return the base address of "m" so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array
	 *          of float.
	 *  @see fromRowMajor
	 */
	float* extractRowMajor(float m[16]) const;

	/** Extract the components of the matrix into the given double-precision
	 *  array in row-major order.
	 *  @param m [OUTPUT ONLY] an array of double of length 16
	 *  @return the base address of "m" so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array
	 *          of double.
	 *  @see fromRowMajor
	 */
	double* extractRowMajor(double m[16]) const;

	/** Retrieve the affine transformation embedded in "this" 4x4 matrix.
	 *  This method assumes that "this" matrix encodes an affine transformation
	 *  and it simply places the upper left 3x3 into M, and
	 *  the first three rows of the fourth column into t. It does not
	 *  check to make sure that the 4th row is (0, 0, 0, 1).
	 *  @param M [OUTPUT ONLY] the 3x3 matrix into which the upper left 3x3
	 *         submatrix of "this" matrix is to be written
	 *  @param t [OUTPUT ONLY] the 3D vector into which the first three rows of
	 *         the fourth column of "this" matrix is to be written.
	 *  @see isAffineTransformation - a method that can be used to verify
	 *       whether "this" matrix does actually encode an affine transformation.
	 */
	void extractAffineMt(Matrix3x3& M, AffVector& t) const;

	/** The following routine attempts to find the unit axis vector (w) and
	 *  angle (theta) which, when interpreted as a rotatition axis and angle,
	 *  will reproduce the upper 3x3 portion of the matrix. It then computes a
	 *  base point B which, when combined with (w,theta) will at least
	 *  approximate the entire 4x4 matrix. At worst, a residual
	 *  "postTranslation" will be required to reproduce the effect
	 *  of the original 4x4 matrix. That translation (possibly 0)
	 *  is returned in "postTranslation".
	 *  @param B [OUTPUT ONLY] the base point for the equivalent rotation axis
	 *  @param w [OUTPUT ONLY] the unit rotation axis vector
	 *  @param theta [OUTPUT ONLY] the computed rotation angle in radians
	 *  @param postTranslation [OUTPUT ONLY] the post-translation required to
	 *         complete the matching of the original 4x4 matrix
	 *  @return <ul><li>Matrix4x4::Extracted_BwTheta if the extraction was successful</li><li>Matrix4x4::NotAffine if the bottom row of "this" matrix is not (0,0,0,1)</li><li>Matrix3x3::NotOrthogonal if the upper left 3x3 is not an orthogonal matrix</li><li>Matrix3x3::NotRightHanded if the upper left 3x3 is not a right-handed matrix</li></ul>
	 */
	int extractAxisAngle(AffPoint& B, AffVector& w, double& theta,
					AffVector& postTranslation) const;

	/** Invert "this" matrix. This method is NOT highly numerical stable. It
	 *  is OK with well-conditioned matrices. For a better matrix inverse
	 *  implementation for when the matrix may be poorly conditioned, use
	 *  MatrixNxN::inverse.
	 *  @param mInv [OUTPUT ONLY] the inverted matriux will be deposited here
	 *  @return true if and only if the inverse was successfully computed
	 */
	bool inverse(Matrix4x4& mInv) const;

	/** Determine whether "this" matrix represents an affine transformation
	 *  @returns an integer with bit 0 set iff M44==1; bit 1 set iff
	 *           (M41, M42, M43) != (0, 0, 0)
	 */
	int isAffineTransformation() const;

	/** Compute b = this * a
	 *  @param a [INPUT ONLY] the untransformed tuple
	 *  @param b [OUTPUT ONLY] the computed product
	 *  @param nElements the length of the 'a' and 'b' arrays. If nElements<4,
	 *         then (i) a[3] is assumed to be 1.0, and (ii) a[i] for
	 *         nElements<=i<3 are assumed to be zero. nElements cannot be
	 *         larger than 4. Only nElements positions of 'b' will be set.
	 */
	void multiply(const double a[], double b[], int nElements=4) const;

	/** Compute b = this * a
	 *  @param a [INPUT ONLY] the untransformed tuple
	 *  @param b [OUTPUT ONLY] the computed product
	 *  @param nElements the length of the 'a' and 'b' arrays. If nElements<4,
	 *         then (i) a[3] is assumed to be 1.0, and (ii) a[i] for
	 *         nElements<=i<3 are assumed to be zero. nElements cannot be
	 *         larger than 4. Only nElements positions of 'b' will be set.
	 */
	void multiply(const float a[], float b[], int nElements=4) const;

	/** Set a specific element of the matrix
	 *  @param r the row containing the desired element
	 *  @param c the column containing the desired element
	 *  @param newValue the value to be placed into this[r][c]
	 *  @see elementAt
	 */
	void setElementAt(int r, int c, double newValue);

	/** Extract the 3x3 submatrix determined by removing row "skipRow" and
	 *  column "skipCol".
	 *  @param skipRow the row to be removed
	 *  @param skipCol the column to be removed
	 *  @return the resulting 3x3 matrix
	 */
	Matrix3x3 subMatrix(int skipRow, int skipCol) const;

	// 1. Special Matrices
	static const Matrix4x4 IdentityMatrix; /**< the 4x4 identity matrix */
	static const Matrix4x4 ZeroMatrix; /**< the 4x4 zero matrix */

	// 2. Return codes for extraction of base point, unit axis vector,
	//    and rotation angle from 4x4 matrices that are 'supposed to
	//    describe' affine transformations whose upper 3x3 matrix is
	//    'supposed to be' orthogonal and right-handed. (See the Matrix3x3
	//    method "extractAxisAngle".)

	// 2.1: Abnormal internal errors that should never be returned:
	static const int InternalBasePointComputationError;

	// 2.2: "Normal" errors if user provides inappropriate matrix:
	static const int NotAffine;
	// (Note that the Matrix3x3 error codes may be returned as well.)

	// 2.3: Successful extraction of unit axis vector and angle:
	//      If the 3x3 extraction of (unit axis vector, angle) is successful,
	//      then the 4x4 utility tries to determine a base point. If
	//      successful, then "Matrix4x4::Extracted_BwTheta" is returned,
	//      otherwise the 3x3 code ("Matrix3x3::Extracted_wTheta") is returned,
	//      signifying that the (unit axis vector, angle) information was
	//      successfully computed and returned, but only an approximation to
	//      a rotation axis base point was able to be determined. In this case,
	//      'postTranslation' will contain a non-zero vector which, if added to
	//      a point after rotation, will yield the correct point.
	static const int Extracted_BwTheta;

protected:
	double mElem[4][4];

private:
	void copy(const Matrix4x4& rhs);

	static int determineBasePoint(AffVector ImMRows[], int maxWCompLoc,
					int r1, int r2, const AffVector& trans, AffPoint& B);

	void installMt(const Matrix3x3& M, const AffVector& t);
};

}

#endif
