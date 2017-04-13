/** @file Matrix3x3.h
 *  Class definition for creating, manipulating, and using 3x3 matrices
 *  @see Matrix4x4
 *  This is OPEN SOURCE software developed by James R. Miller (jrmiller@ku.edu)
 *  Original version: ca. 1996. See README_*.txt for more information.
 */

#ifndef MATRIX3x3_H
#define MATRIX3x3_H

#include "AffPoint.h"
#include "AffVector.h"

namespace cryph
{

/** This class is designed for the specification and manipulation of 3x3
 *  matrices as a part of Affine transformations specified on 3D vectors
 *  As such, it provides facilities to create 3x3 matrices that rotate
 *  and mirror vectors, form tensor products of vectors, and provide other
 *  support required by Matrix4x4 in its role of representing transformations
 *  of 3D geometry. This interface is NOT intended to provide facilities
 *  for representing 2D transformations in homogeneous form.
 */
class Matrix3x3
{
public:
	/** Default constructor creates a 3x3 identity matrix */
	Matrix3x3();
	/** The copy constructor
	 *  @param M the matrix whose elements are copied
	 */
	Matrix3x3(const Matrix3x3& M);
	/** Construct a 3x3 matrix directly from 9 values given in row-major order
	 *  @param mrc The element for row r, column c
	 */
	Matrix3x3(
		double m11, double m12, double m13,
		double m21, double m22, double m23,
		double m31, double m32, double m33);
	/** The destructor */
	virtual ~Matrix3x3();

	/** Factory method to create a 3x3 matrix that can be used to
	 *  rotate a vector by a given angle about the x-axis.
	 *  @param angle the desired rotation angle in degrees
	 *  @return the 3x3 matrix encoding the rotation
	 */
	static Matrix3x3 xRotationDegrees(double angle);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  rotate a vector by a given angle about the x-axis.
	 *  @param angle the desired rotation angle in radians
	 *  @return the 3x3 matrix encoding the rotation
	 */
	static Matrix3x3 xRotationRadians(double angle);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  rotate a vector by a given angle about the y-axis.
	 *  @param angle the desired rotation angle in degrees
	 *  @return the 3x3 matrix encoding the rotation
	 */
	static Matrix3x3 yRotationDegrees(double angle);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  rotate a vector by a given angle about the y-axis.
	 *  @param angle the desired rotation angle in radians
	 *  @return the 3x3 matrix encoding the rotation
	 */
	static Matrix3x3 yRotationRadians(double angle);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  rotate a vector by a given angle about the z-axis.
	 *  @param angle the desired rotation angle in degrees
	 *  @return the 3x3 matrix encoding the rotation
	 */
	static Matrix3x3 zRotationDegrees(double angle);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  rotate a vector by a given angle about the z-axis.
	 *  @param angle the desired rotation angle in radians
	 *  @return the 3x3 matrix encoding the rotation
	 */
	static Matrix3x3 zRotationRadians(double angle);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  rotate a vector by a given angle about the given axis vector.
	 *  @param rotationAxis a 3D vector about which the rotation is to be performed
	 *  @param angle the desired rotation angle in radians
	 *  @return the 3x3 matrix encoding the rotation
	 */
	static Matrix3x3 generalRotationRadians
					(const AffVector& rotationAxis, double angle);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  rotate one 3D vector onto another 3D vector.
	 *  @param uFrom the 3D vector to be aligned with uTo
	 *  @param uTo the 3D vector onto which we want the created matrix to rotate uFrom
	 *  @return the 3x3 matrix encoding the rotation
	 *  @see extractAxisAngle
	 */
	static Matrix3x3 alignVectors
					(const AffVector& uFrom, const AffVector& uTo);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  rotate a pair of 3D vectors onto another pair of 3D vectors. Since
	 *  a rigid rotation to perform this cannot always be created, the
	 *  matrix created will align uFrom with uTo, and it will encode one additional
	 *  rotation that will rotate vFrom into the plane of (uTo, vTo) such that the
	 *  dot product of vTo with the transformed vFrom is non-negative.
	 *  @param uFrom the 3D vector to be aligned with uTo
	 *  @param vFrom the 3D vector to be aligned with vTo
	 *  @param uTo the 3D vector onto which we want the created matrix to rotate uFrom
	 *  @param vTo the 3D vector onto which we want the created matrix to rotate vFrom, subject to the constraint explained above.
	 *  @return the 3x3 matrix encoding the rotation
	 */
	static Matrix3x3 alignVectors
					(const AffVector& uFrom, const AffVector& vFrom,
					 const AffVector& uTo,   const AffVector& vTo);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  scale a vector by the given scale factors along the x-, y-,
	 *  and z-axes.
	 *  @param sx the desired scale factor in the x direction
	 *  @param sy the desired scale factor in the y direction
	 *  @param sz the desired scale factor in the z direction
	 *  @return the 3x3 matrix encoding the scale
	 */
	static Matrix3x3 scale
					(double sx, double sy, double sz);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  shear a vector. The matrix created will shear an arbitrary
	 *  vector, v, by adding to v:
	 *          f*v.dot(nHat)uHat
	 *  where nHat is a unit vector perpendicular to the shear plane;
	 *  uHat is a unit vector in the direction of the component of
	 *  the given u vector perpendicular to the given normal
	 *  vector, n; and f is a constant.
	 *  @param n the vector perpendicular to the shear plane
	 *  @param u a vector in the shear plane that specifies the shear
	 *           direction (only the component of u perpendicular to
	 *           n will be used when creating uHat)
	 *  @param f a constant adjusting the amount of shear per length of
	 *           v in the direction of nHat.
	 *  @return the 3x3 matrix encoding the shear
	 */
	static Matrix3x3 shear
					(const AffVector& n, const AffVector& u, double f);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  encode a cross product with the given vector. That is, if
	 *  U is the 3x3 matrix returned by this method, then for an
	 *  arbitrary vector v: U*v = u x v.
	 *  @param u the desired vector
	 *  @return the 3x3 matrix encoding the cross product
	 */
	static Matrix3x3 crossProductMatrix(const AffVector& u);

	/** Factory method to create a 3x3 matrix that can be used to
	 *  mirror a vector about the plane whose normal is specified
	 *  by the given vector.
	 *  @param mirrorPlaneNormal the normal vector to the desired mirror plane
	 *  @return the 3x3 matrix encoding the mirror
	 */
	static Matrix3x3 mirrorMatrix(const AffVector& mirrorPlaneNormal);

	/** Factory method to create a 3x3 matrix that is the tensor product
	 *  of the two given vectors. That is, matrix[i][j] will be u[i]*v[j]
	 *  @param u the first vector
	 *  @param v the second vector
	 *  @return the matrix that holds the tensor product of u and v.
	 */
	static Matrix3x3 tensorProductMatrix
							(const AffVector& u, const AffVector& v);

	/** The assignment operator
	 *  @param rhs the 3x3 matrix to be assigned to "this" matrix
	 *  @return the assigned matrix
	 */
	Matrix3x3 operator=(const Matrix3x3& rhs);

	/** Assign to "this" matrix the product of "this" matrix and rhs
	 *  @param rhs the matrix to be concatenated onto the right-hand side of "this" matrix
	 *  @return the updated "this" matrix
	 */
	Matrix3x3 operator*=(const Matrix3x3& rhs);

	/** Assign to "this" matrix the product of "this" matrix with the given scalar. That is,
	 *  each element of "this" matrix is multiplied by f.
	 *  @param f the scale factor
	 *  @return the updated "this" matrix
	 */
	Matrix3x3 operator*=(double f);

	/** Assign to "this" matrix the sum of "this" matrix and rhs
	 *  @param rhs the matrix to be added to "this" matrix
	 *  @return the updated "this" matrix
	 */
	Matrix3x3 operator+=(const Matrix3x3& rhs);

	/** Multiply the point p on the left by "this" matrix.
	 *  @param p the point to be multiplied
	 *  @return the transformed point
	 */
	AffPoint operator*(const AffPoint& p) const;

	/** Multiply the vector v on the left by "this" matrix.
	 *  @param v the vector to be multiplied
	 *  @return the transformed vector
	 */
	AffVector operator*(const AffVector& v) const;

	/** Multiply the matrix m2 on the left by "this" matrix.
	 *  @param m2 the matrix to be multiplied
	 *  @return the 3x3 matrix that is the product: "this" * m2
	 */
	Matrix3x3 operator*(const Matrix3x3& m2) const;

	/** Add the matrix m2 to "this" matrix.
	 *  @param m2 the matrix to be added
	 *  @return the 3x3 matrix that is the sum: "this" + m2
	 */
	Matrix3x3 operator+(const Matrix3x3& m2) const;

	/** Subtract the matrix m2 from "this" matrix.
	 *  @param m2 the matrix to be subtracted
	 *  @return the 3x3 matrix that is the difference: "this" - m2
	 */
	Matrix3x3 operator-(const Matrix3x3& m2) const;

	/** Return a matrix that is the product f*m where f is a scale factor
	 *  used to multiply each element of the matrix, m.
	 *  @param f the scale factor
	 *  @param m the matrix to be scaled (m is unchanged)
	 *  @return the product f*m
	 */
	friend Matrix3x3 operator*(double f, const Matrix3x3& m);

	/** Output the matrix m to the given output stream. Elements are output
	 *  in row-major order, all on one line, with a single space between
	 *  each element.
	 *  @param os the stream to which the matrix is written
	 *  @param m the matrix to be written
	 *  @return the updated output stream
	 */
	friend std::ostream& operator<<(std::ostream& os, const Matrix3x3& m);

	/** Read 9 elements into the matrix m from the given input stream.
	 *  Elements are read in row-major order and assumed to be whitespace-delimited.
	 *  @param is the stream from which the matrix is read
	 *  @param m the matrix into which the values read are placed
	 *  @return the updated input stream
	 */
	friend std::istream& operator>>(std::istream& is, Matrix3x3& m);

	/** Return the determinantof "this" 3x3 matrix
	 *  @return the determinant
	 */
	double determinant() const;

	/** Return "this" matrix element at row i and column j.
	 *  @param i the desired row
	 *  @param j the desired column
	 *  @return the matrix element at row i and column j
	 */
	double elementAt(int i, int j) const;

	/** Extract the components of the matrix into the given single-precision
	 *  array in column-major order.
	 *  @param m [OUTPUT ONLY] an array of float of length 9
	 *  @return the base address of "m" so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array
	 *          of float.
	 */
	float* extractColMajor(float m[9]) const;

	/** Extract the components of the matrix into the given double-precision
	 *  array in column-major order.
	 *  @param m [OUTPUT ONLY] an array of double of length 9
	 *  @return the base address of "m" so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array
	 *          of double.
	 */
	double* extractColMajor(double m[9]) const;

	/** Extract the components of the matrix into the given single-precision
	 *  array in row-major order.
	 *  @param m [OUTPUT ONLY] an array of float of length 9
	 *  @return the base address of "m" so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array
	 *          of float.
	 */
	float* extractRowMajor(float m[9]) const;

	/** Extract the components of the matrix into the given double-precision
	 *  array in row-major order.
	 *  @param m [OUTPUT ONLY] an array of double of length 9
	 *  @return the base address of "m" so that the method call can be
	 *          used as a parameter to an OpenGL function expecting an array
	 *          of double.
	 */
	double* extractRowMajor(double m[9]) const;

	/** If "this" matrix is orthogonal and right-handed,  then this method will
	 *  determine the rotation axis and angle which, if passed to
	 *  generalRotationRadians, would exactly reproduce "this" matrix.
	 *  @param w on exit, this vector will have the rotation axis
	 *  @param theta on exit, this will hold the rotation angle in radians
	 *  @return an integer code indicating whether the extraction succeeded:
	 *     <ul><li>Matrix3x3::NotOrthogonal - if "this" matrix is not orthogonal</li>
	 *         <li>Matrix3x3::NotRightHanded - if "this" matrix is not right-handed</li>
	 *         <li>Matrix3x3::Extracted_wTheta - if w and theta were successfully computed</li>
	 *     </ul>
	 *  @see isOrthogonal
	 *  @see isRightHanded
	 *  @see generalRotationRadians
	 */
	int extractAxisAngle(AffVector& w, double& theta) const;

	/** Extract the rows of "this" matrix, writing them into the given output parameters
	 *  @param row1 on exit, this will hold the first row of "this" matrix
	 *  @param row2 on exit, this will hold the second row of "this" matrix
	 *  @param row3 on exit, this will hold the third row of "this" matrix
	 */
	void extractRows(AffVector& row1, AffVector& row2, AffVector& row3) const;

	/** Compute and return the inverse of "this" matrix.
	 *  @param mInv the matrix into which the inverse will be written
	 *  @return true if the inverse was successfully computed
	 */
	bool inverse(Matrix3x3& mInv) const;

	/** Report whether "this" matrix is orthogonal. A matrix is orthogonal
	 *  if its inverse is the same as its transpose.
	 *  @return true if the matrix is orthogonal
	 *  @see extractAxisAngle
	 */
	bool isOrthogonal() const;

	/** Report whether "this" matrix is right-handed. A matrix is right-handed
	 *  if the dot product of (row1 x row2) with row3 is positive.
	 *  @return true if the matrix is right-handed
	 *  @see extractAxisAngle
	 */
	bool isRightHanded() const;

	/** Determine the diagonal element with the largest value.
	 *  Return the position in "pos" and the value itself as the method return.
	 *  @param pos on exit, the row that contains the largest diagonal element
	 *  @return the value of the element with the largest value.
	 */
	double largestDiagonalElement(int& pos) const;

	/** Multiply "a" on the left by "this" matrix, placing the result in "b":
	 *  b = "this" * a.
	 *  @param a an array of length 3 to be multiplied
	 *  @param b an array of length 3 that recieves the product "this" * a.
	 */
	void multiply(const double a[], double b[]) const;

	/** Set this[i][j] = newValue.
	 *  @param i the row containing the element to be set
	 *  @param j the column containing the element to be set
	 *  @param newValue the value to be copied into row i and column j
	 */
	void setElementAt(int i, int j, double newValue);

	/** Compute and return the trace of "this" matrix. The trace is the
	 *  sum of the diagonal elements.
	 *  @return the trace = this[0][0]+this[1][1]+this[2][2]
	 */
	double trace() const;

	/** Transpose "this" matrix in place. On exit, "this" matrix will have
	 *  been transposed.
	 */
	void transpose();

    // ---------- Global constants

	// 1. Special Matrices
	static const Matrix3x3 IdentityMatrix; /**< the 3x3 identity matrix */
	static const Matrix3x3 ZeroMatrix; /**< the 3x3 zero matrix */

	// 2. Return codes for extraction of unit axis vector & rotation angle
	//    from 3x3 matrices that are 'supposed to be' orthogonal and
	//    right-handed.

	// 2.1: Abnormal internal errors that should never be returned:
	static const int ImMDeterminantNotZero;
	static const int CannotDetermineUnitAxisVector;
	static const int CosTermsNotEqual;
	static const int SinTermsNotEqual;

	// 2.2: "Normal" errors if user provides inappropriate matrix:
	static const int NotOrthogonal;
	static const int NotRightHanded;

	// 3.3: Successful extraction of unit axis vector and angle:
	static const int Extracted_wTheta;

	friend class Matrix4x4;

protected:
	double	mElem[3][3];

private:
	void copy(const Matrix3x3& rhs);
	int computeImMRows(Matrix3x3& ImM,
						// the rows of I-M and the indices of the
						// two most linearly independent rows.
						AffVector rows[], int& r1, int& r2) const;
	int extractPrimitiveAxisAngle(int pos, AffVector& w, double& theta) const;
};

}

#endif
