/** @file Inline.h
 *  Definitions of common low-level and typically inline functions
 *  This is OPEN SOURCE software developed by James R. Miller (jrmiller@ku.edu)
 *  Original version: ca. 1996. See README_*.txt for more information.
 */

#ifndef INLINE_H
#define INLINE_H

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>

namespace cryph
{

/** Return the closing bracketing character for the given opening one
 *  For example closeStr('(') returns ')'
 *  @param openStr the opening bracketing character
 *  @return the corresponding closing bracketing character (as an std::string)
 */
inline std::string closeStr(char openStr)
{
	if (openStr == '(')
		return ")";
	if (openStr == '[')
		return "]";
	if (openStr == '{')
		return "}";
	if (openStr == '<')
		return ">";
	if (openStr == ' ')
		return " ";
	return "???";
}

/** Convert the given angle in degrees to radians and return it
 *  @param angleInDegrees the given angle to be converted to radians 
 *  @return the angle converted to radians
 */
inline double degreesToRadians(double angleInDegrees)
{
	return angleInDegrees*M_PI/180.0;
}

/** Determine whether the two given values are equal within the given tolerance
 *  @param s1 the first number to be compared
 *  @param s2 the second number to be compared
 *  @param tol the tolerance to be used to decide if the values are the same
 *  @return true if and only if s1 and s2 are equal within the given tolerance
 */
inline bool equalScalars(double s1, double s2, double tol)
{
	return (fabs(s1-s2) < tol);
}

/** Convert the given angle in radians to degrees and return it
 *  @param angleInRadians the given angle to be converted to degrees
 *  @return the angle converted to degrees
 */
inline double radiansToDegrees(double angleInRadians)
{
	return angleInRadians*180.0/M_PI;
}

/** Return the next random number in the range 0 <= r < 1
 * 	@return the next random number
 */
inline double random0To1_next()
{
	return drand48();
}

/** Set the random number seed
 *  @param seedVal the value to be used as the seed
 */
inline void random0To1_seed(long seedVal)
{
	srand48(seedVal);
}

/** Return the rounded integer for the given floating point value
 *  @param val the floating point number to be rounded
 *  @return the rounded integral value
 */
inline int roundR2I(double val)
{
	return (int) (val + 0.5);
}

/** Skip over the given number of non-blank characters
 *  @param is the input stream on which we wish to skip non-blank characters
 *  @param nNonBlankChars the number of non-blank characters to be skipped
 */
inline void skipNonblankChars(std::istream& is, int nNonBlankChars)
{
	char ch;
	for (int i=0 ; i<nNonBlankChars ; i++)
		is >> ch;
}

/** Return the square of the given number
 *  @param s the number to be squared
 *  @return the square of the given number
 */
inline double square(double s)
{
	return s * s;
}

// Templates


/** Return the maximum of the two given quantities
 *  @param s1 the first quantity to be compared
 *  @param s2 the second quantity to be compared
 *  @return the maximum of the two quantities
 */
template <typename T>
inline T maximum(T s1, T s2)
{
	return (s1 >= s2) ? s1 : s2;
}

/** Return the minimum of the two given quantities
 *  @param s1 the first quantity to be compared
 *  @param s2 the second quantity to be compared
 *  @return the minimum of the two quantities
 */
template <typename T>
inline T minimum(T s1, T s2)
{
	return (s1 >= s2) ? s2 : s1;
}

/** Swap the two given quantities
 *  @param s1 the first quantity to be swapped
 *  @param s2 the second quantity to be swapped
 */
template <typename T> inline void swap2(T& t1, T& t2)
{
	T temp = t1;
	t1 = t2;
	t2 = temp;
}

}
 
#endif
