/** @file Packed3DArray.h
 *  A Template class that can be used to build a 3D array of objects. The
 *  size of this array is fixed at creation time and cannot be altered.
 *  <p>This is OPEN SOURCE software developed by James R. Miller (jrmiller@ku.edu)
 *  Original version: ca. 1996. See README_*.txt for more information.</p>
 *  To use this template class, the type used for "T" must be EITHER
 *  (1) a primitive type, OR
 *  (2) a class that has the following public methods:
 *      (2.a) a default constructor
 *      (2.b) a copy constructor
 *      (2.c) operator=
 *      and for which the standard I/O operators (operator>> and operator<<)
 *      are defined.
 *  @see Packed2DArray
 *  @see ArrayList
 *  @see ImageReader
 */

#ifndef PACKED3DARRAY_H
#define PACKED3DARRAY_H

#include <iostream>

namespace cryph
{

template <typename T>
class Packed3DArray
{
public:
	/** Default constructor and a constructor that takes explicit dimensions.
     *  @param dim1 the first dimension
     *  @param dim2 the second dimension
     *  @param dim3 the third dimension
     */
	Packed3DArray(int dim1=2, int dim2=2, int dim3=2);

	/** The copy constructor.
	 *  @param t3da the Packed3DArray instance to be copied
	 */
	Packed3DArray(const Packed3DArray<T>& t3da);

	/** The destructor */
	virtual ~Packed3DArray();

	/** Returns a pointer to the actual internal array for read-only access
	 *  @return a read-only pointer to the start of the actual internal array
	 */
	const T* getData() const { return mData; }

	/** Return an element at a specific location
	 *  @param i1 the first index
	 *  @param i2 the second index
	 *  @param i3 the third index
	 *  @return the element at the given location, if [i1][i2][i3] represents a
	 *     valid array access. Otherwise returns an "out of bounds" value.
	 *  @see setOutOfBoundsValue
	 */
	T getDataElement(int i1, int i2, int i3) const;

	/** Return a pointer to the item in the internal array at the given location
	 *  @param i1 the first index
	 *  @param i2 the second index
	 *  @param i3 the third index
	 *  @return a pointer into the internal array to the actual element, if
	 *          [i1][i2][i3] represents a valid array access. nullptr otherwise.
	 */
	const T* getDataElementLoc(int i1, int i2, int i3) const;

	/** Return the first array dimension
	 *  @return the first dimension
	 */
	int getDim1() const { return mDim1; }

	/** Return the second array dimension
	 *  @return the second dimension
	 */
	int getDim2() const { return mDim2; }

	/** Return the third array dimension
	 *  @return the third dimension
	 */
	int getDim3() const { return mDim3; }

	/** Return a pointer to the actual internal array for RW access
	 *  @return a RW pointer to the start of the actual internal array
	 */
	T* getModifiableData() { return mData; }

	/** Return the total number of elements in this array
	 *  @return the total number of elements computed as the product of
	 *          the three dimensions
	 */
	int getTotalNumberElements() const { return mDim1 * mDim2 * mDim3; }

	/** Set a specific element of the array. If the indices are not valid,
	 *  no change is made to the array.
	 *  @param i1 the first index
	 *  @param i2 the second index
	 *  @param i3 the third index
	 */
	void setDataElement(int i1, int i2, int i3, const T& elem);

	/** Specify whether errors such as invalid array accesses are reported
	 *  to the console. By default, they are reported.
	 */
	static void	setErrorReporting(bool r) { sReportErrors = r; }

	/** Specify a specific value to be returned by getDataElement if invalid
	 *  array indices are provided.
	 */
	static void	setOutOfBoundsValue(T defaultValue) { sOutOfBoundsValue = defaultValue; }

private:

	int		getOffset(const char* routine, int i1, int i2, int i3) const;

	T*		mData;
	int		mDim1;
	int		mDim2;
	int		mDim3;

	static	bool	sReportErrors;
	static	T		sOutOfBoundsValue;
};

template <typename T>
bool	Packed3DArray<T>::sReportErrors = true;

template <typename T>
T		Packed3DArray<T>::sOutOfBoundsValue;

template <typename T>
Packed3DArray<T>::Packed3DArray(int dim1, int dim2, int dim3) :
		mData(nullptr), mDim1(dim1), mDim2(dim2), mDim3(dim3)
{
	if ( (dim1 < 1) || (dim2 < 1) || (dim3 < 1) )
	{
		mDim1 = 0; mDim2 = 0; mDim3 = 0;
		if (Packed3DArray<T>::sReportErrors)
			std::cerr << "Invalid dimensions in constructor: ("
			     << dim1 << ", " << dim2 << ", " << dim3 << ')' << std::endl;
	}
	else
		mData = new T[dim1*dim2*dim3];
}

template <typename T>
Packed3DArray<T>::Packed3DArray(const Packed3DArray<T>& t3da) :
		mData(nullptr), mDim1(t3da.mDim1), mDim2(t3da.mDim2), mDim3(t3da.mDim3)
{
	int		size = mDim1 * mDim2 * mDim3;
	mData = new T[size];
	for (int i=0 ; i<size ; i++)
		mData[i] = t3da.mData[i];
}

template <typename T>
Packed3DArray<T>::~Packed3DArray()
{
	if (mData != nullptr)
	{
		delete [] mData;
		mData = nullptr;
		mDim1 = 0;
		mDim2 = 0;
		mDim3 = 0;
	}
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Packed3DArray<T>& t3da)
{
	int size = t3da.getTotalNumberElements();
	const T* Tarr = t3da.getData();
	for (int i=0 ; i<size ; i++)
	{
		// WARNING: Assuming integral type for now, hence cast to integer
		//          first. For example, if "typename T" is GLubyte, this ensures
		//          that full 8-bit ints are written as opposed to general
		//          ASCII chars.
		os << (int)Tarr[i];
		if ((i % 20) == 0)
			os << '\n';
		else
			os << ' ';
	}
	os << '\n';
	return os;
}

template <typename T>
std::istream& operator>>(std::istream& is, Packed3DArray<T>& t3da)
{
	int size = t3da.getTotalNumberElements();
	int temp = 0;
	T* Tarr = t3da.getModifiableData();
	for (int i=0 ; i<size ; i++)
	{
		// WARNING: See 'WARNING' in "operator<<" above.
		is >> temp;
		Tarr[i] = (T)temp;
	}
	return is;
}

template <typename T>
T Packed3DArray<T>::getDataElement(int i1, int i2, int i3) const
{
	int loc = getOffset("getDataElement",i1,i2,i3);
	if (loc < 0)
	{
		return Packed3DArray<T>::sOutOfBoundsValue;
	}
	return mData[loc];
}

template <typename T>
const T* Packed3DArray<T>::getDataElementLoc(int i1, int i2, int i3) const
{
	int loc = getOffset("getDataElementLoc",i1,i2,i3);
	if (loc < 0)
		return nullptr;
	return &mData[loc];
}

template <typename T>
int Packed3DArray<T>::getOffset(const char* routine, int i1, int i2, int i3)
	const
{
	if ( (i1 < 0) || (i1 >= mDim1) ||
	     (i2 < 0) || (i2 >= mDim2) ||
	     (i3 < 0) || (i3 >= mDim3) )
	{
		if (Packed3DArray<T>::sReportErrors)
			std::cerr << routine << ": Invalid data element reference: ("
			     << i1 << ", " << i2 << ", " << i3 << ')' << std::endl;
		return -1;
	}
	return (i1 * mDim2 * mDim3) + (i2 * mDim3) + i3;
}

template <typename T>
void Packed3DArray<T>::setDataElement(int i1, int i2, int i3, const T& elem)
{
	int loc = getOffset("setDataElement",i1,i2,i3);
	if (loc >= 0)
		mData[loc] = elem;
}

}

#endif
