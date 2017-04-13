// ProjPoint.c++ -- 4D Projective Space points
// This is OPEN SOURCE software developed by James R. Miller (jrmiller@ku.edu)
// Original version: ca. 1996. See README_*.txt for more information.

#include <stdlib.h>

#include "Inline.h"
#include "ProjPoint.h"

using namespace std;

namespace cryph
{

ProjPoint::ProjPoint() : x(0.0), y(0.0), z(0.0), w(1.0)
{
}

ProjPoint::ProjPoint(const ProjPoint& p) : x(p.x), y(p.y), z(p.z), w(p.w)
{
}

ProjPoint::ProjPoint(const AffPoint& p, double W) :
	x(W*p[X]), y(W*p[Y]), z(W*p[Z]), w(W)
{
}

ProjPoint::ProjPoint(const double p[]) : x(p[0]), y(p[1]), z(p[2]), w(p[3])
{
}

ProjPoint::ProjPoint(const float p[]) : x(p[0]), y(p[1]), z(p[2]), w(p[3])
{
}

ProjPoint::ProjPoint(double xx, double yy, double zz, double ww) :
	x(xx), y(yy), z(zz), w(ww)
{
}

ProjPoint::~ProjPoint()
{
}

double* ProjPoint::aCoords(double coords[], int offset) const
{
	// extract affine coords and place into a double precision array

	coords[offset  ] = x/w;
	coords[offset+1] = y/w;
	coords[offset+2] = z/w;
	return coords;
}

double* ProjPoint::aCoords(double coords[][3], int offset) const
{
	// extract affine coords and place into a double precision array

	coords[offset][0] = x/w;
	coords[offset][1] = y/w;
	coords[offset][2] = z/w;
	return (double*) coords;
}

float* ProjPoint::aCoords(float coords[], int offset) const
{
	// extract affine coords and place into a float array

	coords[offset  ] = static_cast<float>(x/w);
	coords[offset+1] = static_cast<float>(y/w);
	coords[offset+2] = static_cast<float>(z/w);
	return coords;
}

float* ProjPoint::aCoords(float coords[][3], int offset) const
{
	// extract affine coords and place into a float array

	coords[offset][0] = static_cast<float>(x/w);
	coords[offset][1] = static_cast<float>(y/w);
	coords[offset][2] = static_cast<float>(z/w);
	return (float*) coords;
}

ProjPoint ProjPoint::operator=(const ProjPoint& rhs)
{
	x = rhs.x; y = rhs.y ; z = rhs.z; w = rhs.w;
	return *this;
}

ProjPoint ProjPoint::operator+=(const ProjPoint& rhs)
{
	x += rhs.x; y += rhs.y ; z += rhs.z; w += rhs.w;
	return *this;
}

ProjPoint ProjPoint::operator*=(double f)
{
	x *= f; y *= f ; z *= f; w *= f;
	return *this;
}

ProjPoint ProjPoint::operator/=(double f)
{
	x /= f; y /= f ; z /= f; w /= f;
	return *this;
}

static const double unspecifiedValue = -123.456;
double ProjPoint::operator[](int index) const
{
	// read-only indexing

	switch (index)
	{
		case X:
			return x;
		case Y:
			return y;
		case Z:
			return z;
		case W:
			return w;
	}

	return unspecifiedValue;
}

ostream& operator<<(ostream& os, const ProjPoint& p)
{
	os << p[X] << ' ' << p[Y] << ' ' << p[Z] << ' ' << p[W];
	return os;
}

istream& operator>>(istream& is, ProjPoint& p)
{
	double x, y, z, w;
	is >> x >> y >> z >> w;
	p = ProjPoint(x,y,z,w);

	return is;
}

double* ProjPoint::pCoords(double* coords, int offset) const
{
	coords[offset  ] = x;
	coords[offset+1] = y;
	coords[offset+2] = z;
	coords[offset+3] = w;
	return coords;
}

double* ProjPoint::pCoords(double coords[][4], int offset) const
{
	coords[offset][0] = x;
	coords[offset][1] = y;
	coords[offset][2] = z;
	coords[offset][3] = w;
	return (double*) coords;
}

float* ProjPoint::pCoords(float* coords, int offset) const
{
	coords[offset  ] = static_cast<float>(x);
	coords[offset+1] = static_cast<float>(y);
	coords[offset+2] = static_cast<float>(z);
	coords[offset+3] = static_cast<float>(w);
	return coords;
}

float* ProjPoint::pCoords(float coords[][4], int offset) const
{
	coords[offset][0] = static_cast<float>(x);
	coords[offset][1] = static_cast<float>(y);
	coords[offset][2] = static_cast<float>(z);
	coords[offset][3] = static_cast<float>(w);
	return (float*) coords;
}

void ProjPoint::swizzle(char xyzw[4])
{
	double xyzwC[4], xyzwO[4];
	pCoords(xyzwC);
	pCoords(xyzwO);
	for (int i=0 ; i<4 ; i++)
		switch (xyzw[i])
		{
			case 'x': xyzwC[i] = xyzwO[0]; break;
			case 'y': xyzwC[i] = xyzwO[1]; break;
			case 'z': xyzwC[i] = xyzwO[2]; break;
			case 'w': xyzwC[i] = xyzwO[3]; break;
			case 'X': xyzwC[i] = -xyzwO[0]; break;
			case 'Y': xyzwC[i] = -xyzwO[1]; break;
			case 'Z': xyzwC[i] = -xyzwO[2]; break;
			case 'W': xyzwC[i] = -xyzwO[3]; break;
			default:
				; // just ignore
		}
	x = xyzwC[0];
	y = xyzwC[1];
	z = xyzwC[2];
	w = xyzwC[3];
}

}
