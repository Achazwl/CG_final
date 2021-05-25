#ifndef VECS_VECTOR3F
#define VECS_VECTOR3F

#include <cmath>
#include <cstdlib>
#include <algorithm>

struct Vec {
	double x, y, z;
	Vec(double x=0, double y=0, double z=0):x(x), y(y), z(z) {}
	double operator [] (int d) const { return d == 0 ? x : (d == 1 ? y : z); }
	double& operator [] (int d) { return d == 0 ? x : (d == 1 ? y : z); }
	Vec operator + (const Vec &rhs) const { return Vec(x+rhs.x, y+rhs.y, z+rhs.z); } 
	Vec operator - (const Vec &rhs) const { return Vec(x-rhs.x, y-rhs.y, z-rhs.z); } 
	Vec operator - () const { return Vec(-x, -y, -z); } 
	Vec operator * (double d) const { return Vec(x*d, y*d, z*d); } 
	friend Vec operator * (double d, const Vec& rhs) { return rhs*d; } 
	Vec operator / (double d) const { return *this * (1/d); } 
	Vec operator / (const Vec &rhs) const { return Vec(x/rhs.x, y/rhs.y, z/rhs.z); } 
	Vec operator * (const Vec &rhs) const { return Vec(x*rhs.x, y*rhs.y, z*rhs.z); } 
	double norm2() { return x*x + y*y + z*z; }
	double norm() { return sqrt(x*x + y*y + z*z); }
	Vec normal() { return *this / norm(); } 
	double dot(const Vec &rhs) const { return x*rhs.x + y*rhs.y + z*rhs.z; }
	static double dot(const Vec &a, const Vec &b) { return a.dot(b); }
	Vec cross(const Vec &rhs) const {return Vec(y*rhs.z-z*rhs.y, z*rhs.x-x*rhs.z, x*rhs.y-y*rhs.x);} 
	static Vec cross(const Vec &a, const Vec &b) { return a.cross(b); }
	double max() const { return std::max({x, y, z}); }
	int argmax() const { return x>y&&x>z ? 0 : (y>z ? 1 : 2); }
	double min() const { return std::min({x, y, z}); }
	int argmin() const { return x<y&&x<z ? 0 : (y<z ? 1 : 2); }
	Vec ortho() const {
		return (fabs(x)>.1?Vec(0,1):Vec(1)).cross(*this).normal();
	}
}; 

using RGB = Vec;

#endif // VECS_VECTOR3F