#ifndef VECS_VECTOR3F
#define VECS_VECTOR3F

#include <cmath>
#include <cstdlib>
#include <algorithm>

struct Vec {
	F x, y, z;
	Vec(F x=0, F y=0, F z=0):x(x), y(y), z(z) {}
	F operator [] (int d) const { return d == 0 ? x : (d == 1 ? y : z); }
	F& operator [] (int d) { return d == 0 ? x : (d == 1 ? y : z); }
	Vec operator + (const Vec &rhs) const { return Vec(x+rhs.x, y+rhs.y, z+rhs.z); } 
	Vec operator - (const Vec &rhs) const { return Vec(x-rhs.x, y-rhs.y, z-rhs.z); } 
	Vec operator - () const { return Vec(-x, -y, -z); } 
	Vec operator * (F d) const { return Vec(x*d, y*d, z*d); } 
	friend Vec operator * (F d, const Vec& rhs) { return rhs*d; } 
	Vec operator / (F d) const { return *this * (1/d); } 
	Vec operator / (const Vec &rhs) const { return Vec(x/rhs.x, y/rhs.y, z/rhs.z); } 
	Vec operator * (const Vec &rhs) const { return Vec(x*rhs.x, y*rhs.y, z*rhs.z); } 
	F norm2() { return x*x + y*y + z*z; }
	F norm() { return sqrt(x*x + y*y + z*z); }
	Vec normal() { return *this / norm(); } 
	F dot(const Vec &rhs) const { return x*rhs.x + y*rhs.y + z*rhs.z; }
	static F dot(const Vec &a, const Vec &b) { return a.dot(b); }
	Vec cross(const Vec &rhs) const {return Vec(y*rhs.z-z*rhs.y, z*rhs.x-x*rhs.z, x*rhs.y-y*rhs.x);} 
	static Vec cross(const Vec &a, const Vec &b) { return a.cross(b); }
	F max() const { return std::max({x, y, z}); }
	int argmax() const { return x>y&&x>z ? 0 : (y>z ? 1 : 2); }
	F min() const { return std::min({x, y, z}); }
	int argmin() const { return x<y&&x<z ? 0 : (y<z ? 1 : 2); }
	static std::tuple<Vec, Vec, Vec> orthoBase(Vec w) { // input w is normalized
		Vec u = (fabs(w.x)>.1?Vec(0,1):Vec(1)).cross(w).normal();
		Vec v = cross(w, u);
		return {u, v, w};
	}
}; 

using RGB = Vec;

#endif // VECS_VECTOR3F