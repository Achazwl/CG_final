#ifndef VECS_VECTOR3F
#define VECS_VECTOR3F

#include "../utils/rnd.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

struct Vec {
	F x, y, z;
	__device__ __host__ Vec(F v=0) : x(v), y(v), z(v) {}
	__device__ __host__ Vec(F x, F y, F z) : x(x), y(y), z(z) {}
	__device__ __host__ Vec(const Vec& rhs) : x(rhs.x), y(rhs.y), z(rhs.z) {}

	__device__ __host__ F operator [] (int d) const { return d == 0 ? x : (d == 1 ? y : z); }
	__device__ __host__ F& operator [] (int d) { return d == 0 ? x : (d == 1 ? y : z); }

	__device__ __host__ Vec operator + (const Vec &rhs) const { return Vec(x+rhs.x, y+rhs.y, z+rhs.z); } 
	__device__ __host__ Vec operator - (const Vec &rhs) const { return Vec(x-rhs.x, y-rhs.y, z-rhs.z); } 
	__device__ __host__ Vec operator - () const { return Vec(-x, -y, -z); } 

	__device__ __host__ Vec operator * (F d) const { return Vec(x*d, y*d, z*d); } 
	__device__ __host__ friend Vec operator * (F d, const Vec& rhs) { return rhs*d; } 
	__device__ __host__ Vec operator / (F d) const { return *this * (1/d); } 

	__device__ __host__ Vec operator / (const Vec &rhs) const { return Vec(x/rhs.x, y/rhs.y, z/rhs.z); } 
	__device__ __host__ Vec operator * (const Vec &rhs) const { return Vec(x*rhs.x, y*rhs.y, z*rhs.z); } 

	__device__ __host__ F norm2() { return x*x + y*y + z*z; }
	__device__ __host__ F norm() { return sqrt(x*x + y*y + z*z); }
	__device__ __host__ Vec normal() { return *this / norm(); } 

	__device__ __host__ F dot(const Vec &rhs) const { return x*rhs.x + y*rhs.y + z*rhs.z; }
	__device__ __host__ static F dot(const Vec &a, const Vec &b) { return a.dot(b); }

	__device__ __host__ Vec cross(const Vec &rhs) const {return Vec(y*rhs.z-z*rhs.y, z*rhs.x-x*rhs.z, x*rhs.y-y*rhs.x);} 
	__device__ __host__ static Vec cross(const Vec &a, const Vec &b) { return a.cross(b); }

	__device__ __host__ F min() const { return x<y&&x<z ? x : (y<z ? y : z); }
	__device__ __host__ F max() const { return x>y&&x>z ? x : (y>z ? y : z); }
	__device__ __host__ int argmin() const { return x<y&&x<z ? 0 : (y<z ? 1 : 2); }
	__device__ __host__ int argmax() const { return x>y&&x>z ? 0 : (y>z ? 1 : 2); }
	__device__ __host__ static Vec min(const Vec &lhs, const Vec &rhs) {
		return Vec(
			lhs.x < rhs.x ? lhs.x : rhs.x,
			lhs.y < rhs.y ? lhs.y : rhs.y,
			lhs.z < rhs.z ? lhs.z : rhs.z
		);
	}
	__device__ __host__ static Vec max(const Vec &lhs, const Vec &rhs) {
		return Vec(
			lhs.x > rhs.x ? lhs.x : rhs.x,
			lhs.y > rhs.y ? lhs.y : rhs.y,
			lhs.z > rhs.z ? lhs.z : rhs.z
		);
	}

	__device__ __host__ static void swap(Vec &lhs, Vec &rhs) {
		Vec tmp = lhs;
		lhs = rhs;
		rhs = tmp;
	}

	__device__ __host__ static void orthoBase(Vec w, Vec &u, Vec &v) { // input w is normalized
		u = (fabs(w.x)>.1?Vec(0,1,0):Vec(1,0,0)).cross(w).normal();
		v = cross(w, u);
	}

	__device__ static Vec rand(F range, curandState *st) {
		return Vec(rnd(range, st), rnd(range, st), rnd(range, st));
	}

	__device__ __host__ void debug(char end='\n') const {
		printf("%lf %lf %lf%c", x, y, z, end);
	}
}; 

using RGB = Vec;

#endif // VECS_VECTOR3F