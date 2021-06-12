#ifndef VECS_VECTOR2F
#define VECS_VECTOR2F

#include <cmath>
#include <cstdlib>
#include <algorithm>

struct Tex {
	F x, y;
	__device__ __host__ explicit Tex(F x=0, F y=0) : x(x), y(y) { }
	__device__ __host__ Tex(const Tex& rhs) : x(rhs.x), y(rhs.y) { }

	__device__ __host__ F operator [] (int d) const { return d == 0 ? x : y; }
	__device__ __host__ F& operator [] (int d) { return d == 0 ? x : y; }

	__device__ __host__ Tex operator + (const Tex &rhs) const { return Tex(x+rhs.x, y+rhs.y); } 
	__device__ __host__ Tex operator - (const Tex &rhs) const { return Tex(x-rhs.x, y-rhs.y); } 
	__device__ __host__ Tex operator - () const { return Tex(-x, -y); } 

	__device__ __host__ Tex operator * (F d) const { return Tex(x*d, y*d); } 
	__device__ __host__ friend Tex operator * (F d, const Tex& rhs) { return rhs*d; } 
	__device__ __host__ Tex operator / (F d) const { return *this * (1/d); } 

	__device__ __host__ Tex operator / (const Tex &rhs) const { return Tex(x/rhs.x, y/rhs.y); } 
	__device__ __host__ Tex operator * (const Tex &rhs) const { return Tex(x*rhs.x, y*rhs.y); } 

	__device__ __host__ F norm2() { return x*x + y*y; }
	__device__ __host__ F norm() { return sqrt(x*x + y*y); }
	__device__ __host__ Tex normal() { return *this / norm(); } 

	__device__ __host__ F dot(const Tex &rhs) const { return x*rhs.x + y*rhs.y; }
	__device__ __host__ static F dot(const Tex &a, const Tex &b) { return a.dot(b); }

	__device__ __host__ F min() const { return x<y ? x : y; }
	__device__ __host__ F max() const { return x>y ? x : y; }
	__device__ __host__ int argmin() const { return x<y ? 0 : 1; }
	__device__ __host__ int argmax() const { return x>y ? 0 : 1; }
	__device__ __host__ static Tex min(const Tex &lhs, const Tex &rhs) {
		return Tex(
			lhs.x < rhs.x ? lhs.x : rhs.x,
			lhs.y < rhs.y ? lhs.y : rhs.y
		);
	}
	__device__ __host__ static Tex max(const Tex &lhs, const Tex &rhs) {
		return Tex(
			lhs.x > rhs.x ? lhs.x : rhs.x,
			lhs.y > rhs.y ? lhs.y : rhs.y
		);
	}

	__device__ __host__ static void swap(Tex &lhs, Tex &rhs) {
		Tex tmp = lhs;
		lhs = rhs;
		rhs = tmp;
	}

	__device__ __host__ void debug(char end='\n') const {
		printf("%lf %lf%c", x, y, end);
	}
}; 

#endif // VECS_VECTOR2F