#ifndef PT_RAY
#define PT_RAY

#include "../vecs/vector3f.h"

struct Ray { // P(t) = o + d * t
	Vec o, d;

	__device__ __host__ Ray(Vec o, Vec d) : o(o), d(d) {}

	__device__ Vec At(F t) const { return o + d * t; }
}; 

#endif // PT_RAY