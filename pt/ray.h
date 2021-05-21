#ifndef PT_RAY
#define PT_RAY

#include "../vecs/vector3f.h"

struct Ray { // P(t) = o + d * t
	Vec o, d;

	Ray(Vec o, Vec d) : o(o), d(d) {}

	Vec At(double t) const { return o + d * t; }
}; 

#endif // PT_RAY