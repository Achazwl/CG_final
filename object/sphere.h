#ifndef OBJ_SPHERE
#define OBJ_SPHERE

#include "base.h"

struct Sphere { 
	Sphere() = default;
	Sphere(F r, Vec c): r(r), c(c) {} 
	Sphere(const Sphere& rhs) = default;

	__device__ bool intersect(const Ray &ray, Hit &hit) const {
		F tim; // temparary usage
		Vec oc = c - ray.o;
		F b = oc.dot(ray.d);
		F delta = b * b - oc.norm2() + r * r;
		if (delta < 0) return false;
		else delta = sqrt(delta); 
		if ((tim = b - delta) > eps) {
		} else if ((tim = b + delta) > eps) {
		} else return false;
		if (hiteps < tim && tim < hit.t) {
			hit.set(tim, (ray.At(tim) - c).normal()); // TODO texture modify
			return true;
		} else return false;
	} 

public: // TODO protected:
	F r;
	Vec c;
}; 

#endif // OBJ_SPHERE