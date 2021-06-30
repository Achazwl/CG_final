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
			auto p = (ray.At(tim) - c).normal();
			F u = atan2(p.y, p.x)*0.5*M_1_PI, v = asin(p.z)*M_1_PI+0.5;
			hit.set(tim, p, Tex(u+0.5, v));
			return true;
		} else return false;
	} 

protected:
	F r;
	Vec c;
}; 

#endif // OBJ_SPHERE