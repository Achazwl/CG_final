#ifndef OBJ_SPHERE
#define OBJ_SPHERE

#include "base.h"
#include "../vecs/vector3f.h"

struct Sphere : Object3D { 
	static constexpr double eps = 1e-4;
	Sphere(double r, Vec c, Material *material): Object3D(material), r(r), c(c) {
	} 

	bool intersect(const Ray &ray, Hit &hit) const override { // returns distance, 0 if nohit 
		static double tim; // temparary usage
		Vec oc = c - ray.o;
		double b = oc.dot(ray.d);
		double delta = b * b - oc.norm2() + r * r;
		if (delta < 0) return false;
		else delta = sqrt(delta); 
		if ((tim = b - delta) > eps) {
		} else if ((tim = b + delta) > eps) {
		} else return false;
		if (0 < tim && tim < hit.t) {
			hit.set(tim, this->material, (ray.At(tim) - c).normal());
			return true;
		} else return false;
	} 

protected:
	double r;
	Vec c;
}; 

#endif // OBJ_SPHERE