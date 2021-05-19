#ifndef OBJ_SPHERE
#define OBJ_SPHERE

#include "base.h"
#include "../vecs/vector3f.h"

struct Sphere : Object3D { 
	static constexpr double eps = 1e-4;
	Sphere(double r, Vec c, Vec e, Vec col, Refl refl): r(r), c(c), Object3D(e, col, refl) {} 

	bool intersect(const Ray &ray, double &t) const override { // returns distance, 0 if nohit 
		static double ret; // temparary usage
		Vec oc = c - ray.o;
		double b = oc.dot(ray.d);
		double delta = b * b - oc.norm2() + r * r;
		if (delta < 0) return false;
		else delta = sqrt(delta); 
		if ((ret = b - delta) > eps) {
		} else if ((ret = b + delta) > eps) {
		} else return false;
		if (ret < t) {
			t = ret;
			return true;
		} else {
			return false;
		}
	} 

public: // TODO protected
	double r;
	Vec c;
}; 

#endif // OBJ_SPHERE