#ifndef OBJ_SPHERE
#define OBJ_SPHERE

#include "base.h"

struct Sphere : public Object3D { 
	static constexpr F eps = 1e-4;
	Sphere(F r, Vec c, Material *material): Object3D(material), r(r), c(c) {
		bound = Bound(
			c - Vec(r, r, r),
			c + Vec(r, r, r)
		);
	} 

	bool intersect(const Ray &ray, Hit &hit) const override { // returns distance, 0 if nohit 
		static F tim; // temparary usage
		Vec oc = c - ray.o;
		F b = oc.dot(ray.d);
		F delta = b * b - oc.norm2() + r * r;
		if (delta < 0) return false;
		else delta = sqrt(delta); 
		if ((tim = b - delta) > eps) {
		} else if ((tim = b + delta) > eps) {
		} else return false;
		if (hiteps < tim && tim < hit.t) {
			hit.set(tim, this, (ray.At(tim) - c).normal());
			return true;
		} else return false;
	} 

    Vec getColor(const Vec &p) const override {
        if (material->useTexture()) {
			auto v = p - c;
            if (material->filename == "images/volleyball.jpg")
                return material->getcol(
					atan2(v.y, v.x) * 0.5 * M_1_PI,
					asin(v.z / r) * M_1_PI + 0.5
                );
        }
		return material->Kd;
    }

protected:
	F r;
	Vec c;
}; 

#endif // OBJ_SPHERE