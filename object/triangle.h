#ifndef OBJ_TRIANGLE
#define OBJ_TRIANGLE

#include "base.h"
#include "../vecs/vector3f.h"
#include "../vecs/matrix3f.h"

struct Triangle : Object3D { 
	Triangle(const Vec &a, const Vec &b, const Vec &c, Material *m) : Object3D(m), v{a,b,c} {
        this->E1 = v[0] - v[1];
        this->E2 = v[0] - v[2];
        this->n = Vec::cross(E1, E2).normal(); // 0, 1, 2 counter clockwise is front face
    }

	bool intersect(const Ray& ray, Hit& hit) const override {
	    auto S = v[0] - ray.o;
	    auto div = Matrix::det(ray.d, E1, E2);
	    if (fabs(div) < 1e-7) return false;
	    auto t = Matrix::det(S, E1, E2) / div;
	    if (t > hit.t) return false;
        auto a = Matrix::det(ray.d, S, E2) / div;
        if (a < 0 || a > 1) return false;
        auto b = Matrix::det(ray.d, E1, S) / div;
        if (b < 0 || b > 1) return false;
        if (a + b > 1) return false;

        hit.set(t, material, n);
        return true;
	}

protected:
    Vec n;
    Vec v[3];
    Vec E1, E2;
}; 

#endif // OBJ_Triangle