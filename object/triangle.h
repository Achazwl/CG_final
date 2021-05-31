#ifndef OBJ_TRIANGLE
#define OBJ_TRIANGLE

#include "base.h"

struct Triangle { 
	Triangle() = default;
	Triangle(const Vec &a, const Vec &b, const Vec &c) : v{a,b,c} {
        this->E1 = v[2] - v[0];
        this->E2 = v[1] - v[0];
        this->n = Vec::cross(E1, E2).normal(); // 0, 1, 2 counter clockwise is front face
    }
	Triangle(const Triangle &rhs) = default;

	__device__ bool intersect(const Ray& ray, Hit& hit) const {
	    auto S = ray.o - v[0];
        auto p = Vec::cross(ray.d, E2), q = Vec::cross(S, E1); // temparary variable (common calculation)
        auto div = Vec::dot(E1, p);
	    if (fabs(div) < 1e-7) return false;
	    auto t = Vec::dot(q, E2) / div;
	    if (t < hiteps || t > hit.t) return false;
        auto a = Vec::dot(S, p) / div;
        if (a < 0 || a > 1) return false;
        auto b = Vec::dot(ray.d, q) / div;
        if (b < 0 || b > 1) return false;
        if (a + b > 1) return false;
        hit.set(t, div < 0 ? n : -n);
        return true;
	}

public: // TODO protected:
    Vec n;
    Vec v[3];
    Vec E1, E2;
}; 

#endif // OBJ_TRIANGLE