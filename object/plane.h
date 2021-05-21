#ifndef OBJ_PLANE
#define OBJ_PLANE

#include "base.h"
#include "../vecs/vector3f.h"

class Plane : public Object3D { // TODO what the fuck
public:
    Plane() = delete;
    Plane(const Vec &a, const Vec &b, const Vec &c, const Vec& d, Material *m) : Object3D(m) {
        this->n = Vec::cross(b-a, c-a).normal();
        this->d = Vec::dot(d, n);
    }
    ~Plane() override = default;

    bool intersect(const Ray& ray, Hit &hit) const override {
        auto b = Vec::dot(n, ray.d);
        if (fabs(b) < 1e-7) return false;
        auto t = -(Vec::dot(n, ray.o) - d) / b;
        if (t > hit.t) return false;
        hit.set(t, material, n);
        return true;
    }

protected:
    Vec n;
    float d;
};

#endif //OBJ_PLANE