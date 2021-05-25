#ifndef OBJ_PLANE
#define OBJ_PLANE

#include "base.h"

class Plane : public Object3D { // TODO what the fuck
public:
    Plane() = delete;
    Plane(const Vec &n, double d, Material *m) : Object3D(m), n(n), d(d) { }
    ~Plane() override = default;

    bool intersect(const Ray& ray, Hit &hit) const override {
        auto b = Vec::dot(n, ray.d);
        if (fabs(b) < 1e-7) return false;
        auto t = -(Vec::dot(n, ray.o) - d) / b;
        if (t < hiteps || t > hit.t) return false;
        hit.set(t, material, n);
        return true;
    }

protected:
    Vec n;
    double d;
};

#endif //OBJ_PLANE