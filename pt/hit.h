#ifndef PT_HIT
#define PT_HIT

#include "../vecs/vector3f.h"
#include "../object/base.h"

constexpr double hiteps = 1e-3;

struct Hit {
    float t;
    const Object3D *o;
    Vec n;

    Hit(double t): t(t), o(nullptr), n() {}
    ~Hit() = default;

    void set(double t, const Object3D *o, const Vec &n) {
        this->t = t;
        this->o= o;
        this->n = n;
    }
};

#endif // PT_HIT