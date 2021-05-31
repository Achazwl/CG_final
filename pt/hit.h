#ifndef PT_HIT
#define PT_HIT

#include "../vecs/vector3f.h"
#include "../object/base.h"

constexpr F hiteps = 1e-3;

struct Hit {
    F t;
    const Object3D *o;
    Vec n;

    __device__ Hit(F t): t(t), o(nullptr), n() {}

    __device__ void set(F t, const Object3D *o, const Vec &n) {
        this->t = t;
        this->o= o;
        this->n = n;
    }
};

#endif // PT_HIT