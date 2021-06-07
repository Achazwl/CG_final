#ifndef PT_HIT
#define PT_HIT

#include "../vecs/vector3f.h"
#include "../object/material.h"

constexpr F hiteps = 1e-3;

struct Hit {
    F t;
    Material *m;
    F u, v;
    Vec n;

    __device__ Hit(F t): t(t), n() {}

    __device__ void set(F t, const Vec &n, F u=-1, F v=-1) {
        this->t = t;
        this->n = n;
        this->u = u, this->v = v;
    }

    __device__ void setm(Material *m) {
        this->m = m;
    }
};

#endif // PT_HIT