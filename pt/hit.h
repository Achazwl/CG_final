#ifndef PT_HIT
#define PT_HIT

#include "../vecs/vector3f.h"
#include "../object/material.h"

constexpr F hiteps = 1e-4;

struct Hit {
    F t;
    Material *m;
    Tex tex;
    Vec n, pu, pv;

    __device__ Hit(F t): t(t), n() {}

    __device__ void set(F t, const Vec &n, Tex tex = Tex(), Vec pu = Vec(), Vec pv = Vec()) {
        this->t = t;
        this->n = n;
        this->tex = tex;
        this->pu =pu;
        this->pv = pv;
    }

    __device__ void setm(Material *m) {
        this->m = m;
    }
};

#endif // PT_HIT