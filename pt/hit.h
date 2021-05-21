#ifndef PT_HIT
#define PT_HIT

#include "../vecs/vector3f.h"
#include "../object/material.h"

struct Hit {
    float t;
    Material *m;
    Vec n;

    Hit(double t): t(t), m(nullptr), n() {}
    ~Hit() = default;

    void set(double t, Material *m, const Vec &n) {
        this->t = t;
        this->m = m;
        this->n = n;
    }
};

#endif // PT_HIT