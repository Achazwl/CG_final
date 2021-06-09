#ifndef OBJ_AABB
#define OBJ_AABB
#include "base.h"

struct Bound {
    Vec mn, mx;
    Bound() : mn(inf, inf, inf), mx(-inf, -inf, -inf) { }
    Bound(Vec vec) : mn(vec), mx(vec) { }
    Bound(Vec mn, Vec mx) : mn(mn), mx(mx) { }

    Vec center() const {
        return (mn + mx) / 2;
    }

    int maxdim() const {
        return (mx - mn).argmax();
    }

    F surfaceArea() const {
        Vec d = mx - mn;
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    Vec offset(const Vec &p) const {
        return (p - mn) / (mx - mn);
    }

    Bound operator + (const Bound &rhs) const {
        return Bound(Vec::min(mn, rhs.mn), Vec::max(mx, rhs.mx));
    }

    __device__ bool intersect(const Ray &ray) const {
        Vec enter = (mn - Vec(1,1,1) - ray.o) / ray.d; // pm (1,1,1) to make sure it is a cube
        Vec exit = (mx + Vec(1,1,1) - ray.o) / ray.d;
        for (int i = 0; i < 3; ++i) {
            if (ray.d[i] < 0) {
                swap(enter[i], exit[i]);
            }
        }
        F in = enter.max(), out = exit.min();
        return in < out && out > eps; // maybe ray origin is inside the box, thus in < 0 but out >= 0
    } 
};

#endif // OBJ_AABB