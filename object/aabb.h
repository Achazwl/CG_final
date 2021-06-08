#ifndef OBJ_AABB
#define OBJ_AABB
#include "base.h"

struct Bound {
    Vec mn, mx;
    Bound() : mn(inf, inf, inf), mx(inf, -inf, -inf) { }
    Bound(Vec vec) : mn(vec), mx(vec) { }
    Bound(Vec mn, Vec mx) : mn(mn), mx(mx) { }

    __device__ Vec center() const {
        return (mn + mx) / 2;
    }

    __device__ int maxdim() const {
        return (mx - mn).argmax();
    }

    __device__ F surfaceArea() const {
        Vec d = mx - mn;
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    __device__ Vec offset(const Vec &p) const {
        return (p - mn) / (mx - mn);
    }

    __device__ Bound operator + (const Bound &rhs) const {
        return Bound(Vec::min(mn, rhs.mn), Vec::max(mx, rhs.mx));
    }

    __device__ bool intersect(const Ray &ray) const {
        Vec enter = (mn - ray.o) / ray.d;
        Vec exit = (mx - ray.o) / ray.d;
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