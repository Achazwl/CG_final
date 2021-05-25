#ifndef OBJ_BASE
#define OBJ_BASE

#include <memory>

#include "../pt/ray.h"
#include "../pt/hit.h"
#include "../vecs/vector3f.h"
#include "material.h"

struct Bound {
    static constexpr double minval = std::numeric_limits<double>::lowest();
    static constexpr double maxval = std::numeric_limits<double>::max();
    Vec mn, mx;
    Bound() : mn(maxval, maxval, maxval), mx(minval, minval, minval) { }
    Bound(Vec vec) : mn(vec), mx(vec) { }
    Bound(Vec mn, Vec mx) : mn(mn), mx(mx) { }

    Vec center() const {
        return (mn + mx) / 2;
    }

    int maxdim() const {
        return (mx - mn).argmax();
    }

    double surfaceArea() const {
        Vec d = mx - mn;
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    Vec offset(const Vec &p) const {
        return (p - mn) / (mx - mn);
    }

    Bound operator + (const Bound &rhs) const {
        using namespace std;
        return Bound(
            Vec(min(mn.x, rhs.mn.x), min(mn.y, rhs.mn.y), min(mn.z, rhs.mn.z)),
            Vec(max(mx.x, rhs.mx.x), max(mx.y, rhs.mx.y), max(mx.z, rhs.mx.z))
        );
    }

    bool intersect(const Ray &ray) const {
        Vec enter = (mn - ray.o) / ray.d;
        Vec exit = (mx - ray.o) / ray.d;
        for (int i = 0; i < 3; ++i) {
            if (ray.d[i] < 0)
                std::swap(enter[i], exit[i]);
        }
        double in = enter.max(), out = exit.min();
        return in < out && out >= 0; // maybe ray origin is inside the box, thus in < 0 but out >= 0
    } 
};

class Object3D { // Base class for all 3d entities.
public:
    Object3D(Material *material) : material(material) {}
    virtual ~Object3D() = default;

    virtual bool intersect(const Ray &ray, Hit &hit) const = 0;

public:
    Bound bound;
    Material *material;
};

#endif // OBJ_BASE