#ifndef OBJ_BASE
#define OBJ_BASE

#include <memory>

class Object3D;
#include "../pt/ray.h"
#include "../pt/hit.h"
#include "../vecs/vector3f.h"
#include "bbox.h"
#include "material.h"

class Object3D { // Base class for all 3d entities.
public:
    Object3D(Material *material) : material(material) {}
    virtual ~Object3D() = default;

    virtual bool intersect(const Ray &ray, Hit &hit) const = 0;

    virtual Vec getColor(const Vec &p) const {
        return material->col;
    }

public:
    Bound bound;
    Material *material;
};

#endif // OBJ_BASE