#ifndef OBJ_BASE
#define OBJ_BASE

#include <memory>

#include "../pt/ray.h"
#include "../pt/hit.h"
#include "material.h"

// Base class for all 3d entities.
class Object3D {
public:
    Object3D(Material *material) : material(material) {}
    virtual ~Object3D() = default;

    virtual bool intersect(const Ray &ray, Hit &hit) const = 0;

public: // TODO protected
    Material *material;
};

#endif // OBJ_BASE