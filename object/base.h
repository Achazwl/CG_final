#ifndef OBJ_BASE
#define OBJ_BASE

class Object3D;
#include "../pt/ray.h"
#include "../pt/hit.h"
#include "../vecs/vector3f.h"
#include "bbox.h"
#include "material.h"
#include <memory>

class Object3D { // Base class for all 3d entities.
public:
    __device__ __host__ Object3D(Material *material) : material(material) {}

    virtual __device__ bool intersect(const Ray &ray, Hit &hit) const {
        printf("holy shit\n");
        return false;
    }

    virtual __device__ Vec getColor(const Vec &p) const {
        return material->Kd;
    }

public:
    Bound bound;
    Material *material;
};

#endif // OBJ_BASE