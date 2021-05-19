#ifndef OBJ_BASE
#define OBJ_BASE

#include "../pt/ray.h"
#include "refl.h"

// Base class for all 3d entities.
class Object3D {
public:
    Object3D(Vec e, Vec col, Refl refl): e(e), col(col), refl(refl) {}
    virtual bool intersect(const Ray &ray, double &t) const = 0;
public: // TODO protected
	Vec e, col;
	Refl refl;
};

#endif // OBJ_BASE