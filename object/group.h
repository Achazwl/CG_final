#ifndef OBJ_GROUP
#define OBJ_GROUP

#include <iostream>
#include <vector>
#include "base.h"
#include "sphere.h"
#include "../pt/ray.h"

class Group {
public:
	Group(const std::vector<Object3D*> &objs): objs(objs) {}

	inline bool intersect(const Ray &r, Hit &hit) const { // intersect with group
		bool hav = false;
		for (int i = 0; i < objs.size(); ++i)
			hav |= objs[i]->intersect(r, hit);
		return hav;
	}

	Object3D* operator [] (int i) const { return objs[i]; }

protected:
	std::vector<Object3D*> objs;
};

#endif // OBJECT_GROUP