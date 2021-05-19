#ifndef OBJ_GROUP
#define OBJ_GROUP

#include <iostream>
#include <vector>
#include "base.h"
#include "sphere.h"
#include "refl.h"
#include "../pt/ray.h"

class Group {
public:
	Group(const std::vector<Object3D*> &objs): objs(objs) {}

	inline bool intersect(const Ray &r, double &t, int &id) const { // intersect with group
		static constexpr double inf=1e20; 
		t = inf;
		for (int i = 0; i < objs.size(); ++i)
			if (objs[i]->intersect(r, t)) id = i;
		return t < inf; 
	} 

	int size() const { return objs.size(); }
	Object3D* operator [] (int i) const { return objs[i]; }

protected:
	std::vector<Object3D*> objs;
};

#endif // OBJECT_GROUP