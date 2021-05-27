#ifndef OBJ_GROUP
#define OBJ_GROUP

#include <iostream>
#include <vector>
#include "sphere.h"
#include "triangle.h"
#include "mesh.h"

struct BVH {
	Bound bound;
	std::vector<Object3D*> objs;
	BVH *lc, *rc;
};

static constexpr int num_bucket = 12;
struct Bucket {
	int count = 0;
	F cost;
	Bound bound;
};

class Group {
	static const int lim = 4; // TODO: bigger for balancing
public:
	Group(const std::vector<Object3D*> &objs) {
		for (auto& obj: objs) {
			// if (dynamic_cast<Mesh*>(obj)) { // TODO: temparary comment
			// 	Mesh *mesh = dynamic_cast<Mesh*>(obj);
			// 	for (auto& triangle: mesh->triangles) {
			// 		this->objs.emplace_back(triangle);
			// 	}
			// }
			// else {
				this->objs.emplace_back(obj);
			// }
		}
		root = build(this->objs);
	}

	BVH* build(std::vector<Object3D*> objs) {
		BVH *node = new BVH();
		for (auto& obj : objs) 
			node->bound = node->bound + obj->bound;
		if (objs.size() <= lim) {
			node->objs = objs;
			node->lc = node->rc = nullptr;
		} else {
			Bound centerBound;
			for (auto& obj : objs) 
				centerBound = centerBound + obj->bound.center();
			int dir = centerBound.maxdim();

			Bucket buckets[num_bucket]{};
			for (auto& obj: objs) {
				int b = num_bucket * centerBound.offset(obj->bound.center())[dir];
				b = std::min(std::max(b, 0), num_bucket-1); // clamp
				buckets[b].count++;
				buckets[b].bound = buckets[b].bound + obj->bound;
			}
			for (int i = 0; i < num_bucket - 1; ++i) {
				Bound bl, br;
				int countl = 0, countr = 0;
				for (int j = 0; j <= i; ++j) {
					bl = bl + buckets[j].bound;
					countl += buckets[j].count;
				}
				for (int j = i+1; j < num_bucket; ++j) {
					br = br + buckets[j].bound;
					countr += buckets[j].count;
				}
				buckets[i].cost = 1 + (countl * bl.surfaceArea() + countr * br.surfaceArea()) / node->bound.surfaceArea();
			}
			F mincost = buckets[0].cost;
			int where = 0;
			for (int i = 1; i < num_bucket - 1; ++i) {
				if (buckets[i].cost	< mincost) {
					mincost = buckets[i].cost;
					where = i;
				}
			}

			auto sep = std::partition(objs.begin(), objs.end(), [&centerBound, where, dir](const auto& obj) {
				int b = num_bucket * centerBound.offset(obj->bound.center())[dir];
				b = std::min(std::max(b, 0), num_bucket-1); // clamp
				return b <= where;
			});
			node->lc = build(std::vector<Object3D*>(objs.begin(), sep));
			node->rc = build(std::vector<Object3D*>(sep, objs.end()));
		}
		return node;
	}

	bool intersect(const Ray &ray, Hit &hit) const {
        bool hav = false;
        for (auto& obj: objs)
            hav |= obj->intersect(ray, hit);
        return hav;
		// if (!root) return false; // TODO: temparar comment
		// return BVHintersect(root, ray, hit);
	}

	bool BVHintersect(BVH *node, const Ray &ray, Hit &hit) const {
		bool hav = false;
		if (!node->objs.empty()) { // leaf
			for (const auto& obj: objs) {
				hav |= obj->intersect(ray, hit);
			}
			return hav;
		}
		if (!node->bound.intersect(ray)) {
			return false;
		}
		hav |= BVHintersect(node->lc, ray, hit);
		hav |= BVHintersect(node->rc, ray, hit);
		return hav;
	}

	Object3D* operator [] (int i) const { return objs[i]; }

protected:
	std::vector<Object3D*> objs;
	std::vector<Vec> pts;
	BVH* root;
};

#endif // OBJECT_GROUP