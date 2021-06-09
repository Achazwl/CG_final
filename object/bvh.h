#ifndef OBJ_BVH
#define OBJ_BVH

#include "triangle.h"

struct Node {
	Bound bound;
	int id;
    Triangle* obj;
	Node *lc, *rc, *p;

	Node* to() const {
		Node* node = new Node(*this);
		cudaMalloc((void**)&node->obj, sizeof(Triangle));
		// cudaMalloc((void**)&node->lc, sizeof(Node));
		// cudaMalloc((void**)&node->rc, sizeof(Node));
		if (id >= 0) {
			cudaMemcpy(node->obj, obj, sizeof(Triangle), cudaMemcpyHostToDevice);
		}
		else {
			node->lc = lc->to();
			node->rc = rc->to();
			// cudaMemcpy(node->lc, lc->to(), sizeof(Node), cudaMemcpyHostToDevice);
			// cudaMemcpy(node->rc, rc->to(), sizeof(Node), cudaMemcpyHostToDevice);
		}

		Node* device;
		cudaMalloc((void**)&device, sizeof(Node));
		cudaMemcpy(device, node, sizeof(Node), cudaMemcpyHostToDevice);
		return device;
	}
};

static constexpr int num_bucket = 12;
struct Bucket {
	int count = 0;
	double cost;
	Bound bound;
};

class BVH { // based on SAH evaluation
public: // TODO protected
	Node* root;
public:
	BVH() { }
	BVH(Triangle* tris, int num) {
		std::vector<std::pair<Triangle*, int>> objs(num);
		for (int i = 0; i < num; ++i) 
			objs[i] = std::pair<Triangle*, int>{tris+i, i};
		root = build(objs.begin(), objs.end(), 0, num);
	}

	__device__ void debug() const {
	}

	__device__ int intersect(const Ray &ray, Hit &hit) const {
		int id = -1;
		Node* stack[32]; int top;
		stack[top = 0] = root;
        while (true) {
            Node* node = stack[top];
            if (!node->bound.intersect(ray)) {
				if (top-- == 0) break;
				stack[top] = stack[top]->rc;
				continue;
			}
            if (node->id >= 0) {
				if (node->obj->intersect(ray, hit)) 
					id = node->id;
				if (top-- == 0) break;
				stack[top] = stack[top]->rc;
			}
			else {
				stack[++top] = node->lc;
			}
        }
		return id;
	}

	BVH* to() const {
		BVH* bvh = new BVH();
		bvh->root = root->to();
		// cudaMalloc((void**)&bvh->root, sizeof(Node));
		// cudaMemcpy(bvh->root, root->to(), sizeof(Node), cudaMemcpyHostToDevice);

		BVH* device;
		cudaMalloc((void**)&device, sizeof(BVH));
		cudaMemcpy(device, bvh, sizeof(BVH), cudaMemcpyHostToDevice);
		return device;
	}

private:
	using ITER = std::vector<std::pair<Triangle*, int>>::iterator;
	Node* build(ITER bg, ITER ed, int l, int r) {
		Node* node = new Node();
		for (ITER it = bg; it != ed; ++it) {
			node->bound = node->bound + it->first->bound;
		}
		if (r-l <= 1) {
			node->obj = bg->first;
			node->id = bg->second;
			node->lc = node->rc = nullptr;
		}
        else {
			Bound centerBound;
			for (ITER it = bg; it != ed; ++it) { 
				centerBound = centerBound + it->first->bound.center();
			}
			int dir = centerBound.maxdim();

			Bucket buckets[num_bucket]{};
			for (ITER it = bg; it != ed; ++it) {
				int b = num_bucket * centerBound.offset(it->first->bound.center())[dir];
                b = clamp(b, 0, num_bucket-1);
				buckets[b].count++;
				buckets[b].bound = buckets[b].bound + it->first->bound;
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
				if (countl > 0 && countr > 0)
					buckets[i].cost = .125f + (countl * bl.surfaceArea() + countr * br.surfaceArea()) / node->bound.surfaceArea();
				else
					buckets[i].cost = inf;
			}
			double mincost = buckets[0].cost;
			int where = 0;
			for (int i = 1; i < num_bucket - 1; ++i) {
				if (buckets[i].cost	< mincost) {
					mincost = buckets[i].cost;
					where = i;
				}
			}
			ITER sep;
			if (buckets[where].cost == inf) {
				sep = bg + (ed-bg)/2;
			}
			else {
				sep = std::partition(bg, ed, [&centerBound, where, dir](const auto& obj) {
					int b = num_bucket * centerBound.offset(obj.first->bound.center())[dir];
					b = clamp(b, 0, num_bucket-1);
					return b <= where;
				});
			}
            node->obj = nullptr;
			node->id = -1; // change to -1-l if debug needed
			node->lc = build(bg, sep, l, l+sep-bg);
			node->rc = build(sep, ed, l+sep-bg, r);
		}
		return node;
	}
};

#endif // OBJ_KDTREE