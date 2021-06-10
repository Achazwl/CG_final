#ifndef OBJ_GROUP
#define OBJ_GROUP

#include "sphere.h"
#include "bvh.h"
#include "revsurface.h"
#include "material.h"
#include <vector>
#include <cuda.h>

class Group {
public:
	Group() {}
	Group(
		const std::vector<Sphere> &spheres, 
		const std::vector<Triangle> &triangles, 
		const std::vector<RevSurface> &revsurfaces, 
		const std::vector<Material> &materials)
	{
		sphs = new Sphere[num_sph = spheres.size()];
		for (int i = 0; i < num_sph; ++i) sphs[i] = spheres[i];
		
		tris = new Triangle[num_tri = triangles.size()];
		for (int i = 0; i < num_tri; ++i) tris[i] = triangles[i];
		bvh = new BVH(tris, num_tri);

		revs = new RevSurface[num_rev = revsurfaces.size()];
		for (int i = 0; i < num_rev; ++i) revs[i] = revsurfaces[i];

		mats = new Material[num_mat = materials.size()];
		for (int i = 0; i < num_mat; ++i) mats[i] = materials[i];
	}

	__device__ bool intersect(const Ray &ray, Hit &hit) const {
		int id = -1;
        for (int i = 0; i < num_sph; ++i) {
            if (sphs[i].intersect(ray, hit)) id = i;
		}
        // for (int i = 0; i < num_tri; ++i) {
        //     if (tris[i].intersect(ray, hit)) id = i + num_sph;
		// }
		{ // tri
			int tid = bvh->intersect(ray, hit);
			if (tid != -1) id = tid + num_sph;
		}
        for (int i = 0; i < num_rev; ++i) {
            if (revs[i].intersect(ray, hit)) id = i + num_sph + num_tri;
		}
		if (id != -1) {
			hit.setm(&mats[id]);
			return true;
		}
		return false;
	}

	Group* to() const {
		Group *group = new Group(*this);

		cudaMalloc((void**)&group->sphs, num_sph*sizeof(Sphere));
		cudaMemcpy(group->sphs, sphs, num_sph*sizeof(Sphere), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&group->tris, num_tri*sizeof(Triangle));
		cudaMemcpy(group->tris, tris, num_tri*sizeof(Triangle), cudaMemcpyHostToDevice);

		group->bvh = bvh->to();

		cudaMalloc((void**)&group->revs, num_rev*sizeof(RevSurface));
		for (int i = 0; i < num_rev; ++i) {
			cudaMemcpy(group->revs+i, revs[i].to(), sizeof(RevSurface), cudaMemcpyHostToDevice);
		}

		cudaMalloc((void**)&group->mats, num_mat*sizeof(Material));
		for (int i = 0; i < num_mat; ++i) {
			cudaMemcpy(group->mats+i, mats[i].to(), sizeof(Material), cudaMemcpyHostToDevice);
		}

		Group *device;
		cudaMalloc((void**)&device, sizeof(Group));
		cudaMemcpy(device, group, sizeof(Group), cudaMemcpyHostToDevice);
		return device;
	}

public: // TODO protected
	int num_sph; Sphere *sphs;
	int num_tri; Triangle *tris; BVH* bvh;
	int num_rev; RevSurface *revs;
	int num_mat; Material *mats;
};

#endif // OBJECT_GROUP