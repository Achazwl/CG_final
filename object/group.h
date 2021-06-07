#ifndef OBJ_GROUP
#define OBJ_GROUP

#include "sphere.h"
#include "triangle.h"
#include "material.h"
#include <vector>
#include <cuda.h>

class Group {
public:
	__host__ Group() {}
	__host__ Group(
		const std::vector<Sphere> &spheres, 
		const std::vector<Triangle> &triangles, 
		const std::vector<Material> &materials)
	{
		sphs = new Sphere[num_sph = spheres.size()];
		for (int i = 0; i < num_sph; ++i) sphs[i] = spheres[i];
		tris = new Triangle[num_tri = triangles.size()];
		for (int i = 0; i < num_tri; ++i) tris[i] = triangles[i];
		mats = new Material[num_mat = materials.size()];
		for (int i = 0; i < num_mat; ++i) mats[i] = materials[i];
	}

	__device__ bool intersect(const Ray &ray, Hit &hit) const {
		int id = -1;
        for (int i = 0; i < num_sph; ++i) {
            if (sphs[i].intersect(ray, hit)) {
				id = i;
			}
		}
        for (int i = 0; i < num_tri; ++i) {
            if (tris[i].intersect(ray, hit)) {
				id = i + num_sph;
			}
		}
		if (id != -1) {
			hit.setm(&mats[id]);
			return true;
		}
		return false;
	}

	__host__ Group* to() const {
		Group *group = new Group();
		group->num_sph = num_sph;
		cudaMalloc((void**)&group->sphs, num_sph*sizeof(Sphere));
		cudaMemcpy(group->sphs, sphs, num_sph*sizeof(Sphere), cudaMemcpyHostToDevice);
		group->num_tri = num_tri;
		cudaMalloc((void**)&group->tris, num_tri*sizeof(Triangle));
		cudaMemcpy(group->tris, tris, num_tri*sizeof(Triangle), cudaMemcpyHostToDevice);
		group->num_mat = num_mat;
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
	int num_tri; Triangle *tris;
	int num_mat; Material *mats;
};

#endif // OBJECT_GROUP