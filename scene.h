#include "pt/camera.h"
#include "object/group.h"
#include "object/mesh.h"

struct Scene {
	Camera *cam;
	Group *group;

	explicit Scene() {
		initCamera();

		initObject();
		group = new Group(spheres, triangles, materials);
	}

private:
	void initCamera() {
		int w = 1024, h = 768;
		Vec o(50,52,295.6); // o
		Vec _z= Vec(0,-0.042612,-1).normal(); // -z
		Vec x(w*.5135/h); // x
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 140;
		int subpixel = 2;
		int spp = 20;

		cam = new Camera(o, x, y, _z, length, w, h, subpixel, spp);
	}

	void initObject() {
		loadSphere(600, Vec(50, 681.33, 81.6), Material{Vec(12,12,12),  Vec(), Vec(), Refl::GLASS}); // light
		loadSphere(10.5, Vec(30,10.5,93), Material{Vec(),Vec(0.45, 0.45, 0.45), Vec(1,1,1)*0.03, Refl::GLASS}); // left ball
		loadSphere(10.5, Vec(70,10.5,93), Material{Vec(),Vec(0.15, 0.15, 0.15), Vec(1,1,1)*0.98, Refl::GLASS}); // right ball
		MeshFile mesh("mesh/cube.obj");
		loadMesh(Vec(1, 0, 0), Vec(-1, 81.6, 170), &mesh, Material{Vec(), Vec(.75, .25, .25), Vec(1,1,1)*0.02, Refl::GLASS}); // Left
		loadMesh(Vec(99, 0, 0), Vec(1, 81.6, 170), &mesh, Material{Vec(),Vec(.25,.25,.75), Vec(1,1,1)*0.02, Refl::GLASS}); // Right
		loadMesh(Vec(1, 0, 0), Vec(98, 81.6, -1), &mesh, Material{Vec(), Vec(.75,.75,.75), Vec(1,1,1)*0.02, Refl::GLASS}); // Back
		loadMesh(Vec(1, 0, 170), Vec(98, 81.6, 1), &mesh, Material{Vec(), Vec(.9,.75,.75), Vec(1,1,1)*0.02, Refl::GLASS}); // Front
		loadMesh(Vec(1, 0, 0), Vec(98, -1, 170), &mesh, Material{Vec(),Vec(.75,.75,.75), Vec(1,1,1)*0.04, Refl::GLASS}); // Bottom
		loadMesh(Vec(1, 81.6, 0), Vec(98, 1, 170), &mesh, Material{Vec(), Vec(0, 0.9, 0), Vec(1,1,1)*0.02, Refl::GLASS}); // TOP
	}

private:
	std::vector<Sphere> spheres;
	std::vector<Triangle> triangles;
	std::vector<Material> materials;

	void loadSphere(F radius, const Vec &o, const Material &material) {
		spheres.push_back(Sphere{
			radius,
			o
		});
		materials.push_back(material);
	}

	void loadMesh(const Vec &offset, const Vec &scale, MeshFile *mesh, const Material &material) {
		for (auto& tri:  mesh->t) {
			triangles.push_back(Triangle{
				mesh->v[tri[0]] * scale + offset,
				mesh->v[tri[1]] * scale + offset,
				mesh->v[tri[2]] * scale + offset,
			});
			materials.push_back(material);
		}
	}
};
