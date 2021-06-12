#include "pt/camera.h"
#include "object/group.h"
#include "object/mesh.h"

struct Scene {
	Camera *cam;
	Group *group;

	explicit Scene() {
		// CornellBox();
		Room();
		group = new Group(spheres, triangles, revsurfaces, materials, material_ids);
	}

private: // CornellBox
	void CornellBox() {
		CornellCamera();

		CornellObject();
	}

	void CornellCamera() {
		int w = 1024, h = 768;
		Vec o(50,52,295.6);
		Vec _z= Vec(0,-0.042612,-1).normal();
		Vec x(w*.5135/h);
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 140;
		int subpixel = 2;
		int spp = 3000;

		cam = new Camera(o, x, y, _z, length, w, h, subpixel, spp);
	}

	void CornellObject() {
		// loadSphere(600, Vec(50, 681.33, 81.6), Material{Vec(12,12,12),  Vec(), Vec(), Refl::GLASS}); // light
		// loadSphere(10.5, Vec(30,10.5,93), Material{Vec(),Vec(0.45, 0.45, 0.45), Vec(1,1,1)*0.03, Refl::GLASS}); // left ball
		// loadSphere(10.5, Vec(70,10.5,93), Material{Vec(),Vec(0.15, 0.15, 0.15), Vec(1,1,1)*0.98, Refl::GLASS}); // right ball

		// MeshFile* mesh;
		// mesh = new MeshFile("mesh/cube.obj");
		// loadMesh(Vec(1, 0, 0), Vec(-1, 81.6, 170), mesh, Material{Vec(), Vec(.75, .25, .25), Vec(1,1,1)*0.02, Refl::GLASS}); // Left
		// loadMesh(Vec(99, 0, 0), Vec(1, 81.6, 170), mesh, Material{Vec(),Vec(.25,.25,.75), Vec(1,1,1)*0.02, Refl::GLASS}); // Right
		// loadMesh(Vec(1, 0, 170), Vec(98, 81.6, 1), mesh, Material{Vec(), Vec(.9,.75,.75), Vec(1,1,1)*0.02, Refl::GLASS}); // Back
		// loadMesh(Vec(1, 0, 0), Vec(98, 81.6, -1), mesh, Material{Vec(), Vec(.75,.75,.75), Vec(1,1,1)*0.02, Refl::GLASS}); // Front
		// loadMesh(Vec(1, 0, 0), Vec(98, -1, 170), mesh, Material{Vec(),Vec(.75,.75,.75), Vec(1,1,1)*0.01, Refl::GLASS, "images/wood.jpg"}); // Bottom
		// loadMesh(Vec(1, 81.6, 0), Vec(98, 1, 170), mesh, Material{Vec(), Vec(0, 0.9, 0), Vec(1,1,1)*0.02, Refl::GLASS}); // TOP

		// mesh = new MeshFile("mesh/tree.obj");
		// loadMesh(Vec(50, 20, 50), Vec(10, 10, 10), mesh, Material{Vec(), Vec(0, 0.9, 0), Vec(1,1,1)*0.02, Refl::GLASS}); // tree

		// loadRevSurface(Vec(50, 0, 50), 0.5, std::vector<Vec>{
		// 	Vec{10, 5},
		// 	Vec{20, 10},
		// 	Vec{30, 20},
		// 	Vec{20, 50},
		// 	Vec{10, 60},
		// }, Material{Vec(), Vec(0.9, 0.1, 0.1), Vec(1,1,1)*0.02, Refl::GLASS});
	}

private: // Room
	void Room() {
		RoomCamera();

		RoomObject();
	}

	void RoomCamera() {
		int w = 1024, h = 768;
		Vec o(210,40,-5);
		Vec _z= Vec(-1,0.04,0.15).normal();
		Vec x(0, 0, -w*.5135/h);
		Vec y = Vec::cross(_z, x).normal()*.5135; // TODO why not 0.5 test
		int length = 140;
		int subpixel = 2;
		int spp = 250;

		cam = new Camera(o, x, y, _z, length, w, h, subpixel, spp);
	}

	void RoomObject() {
		auto mesh = new MeshFile("fireplace_room");
		loadMesh(Vec(0, 20, 50), Vec(1, 1, 1)*20, mesh); // room
	}

private:
	std::vector<Sphere> spheres;
	std::vector<Triangle> triangles;
	std::vector<RevSurface> revsurfaces;
	std::vector<Material> materials;
	std::vector<int> material_ids;

	void loadSphere(F radius, const Vec &o, const Material& material) {
		spheres.emplace_back(
			radius,
			o
		);
		materials.push_back(material);
		material_ids.push_back(materials.size()-1);
	}

	void loadMesh(const Vec &offset, const Vec &scale, MeshFile *mesh) { // TODO const Material&
		auto tf_v = [scale, offset](const Vec &v) { return v * scale + offset; };
		for (int i = 0; i < mesh->triangles.size(); ++i) {
			const auto& tri = mesh->triangles[i];
			triangles.emplace_back(
				tf_v(tri.v[0]), tf_v(tri.v[1]), tf_v(tri.v[2]), 
				tri.vn[0], tri.vn[1], tri.vn[2], // TODO correct vn tranform
				tri.vt[0], tri.vt[1], tri.vt[2] 
			);
			material_ids.emplace_back(
				materials.size() + mesh->material_ids[i]
			);
		}
		for (const auto& mat : mesh->materials) {
			materials.emplace_back(mat);
		}
	}

	void loadRevSurface(const Vec &offset, F scale, const std::vector<Vec> controls, const Material& material) {
		revsurfaces.emplace_back(
			offset, scale, controls
		);
		materials.push_back(material);
		material_ids.push_back(materials.size()-1);
	}
};
