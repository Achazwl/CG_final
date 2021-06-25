#include "pt/camera.h"
#include "object/group.h"
#include "object/mesh.h"

struct Scene {
	Camera *cam;
	Group *group;

	explicit Scene() {
		CornellBox();
		// Room();
		// Sponza();
		// Bunny();
		// TexBall();

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
		Vec _z= Vec(-0.01,-0.042612,-1).normal();
		Vec x(w*.5135/h, 0, 0);
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 140;
		int focus = 200;
		int subpixel = 2;
		int spp = 100;

		cam = new Camera(o, x, y, _z, length, focus, w, h, subpixel, spp);
	}

	void CornellObject() {
		loadSphere(600, Vec(50, 681.33, 81.6), Material{Vec(1,1,1)*3,  Vec(), Vec(), 0, Refl::GLASS}); // light
		loadSphere(10.5, Vec(30,10.5,93), Material{Vec(),Vec(0.45, 0.45, 0.45), Vec(1,1,1)*0.03, 0, Refl::GLASS}); // left ball
		loadSphere(10.5, Vec(70,10.5,73), Material{Vec(),Vec(0.15, 0.15, 0.15), Vec(1,1,1)*0.98, 0, Refl::GLASS}); // right ball

		MeshFile* mesh;
		mesh = new MeshFile("cube");
		loadMesh(Vec(50, 82.1, 85), Vec(98, 1, 170), mesh); // Top
		loadMesh(Vec(50, -0.5, 85), Vec(98, 1, 170), mesh); // Bottom
		loadMesh(Vec(50, 40.8, -0.5), Vec(98, 81.6, 1), mesh); // Front
		loadMesh(Vec(0.5, 40.8, 85), Vec(1, 81.6, 170), mesh); // Left
		loadMesh(Vec(99.5, 40.8, 85), Vec(1, 81.6, 170), mesh); // Right

		mesh = new MeshFile("cube_bump");

		mesh = new MeshFile("tree");
		loadMesh(Vec(50, 20, 50), Vec(10, 10, 10), mesh); // tree

		loadRevSurface(Vec(50, 0, 50), 0.5, std::vector<Vec>{
			Vec{10, 5, 0},
			Vec{20, 10, 0},
			Vec{30, 20, 0},
			Vec{20, 50, 0},
			Vec{10, 60, 0},
		}, Material{Vec(), Vec(0.9, 0.1, 0.1), Vec(1,1,1)*0.02, 0, Refl::GLASS, "images/vase.png"});
	}

private: // Room
	void Room() {
		RoomCamera();

		RoomObject();
	}

	void RoomCamera() {
		int w = 1024, h = 768;
		Vec o(210,60,-5);
		Vec _z= Vec(-1,-0.1,0.15).normal();
		Vec x(0, 0, -w*.5135/h);
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 140;
		int focus = 160;
		int subpixel = 2;
		int spp = 20;

		cam = new Camera(o, x, y, _z, length, focus, w, h, subpixel, spp);
	}

	void RoomObject() {
		auto mesh = new MeshFile("fireplace_room");
		loadMesh(Vec(0, 20, 50), Vec(1, 1, 1)*20, mesh); // room
	}

private: // Sponza
	void Sponza() {
		SponzaCamera();

		SponzaObject();
	}

	void SponzaCamera() {
		int w = 1024, h = 768;
		Vec o(180, 5, -10);
		Vec _z= Vec(-1, 0.2, 0.2).normal();
		Vec x = Vec(0, 0, -(F)w/h)*.5135;
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 40;
		int focus = 40;
		int subpixel = 2;
		int spp = 20;

		cam = new Camera(o, x, y, _z, length, focus, w, h, subpixel, spp);
	}

	void SponzaObject() {
		loadSphere(100, Vec(0, 400, -200), Material{Vec(1,1,1)*30,  Vec(), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(5, Vec(100 , 10, 20), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(5, Vec(70  , 10, 20), Material{Vec(),  Vec(0.25, 0.85, 0.25), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(5, Vec(40  , 10, 20), Material{Vec(),  Vec(0.25, 0.25, 0.85), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(5, Vec(10  , 10, 20), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(5, Vec(-20 , 10, 20), Material{Vec(),  Vec(0.25, 0.85, 0.25), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(5, Vec(-50 , 10, 20), Material{Vec(),  Vec(0.25, 0.25, 0.85), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(5, Vec(-70 , 10, 20), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(5, Vec(-100, 10, 20), Material{Vec(),  Vec(0.25, 0.85, 0.25), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(5, Vec(-130, 10, 20), Material{Vec(),  Vec(0.25, 0.25, 0.85), Vec(), 0, Refl::GLASS}); // sun
		auto mesh = new MeshFile("sponza");
		loadMesh(Vec(0, 0, 0), Vec(1, 1, 1)*10, mesh); // room
	}

private: // Bunny
	void Bunny() {
		BunnyCamera();

		BunnyObject();
	}

	void BunnyCamera() {
		int w = 1024, h = 768;
		// Vec o(210,40,-5);
		// Vec _z= Vec(-1,0.04,0.15).normal();
		Vec o(0, 10, 100);
		Vec _z= Vec(0, 0, -1).normal();
		Vec x = Vec((F)w/h, 0, 0)*.5135;
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 40;
		int focus = 0;
		int subpixel = 2;
		int spp = 2000; // 18000; // 3h

		cam = new Camera(o, x, y, _z, length, focus, w, h, subpixel, spp);
	}

	void BunnyObject() {
		loadSphere(50, Vec(0, 400, 200), Material{Vec(1,1,1)*50,  Vec(), Vec(), 0, Refl::GLASS}); // sun
		auto mesh = new MeshFile("bunny");
		loadMesh(Vec(0, 0, 0), Vec(1, 1, 1)*10, mesh); // room
	}

private: // TexBall
	void TexBall() {
		TexBallCamera();

		TexBallObject();
	}

	void TexBallCamera() {
		int w = 1024, h = 768;
		Vec o(-20,52,-190);
		Vec _z= Vec(0,-0.042612,1).normal();
		Vec x(-w*.5135/h, 0, 0);
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 40;
		int focus = 320;
		int subpixel = 2;
		int spp = 20;

		cam = new Camera(o, x, y, _z, length, focus, w, h, subpixel, spp);
	}

	void TexBallObject() {
		loadSphere(100, Vec(0, 400, -200), Material{Vec(1,1,1)*50,  Vec(), Vec(), 0, Refl::GLASS}); // sun
		loadSphere(10, Vec(-100, 10,  420), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec(-90, 10,  390), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec(-80, 10,  360), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec(-70, 10,  330), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec(-60, 10,  300), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec(-50, 10,  270), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec(-40, 10,  240), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec(-30, 10,  210), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec(-20, 10,  180), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec(-10, 10,  150), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec( 0 , 10,  120), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec( 10, 10,  90 ), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec( 20, 10,  60 ), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		loadSphere(10, Vec( 30, 10,  30 ), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::GLASS}); // ball
		auto mesh = new MeshFile("texball");
		loadMesh(Vec(100, 0, 0), Vec(1, 1, 1)*50, mesh); // room
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

	void loadMesh(const Vec &offset, const Vec &scale, MeshFile *mesh) {
		auto tf_v = [scale, offset](const Vec &v) { return v * scale + offset; };
		for (int i = 0; i < mesh->triangles.size(); ++i) {
			const auto& tri = mesh->triangles[i];
			triangles.emplace_back(
				tf_v(tri.v[0]), tf_v(tri.v[1]), tf_v(tri.v[2]), 
				tri.vn[0], tri.vn[1], tri.vn[2],
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
