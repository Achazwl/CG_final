#include "pt/camera.h"
#include "object/group.h"
#include "object/mesh.h"

static constexpr int numtri = 3;

struct Scene {
	Camera *cam;
	Group *group;

	explicit Scene() {
		CornellBox();
		// Room();
		// Sponza();
		// Bunny();
		// TexBall();
		// Final();

		group = new Group(spheres, triangles, revsurfaces, materials, material_ids);
	}

private: // Final
	void Final() {
		srand(33);

		FinalCamera();

		FinalObject();
	}

	void FinalCamera() {
		int w = 1024, h = 768;
		Vec o(750,300,700);
		Vec _z= Vec(-1,-0.3,-0.9).normal();
		Vec x = Vec(0.51, -0.05, -0.86) * (w*.5135/h);
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 100;
		int focus = 450;
		int subpixel = 2;
		int spp = 10000;

		cam = new Camera(o, x, y, _z, length, focus, w, h, subpixel, spp);
	}

	void FinalObject() {
		loadSphere(50, Vec(250, 570, 250), Material{Vec(1,1,1)*5,  Vec(), Vec(), 0, Refl::DIFFUSE}); // light
		loadSphere(50, Vec(670, 250, 250), Material{Vec(1,1,1)*5,  Vec(), Vec(), 0, Refl::DIFFUSE}); // light
		loadSphere(50, Vec(250, 250, 670), Material{Vec(1,1,1)*5,  Vec(), Vec(), 0, Refl::DIFFUSE}); // light
		
		loadSphere(30, Vec(380, 30, 120),
			Material{Vec(), Vec(), Vec(), 0, Refl::DIFFUSE, "images/volleyball.jpg", "images/volleybump.png"}); // Right ball
			
		loadSphere(30, Vec(380, 120, 380),
			Material{Vec(), Vec(1,1,1)*0.999, Vec(), 0, Refl::GLASS}); // glass ball

		// loadRevSurface(Vec(100, -5, 400), 1.5, std::vector<Vec>{ // Left vase
		// 	Vec{10, 5, 0},
		// 	Vec{20, 10, 0},
		// 	Vec{30, 20, 0},
		// 	Vec{20, 50, 0},
		// 	Vec{10, 60, 0},
		// 	Vec(20, 70, 0),
		// }, Material{Vec(), Vec(0.2, 0.8, 0.2), Vec(), 0, Refl::DIFFUSE, "images/vase.png"});

		MeshFile *mesh;
		// mesh = new MeshFile("tree");
		// loadMesh(Vec(100, 80, 400), Vec(1,1,1)*30, mesh); // tree

		mesh = new MeshFile("cube");
		loadMesh(Vec(-0.5, 250, 250), Vec(1, 500, 500), mesh); // Left
		loadMesh(Vec(250, 250, -0.5), Vec(500, 500, 1), mesh); // Right
		loadMesh(Vec(250, -0.5, 250), Vec(500, 1, 500), mesh, -1); // Bottom

		mesh = new MeshFile("cube");
		for (int i = 0; i < 15; ++i) 
		for (int j = 0; j < 15; ++j) {
			int K = rand() % (12 - (i+j)/3);
			for (int k = 0; k < K; ++k)
				loadMesh(Vec(i*20+10, k*20+10, j*20+10), Vec(18, 18, 18), mesh, 1);
		}

		for (int tri  = -2; tri < numtri; ++tri) {
			int kd = tri < 0 ? 1 : 0;
			int bump = rand() % 6;
			char s[53];
			sprintf(s, "mesh/cube_bump/normals%d.jpg", bump);
			materials.emplace_back(Material{
				Vec(),
				(tri >= 0) ? Vec::rand() : Vec(1,1,1)*0.9999,
				Vec(),
				0,
				kd == 0 ? Refl::DIFFUSE : (kd == 1 ? Refl::MIRROR : Refl::GLASS),
				nullptr,
				(bump == 0 || tri < 0) ? nullptr : s
			});
		}
	}

private: // CornellBox
	void CornellBox() {
		CornellCamera();

		CornellObject();
	}

	void CornellCamera() {
		int w = 1024, h = 768;
		Vec o(50,52,290);
		Vec _z= Vec(-0.01,-0.1,-1).normal();
		Vec x(w*.5135/h, 0, 0);
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 40;
		int focus = 100;
		int subpixel = 2;
		int spp = 20000; // 1h

		cam = new Camera(o, x, y, _z, length, focus, w, h, subpixel, spp);
	}

	void CornellObject() {
		MeshFile* mesh;

		loadSphere(600, Vec(50, 681.33, 81.6), Material{Vec(1,1,1)*3,  Vec(), Vec(), 0, Refl::DIFFUSE}); // light

		// loadSphere(10, Vec(80,10,133), Material{Vec(), Vec(), Vec(), 0, Refl::DIFFUSE, "images/volleyball.jpg"}); // left ball

		// loadSphere(10, Vec(30, 20, 120),
		// 	Material{Vec(), Vec(1,1,1)*0.999, Vec(), 0, Refl::GLASS}); // glass ball

		loadRevSurface(Vec(80, -2.5, 30), 0.5, std::vector<Vec>{
			Vec{10, 5, 0},
			Vec{20, 10, 0},
			Vec{30, 20, 0},
			Vec{20, 50, 0},
			Vec{10, 60, 0},
			Vec(20, 70, 0),
		}, Material{Vec(), Vec(0.2, 0.8, 0.2), Vec(), 0, Refl::DIFFUSE, "images/vase.png"});

		mesh = new MeshFile("tree");
		loadMesh(Vec(80, 30, 30), Vec(10, 10, 10), mesh); // tree

		mesh = new MeshFile("cube");
		loadMesh(Vec(50, 82.1, 85), Vec(98, 1, 170), mesh); // Top
		loadMesh(Vec(50, 40.8, -0.5), Vec(98, 81.6, 1), mesh); // Front
		loadMesh(Vec(0.5, 40.8, 85), Vec(1, 81.6, 170), mesh); // Left
		loadMesh(Vec(99.5, 40.8, 85), Vec(1, 81.6, 170), mesh); // Right
;
		mesh = new MeshFile("cube");
		loadMesh(Vec(50, -0.5, 85), Vec(98, 1, 170), mesh, -1); // Bottom

		// for (int i = 1; i < 6; ++i) 
		// for (int j = 0; j < 15; j += 4) {
		// 	int k;
		// 	int size = (9 - j / 2);
		// 	int u = size+1, v = u/2;
		// 	k = (i == 1 || i == 5) ? 3 : ((i == 2 || i == 4) ? 4 : 3);
		// 	loadMesh(Vec(i*u+v+size*(4-j/4), k*u+v, j*12+5), Vec(1,1,1)*size, mesh, 1);
		// 	k = (i == 1 || i == 5) ? 2 : ((i == 2 || i == 4) ? 1 : 0);
		// 	loadMesh(Vec(i*u+v+size*(4-j/4), k*u+v, j*12+5), Vec(1,1,1)*size, mesh, 1);
		// }

		for (int i = 1; i < 6; ++i) 
		for (int j = 12; j < 13; j += 4) {
			int k;
			int size = 9;
			int u = size+1, v = u/2;
			k = (i == 1 || i == 5) ? 3 : ((i == 2 || i == 4) ? 4 : 3);
			loadMesh(Vec(i*u+v, k*u+v, j*12-40), Vec(1,1,1)*size, mesh, 1);
			k = (i == 1 || i == 5) ? 2 : ((i == 2 || i == 4) ? 1 : 0);
			loadMesh(Vec(i*u+v, k*u+v, j*12-40), Vec(1,1,1)*size, mesh, 1);
		}

		for (int tri  = -2; tri < numtri; ++tri) {
			int kd = tri < 0 ? 1 : 0;
			int bump = rand() % 6;
			char s[53];
			sprintf(s, "mesh/cube_bump/normals%d.jpg", bump);
			materials.emplace_back(Material{
				Vec(),
				(tri >= 0) ? Vec::rand() : Vec(1,1,1)*0.9999,
				Vec(),
				0,
				kd == 0 ? Refl::DIFFUSE : (kd == 1 ? Refl::MIRROR : Refl::GLASS),
				nullptr,
				(bump == 0 || tri < 0) ? nullptr : s
			});
		}
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
		Vec o(180, 5, -30);
		Vec _z= Vec(-1, 0.2, 0.4).normal();
		Vec x = Vec(0, 0, -(F)w/h)*.5135;
		Vec y = Vec::cross(_z, x).normal()*.5135;
		int length = 40;
		int focus = 40;
		int subpixel = 2;
		int spp = 20;

		cam = new Camera(o, x, y, _z, length, focus, w, h, subpixel, spp);
	}

	void SponzaObject() {
		loadSphere(100, Vec(0, 400, -200), Material{Vec(1,1,1)*30,  Vec(), Vec(), 0, Refl::DIFFUSE}); // sun
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
		loadSphere(50, Vec(0, 400, 200), Material{Vec(1,1,1)*50,  Vec(), Vec(), 0, Refl::DIFFUSE}); // sun
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
		loadSphere(100, Vec(0, 400, -200), Material{Vec(1,1,1)*50,  Vec(), Vec(), 0, Refl::DIFFUSE}); // sun
		loadSphere(10, Vec(-100, 10, 420), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec(-90, 10,  390), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec(-80, 10,  360), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec(-70, 10,  330), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec(-60, 10,  300), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec(-50, 10,  270), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec(-40, 10,  240), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec(-30, 10,  210), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec(-20, 10,  180), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec(-10, 10,  150), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec( 0 , 10,  120), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec( 10, 10,  90 ), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec( 20, 10,  60 ), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
		loadSphere(10, Vec( 30, 10,  30 ), Material{Vec(),  Vec(0.85, 0.25, 0.25), Vec(), 0, Refl::DIFFUSE}); // ball
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

	void loadMesh(const Vec &offset, const Vec &scale, MeshFile *mesh, int userandom=0) {
		auto tf_v = [scale, offset](const Vec &v) { return v * scale + offset; };
		int fix = rand() % numtri;
		for (int i = 0; i < mesh->triangles.size(); ++i) {
			const auto& tri = mesh->triangles[i];
			triangles.emplace_back(
				tf_v(tri.v[0]), tf_v(tri.v[1]), tf_v(tri.v[2]), 
				tri.vn[0], tri.vn[1], tri.vn[2],
				tri.vt[0], tri.vt[1], tri.vt[2] 
			);
			if (userandom != 0) {
				if (userandom > 0) {
					material_ids.emplace_back(
						materials.size() + fix + 2
					);
				}
				else {
					material_ids.emplace_back(
						materials.size() + 2 + userandom
					);
				}
			}
			else {
				material_ids.emplace_back(
					materials.size() + mesh->material_ids[i]
				);
			}
		}
		if (userandom == 0) {
			for (const auto& mat : mesh->materials) {
				materials.emplace_back(mat);
			}
		}
	}

	void loadRevSurface(const Vec &offset, F scale, const std::vector<Vec> controls, const Material& material) {
		revsurfaces.emplace_back(offset, scale, controls);
		materials.push_back(material);

		{ // debug
			const auto& revsurface = revsurfaces.back();
			F usp = 30, vsp = 40;
			for (int u1 = 0; u1 < usp; ++u1) {
				int u2 = u1 + 1;
				Vec P1 = revsurface.deCasteljau(revsurface.controls, revsurface.n, u1/usp);
				Vec dP1 = revsurface.deCasteljau(revsurface.deltas, revsurface.n-1, u1/usp);
				Vec n1 = Vec::cross(dP1, Vec(0, 0, 1)).normal();
				Vec P2 = revsurface.deCasteljau(revsurface.controls, revsurface.n, u2/usp);
				Vec dP2 = revsurface.deCasteljau(revsurface.deltas, revsurface.n-1, u2/usp);
				Vec n2 = Vec::cross(dP2, Vec(0, 0, 1)).normal();
				for (int i1 = 0; i1 < vsp; ++i1) {
					int i2 = (i1 + 1 == (int)vsp) ? 0 : i1 + 1;
					F t1 = i1/vsp*2*M_PI;
					F t2 = i2/vsp*2*M_PI;
					i2 = i1 + 1;
					auto f = [scale, offset](const Vec &v, F t) {return Vec(v.x * cos(t), v.y, v.x * sin(t)) * scale + offset;};
					triangles.emplace_back(
						f(P1, t1), f(P1, t2), f(P2, t1),
						Tex(i1/vsp-0.5, u1/usp), Tex(i2/vsp-0.5, u1/usp), Tex(i1/vsp-0.5, u2/usp)
					);
					material_ids.emplace_back(materials.size()-1);
					triangles.emplace_back(
						f(P2, t2), f(P2, t1), f(P1, t2),
						Tex(i2/vsp-0.5, u2/usp), Tex(i1/vsp-0.5, u2/usp), Tex(i2/vsp-0.5, u1/usp)
					);
					material_ids.emplace_back(materials.size()-1);
				}
			}
		}
	}
};
