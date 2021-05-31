#include "pt/camera.h"
#include "object/group.h"

static int w = 1024, h = 768;

namespace Scene {
	Camera cam(
		Vec(50,52,295.6), // o
		Vec(w*.5135/h), // x
		(Vec::cross(Vec(w*.5135/h), Vec(0,-0.042612,-1).normal())).normal()*.5135, // y
		Vec(0,-0.042612,-1).normal(), // -z
		140, // length
		w, h, // w, h
		2, // subpixel
		50 // spp
	);

	// Group group(Vector<Object3D*>{ // x, y, -z
	// 	new Mesh(Vec(1, 0, 0), Vec(-1, 81.6, 170), "mesh/cube.obj", new Material(Vec(), Vec(.75, .25, .25), Vec(1,1,1)*0.02, Refl::GLASS)), // Left
	// 	new Mesh(Vec(99, 0, 0), Vec(1, 81.6, 170), "mesh/cube.obj", new Material{Vec(),Vec(.25,.25,.75), Vec(1,1,1)*0.02, Refl::GLASS}), // Right

	// 	new Mesh(Vec(1, 0, 0), Vec(98, 81.6, -1), "mesh/cube.obj", new Material{Vec(), Vec(.75,.75,.75), Vec(1,1,1)*0.02, Refl::GLASS, "images/Teacup.png"}), // Back
	// 	new Mesh(Vec(1, 0, 170), Vec(98, 81.6, 1), "mesh/cube.obj", new Material{Vec(), Vec(.9,.75,.75), Vec(1,1,1)*0.02, Refl::GLASS}), // Front

	// 	new Mesh(Vec(1, 0, 0), Vec(98, -1, 170), "mesh/cube.obj", new Material{Vec(),Vec(.75,.75,.75), Vec(1,1,1)*0.04, Refl::GLASS, "images/wood.jpg"}), //Botm 
	// 	new Mesh(Vec(1, 81.6, 0), Vec(98, 1, 170), "mesh/cube.obj", new Material{Vec(), Vec(0, 0.9, 0), Vec(1,1,1)*0.02, Refl::GLASS}), // TOP

	// 	new Sphere(600, Vec(50,681.6-.27,81.6), new Material{Vec(12,12,12),  Vec(), Vec(), Refl::GLASS}), //Light

	// 	// new Sphere(10.5, Vec(42,10.5,93),        new Material{Vec(),Vec(1,1,1)*.999, Vec(), Refl::GLASS, "images/volleyball.jpg"}), //ball glass

	// 	// new Sphere(16.5, Vec(73,16.5,78),        new Material{Vec(),Vec(1,1,1)*.999, Vec(),  Refl::MIRROR  }), //ball mirror

	// 	// new Sphere(10.5, Vec(42,10.5,93),        new Material{Vec(),Vec(),  Refl::GLOSS, "images/volleyball.jpg"}), // volleyball

	// 	// new Mesh(Vec(50, 30, 50), Vec(1,1,1)*100, "mesh/bunny_200.obj", new Material(Vec(), Vec(1, 1, 1)*0.75, Refl::DIFFUSE)),

	// 	new Sphere(10.5, Vec(30,10.5,93),        new Material{Vec(),Vec(0.45, 0.45, 0.45), Vec(1,1,1)*0.03, Refl::GLASS}), // left test
	// 	new Sphere(10.5, Vec(70,10.5,93),        new Material{Vec(),Vec(0.15, 0.15, 0.15), Vec(1,1,1)*0.98, Refl::GLASS}), // right test
	// }); // TODO: memory leak
}
