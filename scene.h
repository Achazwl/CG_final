#include "object/group.h"

static Group group(std::vector<Object3D*>{ // x, y, -z
	new Mesh(Vec(1, 0, 0), Vec(-1, 81.6, 170), "mesh/cube.obj", new Material(Vec(), Vec(.75, .25, .25), Refl::DIFFUSE)), // Left
	new Mesh(Vec(99, 0, 0), Vec(1, 81.6, 170), "mesh/cube.obj", new Material{Vec(),Vec(.25,.25,.75), Refl::DIFFUSE}), // Right

	new Mesh(Vec(1, 0, 0), Vec(98, 81.6, -1), "mesh/cube.obj", new Material{Vec(), Vec(.75,.75,.75), Refl::GLOSS, "images/Teacup.png"}), // Back
	new Mesh(Vec(1, 0, 170), Vec(98, 81.6, 1), "mesh/cube.obj", new Material{Vec(), Vec(.9,.75,.75), Refl::DIFFUSE}), // Front

	new Mesh(Vec(1, 0, 0), Vec(98, -1, 170), "mesh/cube.obj", new Material{Vec(),Vec(.75,.75,.75), Refl::GLOSS, "images/wood.jpg"}), //Botm 
	new Mesh(Vec(1, 81.6, 0), Vec(98, 1, 170), "mesh/cube.obj", new Material{Vec(), Vec(0, 0.9, 0), Refl::DIFFUSE}), // TOP

	new Sphere(600, Vec(50,681.6-.27,81.6), new Material{Vec(12,12,12),  Vec(),  Refl::DIFFUSE}), //Light

	// new Sphere(10.5, Vec(42,10.5,93),        new Material{Vec(),Vec(1,1,1)*.999,  Refl::GLASS }), //ball glass

	// new Sphere(16.5, Vec(73,16.5,78),        new Material{Vec(),Vec(1,1,1)*.999,  Refl::MIRROR  }), //ball mirror

	new Sphere(10.5, Vec(42,10.5,93),        new Material{Vec(),Vec(),  Refl::GLOSS, "images/volleyball.jpg"}), // volleyball

	// new Mesh(Vec(50, 30, 50), Vec(1,1,1)*100, "mesh/bunny_200.obj", new Material(Vec(), Vec(1, 1, 1)*0.75, Refl::DIFFUSE)),
});

// TODO: memory leak