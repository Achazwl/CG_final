#include "object/group.h"

static Group group(std::vector<Object3D*>{ // x, y, -z
	// new Triangle(Vec(1, 0, 170), Vec(1, 0, 0), Vec(1, 81.6, 0), new Material(Vec(), Vec(.75, .25, .25), Refl::DIFFUSE)), // Left
	// new Triangle(Vec(1, 0, 170), Vec(1, 81.6, 0), Vec(1, 81.6, 170), new Material(Vec(), Vec(.75, .25, .25), Refl::DIFFUSE)),

	// new Triangle(Vec(99, 0, 170), Vec(99, 81.6, 170), Vec(99, 81.6, 0), new Material{Vec(),Vec(.25,.25,.75), Refl::DIFFUSE}), // Right
	// new Triangle(Vec(99, 0, 170), Vec(99, 81.6, 0), Vec(99, 0, 0), new Material{Vec(),Vec(.25,.25,.75), Refl::DIFFUSE}),

	// new Triangle(Vec(1, 0, 0), Vec(99, 0, 0), Vec(99, 81.6, 0),  new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}), // Back
	// new Triangle(Vec(1, 0, 0), Vec(99, 81.6, 0), Vec(0, 81.6, 0),  new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}),

	// new Triangle(Vec(1, 0, 170), Vec(1, 81.6, 170), Vec(99, 81.6, 170), new Material{Vec(),Vec(.75, .75, .25), Refl::DIFFUSE}), // Front
	// new Triangle(Vec(1, 0, 170), Vec(99, 81.6, 170), Vec(99, 0, 170), new Material{Vec(),Vec(.75, .75, .25), Refl::DIFFUSE}),

	// new Triangle(Vec(1, 0, 170), Vec(99, 0, 170), Vec(99, 0, 0), new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}), //Botm 
	// new Triangle(Vec(1, 0, 170), Vec(99, 0, 0), Vec(1, 0, 0), new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}),

	// new Triangle(Vec(1, 81.6, 170), Vec(1, 81.6, 0), Vec(99, 81.6, 0), new Material{Vec(), Vec(0, 0.9, 0), Refl::DIFFUSE}), // TOP
	// new Triangle(Vec(1, 81.6, 170), Vec(99, 81.6, 0), Vec(99, 81.6, 170), new Material{Vec(), Vec(0, 0.9, 0), Refl::DIFFUSE}),

	// new Triangle(Vec(11, 0, 60), Vec(21, 0, 20), Vec(21, 43, 20), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)), // box
	// new Triangle(Vec(41, 0, 60), Vec(21, 0, 20), Vec(21, 43, 20), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),
	// new Triangle(Vec(11, 0, 60), Vec(21, 0, 80), Vec(21, 43, 80), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),
	// new Triangle(Vec(41, 0, 60), Vec(21, 0, 80), Vec(21, 43, 80), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),
	// new Triangle(Vec(11, 43, 60), Vec(11, 0, 60), Vec(21, 43, 20), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),
	// new Triangle(Vec(41, 43, 60), Vec(41, 0, 60), Vec(21, 43, 20), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),
	// new Triangle(Vec(11, 43, 60), Vec(11, 0, 60), Vec(21, 43, 80), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),
	// new Triangle(Vec(41, 43, 60), Vec(41, 0, 60), Vec(21, 43, 80), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),

	new Sphere(600, Vec(50,681.6-.27,81.6), new Material{Vec(12,12,12),  Vec(),  Refl::DIFFUSE}), //Light

	// new Sphere(10.5,Vec(42,10.5,93),        new Material{Vec(),Vec(1,1,1)*.999,  Refl::GLASS }), //ball 1

	// new Sphere(16.5,Vec(73,16.5,78),        new Material{Vec(),Vec(1,1,1)*.999,  Refl::MIRROR  }), //ball 2

	new Mesh(Vec(50, 30, 50), 100, "mesh/body.obj", new Material(Vec(), Vec(1, 1, 1)*0.75, Refl::DIFFUSE)),
});

// TODO: memory leak