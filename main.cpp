#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <memory>

#include "object/group.h"
#include "pt/tracing.h"

Group group(std::vector<Object3D*>{ // x, y, -z
	// new Sphere(1e5, Vec( 1e5+1,40.8,81.6),  new Material{Vec(),Vec(.75,.25,.25), Refl::DIFFUSE}), //Left 
	new Triangle(Vec(1, 0, 170), Vec(1, 0, 0), Vec(1, 81.6, 0), new Material(Vec(), Vec(.75, .25, .25), Refl::DIFFUSE)), // Left
	new Triangle(Vec(1, 0, 170), Vec(1, 81.6, 0), Vec(1, 81.6, 170), new Material(Vec(), Vec(.75, .25, .25), Refl::DIFFUSE)), // Left

	// new Sphere(1e5, Vec(-1e5+99,40.8,81.6), new Material{Vec(),Vec(.25,.25,.75), Refl::DIFFUSE}), //Rght 
	new Triangle(Vec(99, 0, 170), Vec(99, 81.6, 170), Vec(99, 81.6, 0), new Material{Vec(),Vec(.25,.25,.75), Refl::DIFFUSE}),
	new Triangle(Vec(99, 0, 170), Vec(99, 81.6, 0), Vec(99, 0, 0), new Material{Vec(),Vec(.25,.25,.75), Refl::DIFFUSE}),

	// new Sphere(1e5, Vec(50,40.8, 1e5),      new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}), //Back 
	new Triangle(Vec(1, 0, 0), Vec(99, 0, 0), Vec(99, 81.6, 0),  new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}),
	new Triangle(Vec(1, 0, 0), Vec(99, 81.6, 0), Vec(0, 81.6, 0),  new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}),

	// new Sphere(1e5, Vec(50,40.8,-1e5+170),  new Material{Vec(),Vec(), Refl::DIFFUSE}), //Frnt 
	new Triangle(Vec(1, 0, 170), Vec(1, 81.6, 170), Vec(99, 81.6, 170), new Material{Vec(),Vec(.75, .75, .25), Refl::DIFFUSE}),
	new Triangle(Vec(1, 0, 170), Vec(99, 81.6, 170), Vec(99, 0, 170), new Material{Vec(),Vec(.75, .75, .25), Refl::DIFFUSE}),

	// new Sphere(1e5, Vec(50, 1e5, 81.6),     new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}), //Botm 
	new Triangle(Vec(1, 0, 170), Vec(99, 0, 170), Vec(99, 0, 0), new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}), //Botm 
	new Triangle(Vec(1, 0, 170), Vec(99, 0, 0), Vec(1, 0, 0), new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}), //Botm 

	// new Sphere(1e10, Vec(50,1e10+81.6,81.6), new Material{Vec(),Vec(.25,.90,.25), Refl::DIFFUSE}), //Top 
	new Triangle(Vec(1, 81.6, 170), Vec(1, 81.6, 0), Vec(99, 81.6, 0), new Material{Vec(), Vec(0, 0.9, 0), Refl::DIFFUSE}),
	new Triangle(Vec(1, 81.6, 170), Vec(99, 81.6, 0), Vec(99, 81.6, 170), new Material{Vec(), Vec(0, 0.9, 0), Refl::DIFFUSE}),

	new Sphere(10.5,Vec(42,10.5,93),        new Material{Vec(),Vec(1,1,1)*.999,  Refl::GLASS }), //ball 1

	new Sphere(16.5,Vec(73,16.5,78),        new Material{Vec(),Vec(1,1,1)*.999,  Refl::MIRROR  }), //ball 2

	new Triangle(Vec(11, 0, 60), Vec(21, 0, 20), Vec(21, 43, 20), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)), // box
	new Triangle(Vec(41, 0, 60), Vec(21, 0, 20), Vec(21, 43, 20), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),
	new Triangle(Vec(11, 0, 60), Vec(21, 0, 80), Vec(21, 43, 80), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)), // box
	new Triangle(Vec(41, 0, 60), Vec(21, 0, 80), Vec(21, 43, 80), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),
	new Triangle(Vec(11, 43, 60), Vec(11, 0, 60), Vec(21, 43, 20), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)), // box
	new Triangle(Vec(41, 43, 60), Vec(41, 0, 60), Vec(21, 43, 20), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),
	new Triangle(Vec(11, 43, 60), Vec(11, 0, 60), Vec(21, 43, 80), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)), // box
	new Triangle(Vec(41, 43, 60), Vec(41, 0, 60), Vec(21, 43, 80), new Material(Vec(), Vec(1, 1, 1)*0.9, Refl::DIFFUSE)),

	new Sphere(600, Vec(50,681.6-.27,81.6), new Material{Vec(12,12,12),  Vec(),  Refl::DIFFUSE}), //Light
});

inline double clamp(double x) {
	return x<0 ? 0 : x>1 ? 1 : x;
} 
inline int toInt(double  x) {
	return int(pow(clamp(x),1/2.2)*255+.5);
} 

int main(int argc, char *argv[]) { 
	static constexpr int w=1024, h=768; // TODO bigger, better
	static constexpr int subpixel = 2, subpixel2 = subpixel*subpixel; // TODO bigger, better ?
	int samps = atoi(argv[1]) / subpixel2; // TODO bigger, better
	static constexpr double CAMERA_LEN_DISTANCE = 140;
	Ray cam(Vec(50,52,295.6), Vec(0,-0.042612,-1).normal());
	Vec camx = Vec(w*.5135/h), camy = (Vec::cross(camx, cam.d)).normal()*.5135; // (x, y, -z) coordinate
	Vec r;
	Vec *img = new RGB[w*h]; 

	#pragma omp parallel for schedule(dynamic, 1) private(r)

	for (int y = 0; y < h; y++)
	for (int x = 0; x < w; x++) { // loop image
		fprintf(stderr,"\rRendering (%d spp) %5.2f%%",samps*subpixel2,100.0*(y*w+x)/(h*w-1)); // progress bar
		Vec &col = img[(h-y-1)*w+x];
		for (int sy = 0; sy < subpixel; sy++)
		for (int sx = 0; sx < subpixel; sx++) { // loop subpixel
			double cx = x + (sx+.5) / subpixel, cy = y + (sy+.5) / subpixel;
			r = Vec();
			for (int s = 0; s < samps; s++){ 
				double dx = tent_filter(1/subpixel), dy = tent_filter(1/subpixel); // TODO: better filter (like bicubic)
				Vec d = camx * ( ( cx + dx ) / w - 0.5) +
						camy * ( ( cy + dy ) / h - 0.5) + 
						cam.d; 
				r = r + tracing(group, Ray(cam.o + d*CAMERA_LEN_DISTANCE, d.normal()), 0) / samps; // average over samps
			}
			col = col + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) / subpixel2; // average over 2*2 subpixel
		} 
	} 

	// Write image to PPM file. 
	FILE *f = fopen("image.ppm", "w");         
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
	for (int i = 0; i < w*h; i++) 
		fprintf(f,"%d %d %d ", toInt(img[i].x), toInt(img[i].y), toInt(img[i].z)); 
} 
