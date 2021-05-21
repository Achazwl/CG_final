#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <memory>

#include "object/group.h"
#include "pt/tracing.h"

Group group(std::vector<Object3D*>{
	new Sphere(1e5, Vec( 1e5+1,40.8,81.6),  new Material{Vec(),Vec(.75,.25,.25), Refl::DIFFUSE}), //Left 
	new Sphere(1e5, Vec(-1e5+99,40.8,81.6), new Material{Vec(),Vec(.25,.25,.75), Refl::DIFFUSE}), //Rght 
	new Sphere(1e5, Vec(50,40.8, 1e5),      new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}), //Back 
	new Sphere(1e5, Vec(50,40.8,-1e5+170),  new Material{Vec(),Vec(),            Refl::DIFFUSE}), //Frnt 
	new Sphere(1e5, Vec(50, 1e5, 81.6),     new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}), //Botm 
	new Sphere(1e5, Vec(50,-1e5+81.6,81.6), new Material{Vec(),Vec(.75,.75,.75), Refl::DIFFUSE}), //Top 
	new Sphere(16.5,Vec(27,16.5,47),        new Material{Vec(),Vec(1,1,1)*.999,  Refl::MIRROR }), //ball 1
	new Sphere(16.5,Vec(73,16.5,78),        new Material{Vec(),Vec(1,1,1)*.999,  Refl::GLASS  }), //ball 2
	new Sphere(600, Vec(50,681.6-.27,81.6), new Material{Vec(12,12,12),  Vec(),  Refl::DIFFUSE}), //Light
});

inline double clamp(double x) {
	return x<0 ? 0 : x>1 ? 1 : x;
} 
inline int toInt(double  x) {
	return int(pow(clamp(x),1/2.2)*255+.5);
} 

int main(int argc, char *argv[]) { 
	static constexpr int w=512, h=384; // TODO bigger, better
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
