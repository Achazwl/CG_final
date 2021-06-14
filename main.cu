#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <memory>

#include "config/config.h"
#include "pt/tracing.h"
#include "scene.h"
#include "utils/rnd.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void debug(Group *group) {
}

__global__ void init(Camera *cam, Vec *result, curandState *states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= cam->n_sub) return;
	int pixel_idx = idx / cam->subpixel2;
	int y = pixel_idx / cam->w;
	int x = pixel_idx % cam->w;
	int sy = (idx % cam->subpixel2) / cam->subpixel;
	int sx = (idx % cam->subpixel2) % cam->subpixel;

	curand_init(y*y*y, idx, 0, &states[idx]);

	result[idx] = Vec();
}

__global__ void kernelRayTrace(Group *group, Camera *cam, Vec *result, curandState *states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= cam->n_sub) return;
	int pixel_idx = idx / cam->subpixel2;
	int y = pixel_idx / cam->w;
	int x = pixel_idx % cam->w;
	int sy = (idx % cam->subpixel2) / cam->subpixel;
	int sx = (idx % cam->subpixel2) % cam->subpixel;

	curandState* st = &states[idx];

	F cx = x + (sx+.5) / cam->subpixel, cy = y + (sy+.5) / cam->subpixel;
	F dx = tent_filter(1/cam->subpixel, st), dy = tent_filter(1/cam->subpixel, st); // TODO: better filter (like bicubic)
	Vec d = cam->x * ( (cx + dx) / cam->w - 0.5 ) + cam->y * ( (cy + dy) / cam->h - 0.5 ) + cam->_z; 
	result[idx] = result[idx] + tracing(group, Ray(cam->o+d*cam->length, d.normal()), st);
}

__global__ void kernelCombResult(Vec *subpixel, Vec *pixel, Camera *cam, int samp) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= cam->n_pixel) return;

	Vec res = Vec();
	F div = 1. / samp;
	#pragma unroll
	for (int i = 0; i < cam->subpixel2; i++) {
		Vec sub = subpixel[idx * cam->subpixel2 + i] * div;
		res = res + Vec(clamp(sub.x), clamp(sub.y), clamp(sub.z)) / cam->subpixel2;
	}

	pixel[idx] = res;
}

int ceil_div(int x, int y) {
	return (x + y - 1) / y;
}

int main(int argc, char *argv[]) { 
	printf("initial begin\n");
	Scene scene;
	printf("load scene finished\n");
	Camera *cam;
	cudaMalloc((void**)&cam, sizeof(Camera));
	cudaMemcpy(cam, scene.cam, sizeof(Camera), cudaMemcpyHostToDevice); // cpu -> gpu
	Group *group = scene.group->to(); // cpu -> gpu
	printf("initial end\n");

	// { // debug block
	// 	debug<<<dim3(1), dim3(1)>>>(group);
	// 	gpuErrchk( cudaDeviceSynchronize() );
	// 	printf("debug test pass\n");
	// }

	curandState *states;
	Vec *sub_result;
	Vec *pixel_result;
	cudaMalloc((void**)&states, scene.cam->n_sub*sizeof(curandState));
	cudaMalloc((void**)&sub_result, scene.cam->n_sub*sizeof(Vec));
	cudaMalloc((void**)&pixel_result, scene.cam->n_pixel*sizeof(Vec));

	dim3 blockDim(blocksize);
	dim3 gridDim1(ceil_div(scene.cam->n_sub, blocksize));
	dim3 gridDim2(ceil_div(scene.cam->n_pixel, blocksize));

	init<<<gridDim1, blockDim>>>(cam, sub_result, states);

	for (int samp = 1; samp <= scene.cam->samps; ++samp) {
		fprintf(stderr, "\rrendering %6d of %d", samp, scene.cam->samps);
		kernelRayTrace<<<gridDim1, blockDim>>>(group, cam, sub_result, states);
		gpuErrchk( cudaDeviceSynchronize() ); // wait all

		kernelCombResult<<<gridDim2, blockDim>>>(sub_result, pixel_result, cam, samp);
		gpuErrchk( cudaDeviceSynchronize() ); // wait all

		Vec *img = new RGB[scene.cam->n_pixel]; 
		cudaMemcpy(img, pixel_result, scene.cam->n_pixel*sizeof(Vec), cudaMemcpyDeviceToHost); // gpu to cpu
		FILE *f = fopen("image.ppm", "w");
		fprintf(f, "P3\n%d %d\n%d\n", scene.cam->w, scene.cam->h, 255); 
		for (int i = 0; i < scene.cam->n_pixel; i++) {
			fprintf(f, "%d %d %d ", toInt(img[i].x), toInt(img[i].y), toInt(img[i].z));
		}
		fclose(f);
		delete[] img;
	}
		
	cudaFree(states);
	cudaFree(sub_result);
	cudaFree(pixel_result);
	cudaFree(group);
	cudaFree(cam);

	return 0;
} 
