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

__global__ void kernelRayTrace(Group *group, Camera *cam, Vec *result, curandState *states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= cam->n_sub) return;
	int pixel_idx = idx / cam->subpixel2;
	int y = pixel_idx / cam->w;
	int x = pixel_idx % cam->w;
	int sy = (idx % cam->subpixel2) / cam->subpixel;
	int sx = (idx % cam->subpixel2) % cam->subpixel;

	curand_init(y*y*y, idx, 0, &states[idx]); // TODO: no set seed each thread
	curandState st = states[idx];

	Vec r = Vec();
	F cx = x + (sx+.5) / cam->subpixel, cy = y + (sy+.5) / cam->subpixel;
	for (int s = 0; s < cam->samps; s++){ 
		F dx = tent_filter(1/cam->subpixel, &st), dy = tent_filter(1/cam->subpixel, &st); // TODO: better filter (like bicubic)
		Vec d = cam->x * ( ( cx + dx ) / cam->w - 0.5) +
				cam->y * ( ( cy + dy ) / cam->h - 0.5) + 
				cam->_z; 
		r = r + tracing(group, Ray(cam->o+d*cam->length, d.normal()), &st) / cam->samps; // average over samps
	}
	result[idx] = r;
}

__global__ void kernelCombResult(Vec *subpixel, Vec *pixel, Camera *cam) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cam->n_pixel) return;

    Vec res = Vec();
    #pragma unroll
    for (int i = 0; i < cam->subpixel2; i++) {
        Vec sub = subpixel[idx * cam->subpixel2 + i];
        res = res + Vec(clamp(sub.x), clamp(sub.y), clamp(sub.z)) / cam->subpixel2;
    }

    pixel[idx] = res;
}

__host__ int ceil_div(int x, int y) {
	return (x + y - 1) / y;
}

int main(int argc, char *argv[]) { 
	printf("initial begin\n");
	Scene scene;
	Camera *cam;
	cudaMalloc((void**)&cam, sizeof(Camera));
	cudaMemcpy(cam, scene.cam, sizeof(Camera), cudaMemcpyHostToDevice); // cpu -> gpu
	Group *group = scene.group->to(); // cpu -> gpu
	printf("initial end\n");

	curandState *states;
	Vec *sub_result;
	cudaMalloc((void**)&states, scene.cam->n_sub*sizeof(curandState));
	cudaMalloc((void**)&sub_result, scene.cam->n_sub*sizeof(Vec));
	dim3 blockDim(blocksize, 1);
	dim3 gridDim(ceil_div(scene.cam->n_sub, blocksize), 1);
	kernelRayTrace<<<gridDim, blockDim>>>(group, cam, sub_result, states);
	cudaPeekAtLastError();
	printf("render begin\n");
	gpuErrchk( cudaDeviceSynchronize() ); // wait all
	printf("render end\n");

	Vec *pixel_result;
	cudaMalloc((void**)&pixel_result, scene.cam->n_pixel*sizeof(Vec));
	dim3 gridDim2(ceil_div(scene.cam->n_pixel, blocksize), 1);
	kernelCombResult<<<gridDim2, blockDim>>>(sub_result, pixel_result, cam);
	gpuErrchk( cudaDeviceSynchronize() ); // wait all
	printf("combine end\n");

	Vec *img = new RGB[scene.cam->n_pixel]; 
	cudaMemcpy(img, pixel_result, scene.cam->n_pixel*sizeof(Vec), cudaMemcpyDeviceToHost); // gpu to cpu
	FILE *f = fopen("image.ppm", "w"); // write to image file
	fprintf(f, "P3\n%d %d\n%d\n", scene.cam->w, scene.cam->h, 255); 
	for (int i = 0; i < scene.cam->n_pixel; i++) 
		fprintf(f,"%d %d %d ", toInt(img[i].x), toInt(img[i].y), toInt(img[i].z)); 
	printf("image output end\n");

	cudaFree(states);
	cudaFree(sub_result);
	cudaFree(pixel_result);
	cudaFree(group);
	cudaFree(cam);

	return 0;
} 