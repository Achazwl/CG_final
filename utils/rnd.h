#ifndef UTILS_RND
#define UTILS_RND

#include "../config/config.h"
#include "../utils/math.h"
#include <cstdlib>
#include <random>
#include <curand.h>
#include <curand_kernel.h>

inline __device__ F rnd(F range, curandState *st) {
	return curand_uniform_double(st) * range;
}

inline __device__ void rndCosWeightedHSphere(F &a, F &b, F &c, curandState *st) {
	F rad = rnd(2*M_PI, st), r2 = rnd(1, st), r = sqrt(r2); 
	a = cos(rad) * r;
	b = sin(rad) * r;
	c = sqrt(1 - r2);
}

inline __device__ std::tuple<F, F, F> rndHSphere(curandState *st) {
	F phi = rnd(2*M_PI, st), cost = rnd(1, st), sint = sqrt(1 - sqr(cost));
	return {cos(phi) * sint, sin(phi) * sint, cost};
}

inline __device__ F tent_filter(F scale, curandState *st) {
	F r = rnd(2, st);
	F d = r < 1 ? sqrt(r)-1: 1-sqrt(2-r); 
	return d * scale;
}

#endif // UTILS_RND