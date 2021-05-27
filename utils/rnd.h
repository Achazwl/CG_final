#ifndef UTILS_RND
#define UTILS_RND

#include "../config/config.h"
#include "../utils/math.h"
#include <cstdlib>
#include <random>

inline F rnd(F range=1) {
	static std::random_device rd;
	static std::default_random_engine e{rd()}; // or std::default_random_engine e{rd()};
	static std::uniform_real_distribution<F> dist{0, 1};
	return dist(e) * range;
}

inline std::tuple<F, F, F> rndCosWeightedHSphere() {
	F rad = rnd(2*M_PI), r2 = rnd(), r = sqrt(r2); 
	return {cos(rad) * r, sin(rad) * r, sqrt(1 - r2)};
}

inline std::tuple<F, F, F> rndHSphere() {
	F phi = rnd(2*M_PI), cost = rnd(1), sint = sqrt(1 - sqr(cost));
	return {cos(phi) * sint, sin(phi) * sint, cost};
}

inline F tent_filter(F scale=1) {
	F r = rnd(2);
	F d = r < 1 ? sqrt(r)-1: 1-sqrt(2-r); 
	return d * scale;
}

#endif // UTILS_RND