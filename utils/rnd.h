#ifndef UTILS_RND
#define UTILS_RND

#include <cstdlib>
#include <random>

inline double rnd(double range=1) {
	static std::random_device rd;
	static std::default_random_engine e{rd()}; // or std::default_random_engine e{rd()};
	static std::uniform_real_distribution<double> dist{0, 1};
	return dist(e) * range;
}

inline double tent_filter(double scale=1) {
	double r = rnd(2);
	double d = r < 1 ? sqrt(r)-1: 1-sqrt(2-r); 
	return d * scale;
}

#endif // UTILS_RND