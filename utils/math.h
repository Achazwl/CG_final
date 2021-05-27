#ifndef UTILS_MATH
#define UTILS_MATH

#include "../config/config.h"

inline F sqr(F x) { return x * x; }

inline F clamp(F x) {
	return x<0 ? 0 : x>1 ? 1 : x;
} 

#endif // UTILS_MATH