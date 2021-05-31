#ifndef UTILS_MATH
#define UTILS_MATH

#include "../config/config.h"

inline __device__ F sqr(F x) { return x * x; }
inline __device__ F pow4(F x) { return x * x * x * x; }
inline __device__ F pow5(F x) { return x * x * x * x * x; }

template<typename T>
inline __device__ __host__ T min2(T x, T y) { return x < y ? x : y; }
template<typename T>
inline __device__ __host__ T max2(T x, T y) { return x > y ? x : y; }

inline __device__ __host__ F clamp(F x) {
	return x<0 ? 0 : x>1 ? 1 : x;
} 

inline __host__ int toInt(F  x) {
	return int(pow(clamp(x),1/2.2)*255+.5);
} 

#endif // UTILS_MATH