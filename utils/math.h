#ifndef UTILS_MATH
#define UTILS_MATH

#include "../config/config.h"

inline __device__ F sqr(F x) { return x * x; }
inline __device__ F pow4(F x) { return x * x * x * x; }
inline __device__ F pow5(F x) { return x * x * x * x * x; }

template<typename T>
inline __device__ __host__ T min(T x, T y) { return x < y ? x : y; }
template<typename T>
inline __device__ __host__ T max(T x, T y) { return x > y ? x : y; }
template<typename T>
inline __device__ __host__ F clamp(T x, T mn = 0, T mx = 1) {
	return x<mn ? mn : x>mx ? mx : x;
} 
inline __device__ void swap(F &x, F &y) {
	F tmp = x;
	x = y;
	y = tmp;
}

inline __host__ int toInt(F x) {
	return int(pow(clamp(x),1/2.2)*255+.5);
} 

#endif // UTILS_MATH