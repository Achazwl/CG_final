#ifndef UTILS_PAIR
#define UTILS_PAIR

template<typename A, typename B>
struct Pair {
    A first;
    B second;

    __device__ __host__ Pair() {}
    __device__ __host__ Pair(A first, B second) : first(first), second(second) {}
    __device__ __host__ Pair(const Pair &rhs) : first(rhs.first), second(rhs.second) {}
    __device__ __host__ Pair operator = (const Pair &rhs) { first = rhs.first; second = rhs.second; return *this; }
};

template<typename A>
struct Tuple {
    A x, y, z;

    __device__ __host__ Tuple() {}
    __device__ __host__ Tuple(A x, A y, A z) : x(x), y(y), z(z) {}
    __device__ __host__ Tuple(const Tuple &rhs) : x(rhs.x), y(rhs.y), z(rhs.z) {}
    __device__ __host__ Tuple operator = (const Tuple &rhs) { first = rhs.first; second = rhs.second; return *this; }
};

#endif //UTILS_PAIR