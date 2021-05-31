#ifndef UTILS_VECTOR
#define UTILS_VECTOR

static constexpr int default_capacity = 2;

template<typename T>
class Vector {
    int _capacity;
    int _size;
    T* _a;
private:
    __device__ __host__ void expand() {
        if (_size < _capacity) return;
        T* old = _a;
        _a = new T[_capacity <<= 1];
        for (int i = 0; i < _size; i++)
            _a[i] = old[i];
        delete[] old;
    }

    __device__ __host__ void duplicate(const Vector &rhs) {
        _capacity = rhs._capacity;
        _size = rhs._size;
        _a = new T[_capacity];
        for (int i = 0; i < _size; ++i)
            _a[i] = rhs._a[i];
    }
public:
    using iterator = T*;
    __device__ __host__ explicit Vector(int capacity = default_capacity): _capacity(capacity), _size(0), _a(new T[capacity]) {}
    __device__ __host__ Vector(const std::initializer_list<T> &args) {
        for (auto itm : args) {
            push_back(itm);
        }
    }

    __device__ __host__ Vector(const Vector& rhs) { duplicate(rhs); }
    __device__ __host__ Vector& operator = (const Vector &rhs) { delete[] _a; duplicate(rhs); return *this; }

    __device__ __host__ iterator begin() const { return _a; }
    __device__ __host__ iterator end() const { return _a + _size; }

    __device__ __host__ T& operator [] (int x) { return _a[x]; }
    __device__ __host__ T operator [] (int x) const { return _a[x]; }

    __device__ __host__ void push_back(T v) {
        expand();
        *end() = v;
        ++_size;
    }

    __device__ __host__ int size() const { return _size; }
    __device__ __host__ bool empty() const { return _size == 0; }

    __device__ __host__ ~Vector() { delete[] _a; }
};

#endif //UTILS_VECTOR
