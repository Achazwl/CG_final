#ifndef PT_CAMERA
#define PT_CAMERA

struct Camera {
    Vec o, x, y, _z;
    F length, focus;
    int w, h, subpixel, subpixel2, samps, n_pixel, n_sub;

    Camera(Vec o, Vec x, Vec y, Vec _z, F length, F focus, int w, int h, int subpixel, int spp)
        : o(o), x(x), y(y), _z(_z), length(length), focus(focus), w(w), h(h), subpixel(subpixel) {
            subpixel2 = subpixel * subpixel;
            samps = spp / subpixel2;
            n_pixel = w * h;
            n_sub = n_pixel * subpixel2;
        }
    __device__ Camera(const Camera &rhs)
        : o(rhs.o), x(rhs.x), y(rhs.y), _z(rhs._z), length(rhs.length),
        w(rhs.w), h(rhs.h), subpixel(rhs.subpixel), subpixel2(rhs.subpixel2), samps(rhs.samps),
        n_pixel(rhs.n_pixel), n_sub(rhs.n_sub) {}
};

#endif // PT_CAMERA