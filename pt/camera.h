#ifndef PT_CAMERA
#define PT_CAMERA

struct Camera {
    Vec o, x, y, _z;
    F length;
    int w, h, subpixel, subpixel2, samps, n_pixel, n_sub;

    __host__ Camera(Vec o, Vec x, Vec y, Vec _z, F length, int w, int h, int subpixel, int spp)
        : o(o), x(x), y(y), _z(_z), length(length), w(w), h(h), subpixel(subpixel) {
            subpixel2 = subpixel * subpixel;
            samps = spp / subpixel2;
            n_pixel = w * h;
            n_sub = n_pixel * subpixel2;
        }
};

#endif // PT_CAMERA