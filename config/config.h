#ifndef CG_CONFIG_H
#define CG_CONFIG_H

using F = double; // TODO: double, longdouble?

namespace Config {
    constexpr int imageW = 1024, imageH = 768;
    constexpr int num_subpixel = 2; // TODO bigger, better ?
    constexpr int spp = 4000;

    constexpr F inf = 1e+10;
}

#endif // CG_CONFIG_H