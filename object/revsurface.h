#ifndef OBJ_REV
#define OBJ_REV

#include "base.h"

struct RevSurface { // rotate around y aixs
	RevSurface() = default;
    RevSurface(const Vec &offset, const Vec &scale, const std::vector<Vec> &points)
    : offset(offset), scale(scale) { // TODO change to BSPline
        n = points.size() - 1; // label from 0..n (size = n + 1)
        controls = new Vec[n+1];
        for (int i = 0; i <= n; ++i) controls[i] = points[i];
        deltas = new Vec[n];
        for (int i = 0; i < n; ++i)
            deltas[i] = n * (controls[i+1] - controls[i]);
    }
	RevSurface(const RevSurface& rhs) = default;

	__device__ bool intersect(const Ray &ray, Hit &hit) const {
        // TODO offset scale
        int resolution = 2;
        F dis = 1. / resolution;
        for (int ini = 1; ini < resolution; ++ini) {
            for (int loop = 0; loop < 30; ++loop) {
                F u = ini * dis;
                Vec P = deCasteljau(controls, n, u);
                Vec dP = deCasteljau(deltas, n-1, u);
                F T = (P.y - ray.o.y) / ray.d.y;
                F dT = dP.y / ray.d.y;
                F f = sqr(P.x)
                    - sqr(T * ray.d.z + ray.o.z)
                    - sqr(T * ray.d.x + ray.o.x);
                if (fabs(f) < eps) {
                    F cost = (T * ray.d.x + ray.o.x) / P.x;
                    F sint = (T * ray.d.z + ray.o.z) / P.x;
                    Vec n = Vec::cross(dP, Vec(0, 0, 1)).normal();
                    hit.set(T, Vec(n.x * cost, n.y, n.x * sint));
                    return true;
                }
                F df = 2 * P.x * dP.x
                    - 2 * (T * ray.d.z + ray.o.z) * dT * ray.d.z
                    - 2 * (T * ray.d.x + ray.o.x) * dT * ray.d.x;
                u -= f / df;
                printf("%lf %lf %lf\n", u, f, df);
                if (u < 0 || u > 1) break;
            }
        }
        return false;
        // F t, u, theta; 
        // t = 10, u = 0, theta = 0; // TODO cylinder bbox initial value
        // for (int i = 0; i < 2; ++i) { // TODO this limit
        //     Vec P = deCasteljau(controls, n, u);
        //     Vec dP = deCasteljau(deltas, n-1, u);
        //     F cost = cos(theta), sint = sin(theta);
        //     Vec _F = - (ray.o + t * ray.d - Vec(P.x * cost, P.y, P.x * sint));
        //     if (fabs(_F.x) < eps && fabs(_F.y) < eps && fabs(_F.z) < eps) {
        //         Vec n = Vec::cross(dP, Vec(0, 0, 1)).normal();
        //         hit.set(t, Vec(n.x * cost, n.y, n.x * sint));
        //         return true;
        //     }
        //     // gauss elimination
        //     F a[3][4] = {
        //         {ray.d.x, - dP.x * cost,   P.x * sint, _F.x},
        //         {ray.d.y, - dP.y       ,   0         , _F.y},
        //         {ray.d.z, - dP.x * sint, - P.x * cost, _F.z}
        //     };
        //     for (int k = 0; k < 3; ++k) {
        //         int i_max = k;
        //         int v_max = a[i_max][k];
        //         for (int i = k+1; i < 3; ++i)
        //             if (abs(a[i][k]) > v_max)
        //                 v_max = a[i][k], i_max = i;
        //         // assert(v_max != 0);
        //         if (i_max != k) {
        //             for (int i = 0; i <= 3; ++i) {
        //                 F tmp = a[i_max][i];
        //                 a[i_max][i] = a[k][i];
        //                 a[k][i] = tmp;
        //             }
        //         }
        //         for (int i = k+1; i < 3; ++i) {
        //             F f = a[i][k]/a[k][k];
        //             for (int j = k+1; j <= 3; ++j)
        //                 a[i][j] -= a[k][j] * f;
        //             a[i][k] = 0;
        //         }
        //     }
        //     F sol[3];
        //     for (int i = 3-1; i >= 0; --i) {
        //         sol[i] = a[i][3];
        //         for (int j = i+1; j < 3; ++j)
        //             sol[i] -= a[i][j] * sol[j];
        //         sol[i] = sol[i] / a[i][i];
        //     }
        //     {
        //         t += sol[0];
        //         u += sol[1];
        //         theta += sol[2];
        //     }
        // }
        // return false; 
	} 

    // TODO V = deCasteljau(controls, n, u),  T = deCasteljau(deltas, n-1, u)
    __device__ Vec deCasteljau(const Vec *P_, int n, float u) const {
        Vec *P = new Vec[n+1];
        for (int i = 0; i <= n; ++i) P[i] = P_[i];
        for (int i = n; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                P[j] = u * P[j+1] + (1 - u) * P[j];
            }
        }
        Vec res = P[0];
        delete[] P;
        return res;
    }

    __host__ RevSurface* to() const {
        RevSurface* rev = new RevSurface(*this);
		cudaMalloc((void**)&rev->controls, (n+1)*sizeof(Vec));
		cudaMemcpy(rev->controls, controls, (n+1)*sizeof(Vec), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&rev->deltas, (n)*sizeof(Vec));
		cudaMemcpy(rev->deltas, deltas, (n)*sizeof(Vec), cudaMemcpyHostToDevice);
        return rev;
    }

public: // TODO protected:
    Vec *controls, *deltas;
    Vec offset, scale;
    int n;
}; 

#endif // OBJ_REV