#ifndef OBJ_REV
#define OBJ_REV

#include "base.h"

struct RevSurface { // rotate around y aixs
	RevSurface() = default;
    RevSurface(const Vec &offset, F scale, const std::vector<Vec> &points)
    : offset(offset), scale(scale) {
        n = points.size() - 1; // label from 0..n (size = n + 1)
        controls = new Vec[n+1];
        for (int i = 0; i <= n; ++i) controls[i] = points[i];
        deltas = new Vec[n];
        for (int i = 0; i < n; ++i)
            deltas[i] = n * (controls[i+1] - controls[i]);
        
        {
            F mxx = 0, mxy = 0;
            for (int i = 0; i <= n; ++i) {
                if (controls[i].x > mxx) mxx = controls[i].x;
                if (controls[i].y > mxy) mxy = controls[i].y;
            }
            bound = Bound(
                Vec(-mxx, 0, -mxx),
                Vec(mxx, mxy, mxx)
            );
            bound.mn = bound.mn * scale + offset;
            bound.mx = bound.mx * scale + offset;
            bound.mn.debug();
            bound.mx.debug();
        }
    }
	RevSurface(const RevSurface& rhs) = default;

	__device__ bool intersect(const Ray &ray, Hit &hit) const {
        F Tini;
        if (!bound.intersect(ray, Tini)) return false;
        Vec pt = ray.At(Tini) - offset;
        F tini = atan2(-pt.z, pt.x) + M_PI;
        F uini = (controls[n].y - pt.y/scale) / (controls[n].y - controls[0].y);
        bool flag = false;
        for (int tri = 1; tri <= 1; ++tri) {
            F T = Tini, t = tini, u = uini + tri * 0.2;
            for (int loop = 0; loop < 20; ++loop) {
                t = t - int(t/(2*M_PI))*2*M_PI;
                if (t < 0) t += 2*M_PI;
                u = clamp<F>(u, eps, 1-eps);
                Vec P = deCasteljau(controls, n, u) * scale;
                Vec dP = deCasteljau(deltas, n-1, u) * scale;
                F cost = cos(t), sint = sin(t);
                Vec pt(P.x * cost, P.y, P.x * sint);
                Vec du(dP.x * cost, dP.y, dP.x * sint);
                Vec dt(-P.x * sint, 0, P.x * cost);
                Vec f = ray.At(T) - offset - pt;
                F dis = f.norm2();
                Vec no = Vec::cross(du, dt);
                if (dis < eps) {
                    if (hiteps < T && T < hit.t) {
                        hit.set(T, no, Tex(t/(2*M_PI), u));
                        flag = true;
                    }
                    break;
                }
                else {
                    F div = 1. / Vec::dot(ray.d, no);
                    T -= Vec::dot(du, Vec::cross(dt, f)) * div;
                    u -= Vec::dot(ray.d, Vec::cross(dt, f)) * div;
                    t += Vec::dot(ray.d, Vec::cross(du, f)) * div;
                }
            }
        }
        return flag;
	} 
    
    __device__ bool intersect2(const Ray &ray, Hit &hit) const {
        F unused;
        if (!bound.intersect(ray, unused)) return false;
        bool flag = false;
        if (true) {
            int resolution = 8, iter = 10;
            F dis = 1. / resolution;
            for (int ini = 1; ini < resolution; ++ini) {
                F u = ini * dis;
                for (int loop = 0; loop < iter; ++loop) {
                    Vec P = deCasteljau(controls, n, u) * scale;
                    Vec dP = deCasteljau(deltas, n-1, u) * scale;
                    Vec O = ray.o - offset;
                    F T = (P.y - O.y) / ray.d.y;
                    F dT = dP.y / ray.d.y;
                    F f = sqr(P.x)
                        - sqr(T * ray.d.z + O.z)
                        - sqr(T * ray.d.x + O.x);
                    if (fabs(f) < eps) {
                        if (hiteps < T && T < hit.t) {
                            F cost = (T * ray.d.x + O.x) / P.x;
                            F sint = (T * ray.d.z + O.z) / P.x;
                            Vec n = Vec::cross(dP, Vec(0, 0, 1)).normal();
                            Vec no = Vec(n.x * cost, n.y, n.x * sint);
                            Tex tex(0.2+atan2(sint, cost)*0.5*M_1_PI, u);
                            hit.set(T, no, tex);
                            flag = true;
                        }
                        break;
                    }
                    F df = 2 * P.x * dP.x
                        - 2 * (T * ray.d.z + O.z) * dT * ray.d.z
                        - 2 * (T * ray.d.x + O.x) * dT * ray.d.x;
                    u -= f / df;
                    if (u < eps || u > 1-eps) break;
                }
            }
        }
        return flag;
	} 

    __device__ __host__ Vec deCasteljau(const Vec *P_, int n, F u) const {
        Vec *P = new Vec[n+1];
        for (int i = 0; i <= n; ++i) {
            P[i] = P_[i];
        }
        for (int i = n-1; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                P[j] = u * P[j+1] + (1 - u) * P[j];
            }
        }
        Vec res = P[0];
        delete[] P;
        return res;
    }

    RevSurface* to() const {
        RevSurface* rev = new RevSurface(*this);
		cudaMalloc((void**)&rev->controls, (n+1)*sizeof(Vec));
		cudaMemcpy(rev->controls, controls, (n+1)*sizeof(Vec), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&rev->deltas, (n)*sizeof(Vec));
		cudaMemcpy(rev->deltas, deltas, (n)*sizeof(Vec), cudaMemcpyHostToDevice);
        return rev;
    }

public:
    Vec *controls, *deltas;
    Vec offset; F scale;
    int n;
    Bound bound;
}; 

#endif // OBJ_REV
