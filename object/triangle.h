#ifndef OBJ_TRIANGLE
#define OBJ_TRIANGLE

#include "base.h"

struct Triangle { 
	Triangle() = default;
	Triangle(
        const Vec &a, const Vec &b, const Vec &c
    ) : v{a,b,c} {
        init();
        this->vn[0] = this->vn[1] = this->vn[2] = n;
    }
	Triangle(
        const Vec &a, const Vec &b, const Vec &c,
        const Vec &na, const Vec &nb, const Vec &nc,
        const Tex &ta, const Tex &tb, const Tex &tc
    ) : v{a,b,c}, vn{na, nb, nc}, vt{ta, tb, tc} {
        init();
    }
	Triangle(
        const Vec &a, const Vec &b, const Vec &c,
        const Tex &ta, const Tex &tb, const Tex &tc
    ) : v{a,b,c}, vt{ta, tb, tc} {
        init();
        this->vn[0] = this->vn[1] = this->vn[2] = n;
    }
	Triangle(const Triangle &rhs) = default;

    void init() {
        this->bound = Bound(v[0]) + Bound(v[1]) + Bound(v[2]);
        this->E1 = v[1] - v[0];
        this->E2 = v[2] - v[0];
        this->T1 = vt[1] - vt[0];
        this->T2 = vt[2] - vt[0];
        this->n = Vec::cross(E1, E2).normal(); // 0, 1, 2 counter clockwise is front face
    }

    void debug() const {
        v[0].debug();
        v[1].debug();
        v[2].debug();
        printf("---------------\n");
        vn[0].debug();
        vn[1].debug();
        vn[2].debug();
        printf("---------------\n");
        vt[0].debug();
        vt[1].debug();
        vt[2].debug();
        printf("***************\n");
    }

	__device__ bool intersect(const Ray& ray, Hit& hit) const {
	    Vec S = ray.o - v[0];
        Vec p = Vec::cross(ray.d, E2), q = Vec::cross(S, E1); // temparary variable (common calculation)
        F div = Vec::dot(E1, p);
	    if (fabs(div) < 1e-7) return false;
        F idiv = 1 / div;
	    auto t = Vec::dot(q, E2) * idiv;
	    if (t < hiteps || t > hit.t) return false;
        auto a = Vec::dot(S, p) * idiv;
        if (a < 0 || a > 1) return false;
        auto b = Vec::dot(ray.d, q) * idiv;
        if (b < 0 || b > 1) return false;
        if (a + b > 1) return false;

        Vec P = v[0] + a * E1 + b * E2; // equiv to (1-a-b) * v[0] + a * v[1] + b * v[2];
        Tex T = vt[0] + a * T1 + b * T2; // equiv to (1-a-b) * vt[0] + a * vt[1] + b * vt[2];
        // dpda = E1, dpdb = E2
        // du = (vt[1]-vt[0])_u * da + (vt[2]-vt[0])_u * db
        // dv = (vt[1]-vt[0])_v * da + (vt[2]-vt[0])_v * db
        // [1u 2u] [dadu dadv] = [1 0]
        // [1v 2v] [dbdu dbdv]   [0 1]
        F fac = 1. / (T1.x * T2.y - T2.x * T1.y);
        Vec pu = (E1 * T2.y - E2 * T1.y) * fac;
        Vec pv = (- E1 * T2.x + E2 * T1.x) * fac;
        pu = pu.normal();
        pv = pv.normal();

        // auto norm = ((1-a-b) * vn[0] + a * vn[1] + b * vn[2]).normal(); // TODO normal interpolation
        auto norm = n;
        norm = div < 0 ? norm : -norm; 
        hit.set(t, norm, T, pu, pv);
        return true;
	}

public: // TODO protected
    Vec v[3];
    Vec vn[3];
    Tex vt[3];
    Vec n, E1, E2;
    Tex T1, T2;
    Bound bound;
}; 

#endif // OBJ_TRIANGLE