#ifndef PT_TRACING
#define PT_TRACING

#include "ray.h"
#include "hit.h"
#include "../utils/rnd.h"
#include "../utils/math.h"
#include "../object/group.h"
#include "../vecs/vector3f.h"

inline __device__ Vec tracing(Group *group, Ray ray, curandState *st) {
	Vec eres = Vec(0,0,0);
	Vec fres = Vec(1,1,1);
	int depth = 0;
	//a + b(c+d()) = a+bc + bd() // recursive -> non-recursive

	while (true) {
		Hit hit(1e20);
		if (!group->intersect(ray, hit)) return eres;

		Vec x = ray.At(hit.t);
		bool into = Vec::dot(hit.n, ray.d) < 0;
		Material* m = hit.m;
		RGB Kd = m->getColor(hit.tex, x, hit.n, hit.pu, hit.pv);
		Vec nl = into ? hit.n : -hit.n;
		F p = Kd.max();

		eres = eres + fres * m->e;

		if (++depth > 5 || depth > 20) {  // || !p
			if (rnd(1, st) < p) { // expeted = p * (f / p) + (1 - p) * 0 = f)
				fres = fres / p;
			}
			else {
				return eres;
			}
		}

		F a, b, c; rndCosWeightedHSphere(a, b, c, st);
		Vec u, v, w = nl; Vec::orthoBase(w, u, v);
		Vec wo = -ray.d;
		Vec wi = (u * a + v * b + w * c).normal();
		fres = fres * Kd;
		if (m->refl == Refl::DIFFUSE) {
			ray = Ray(x, wi);
		}
		else if (m->refl == Refl::MIRROR) {
			ray = Ray(x, ray.d - 2 * Vec::dot(ray.d, nl) * nl);   
		}
		else if (m->refl == Refl::GLASS) {
			F na = into ? 1 : 1.5, nb = into ? 1.5 : 1;
			F cosi = -Vec::dot(ray.d, nl);
			F nn = na / nb;
			F sinr2 = sqr(nn) * (1 - sqr(cosi)); // sin(r)^2 = 1 - (na/nb sin(i))^2
			if (sinr2 > 1) { // 超过临界角，全反射, 按照MIRROR的方式算
				ray = Ray(x, ray.d - 2 * Vec::dot(ray.d, nl) * nl);   
			}
			F cosr = sqrt(1-sinr2);
			Vec rd = nn*(ray.d + nl * cosi) + (-nl) * cosr;

			F F0 = sqr(nn-1)/sqr(nn+1);
			F Re = F0 + (1-F0) * pow(1-cosi, 5), Tr = 1-Re; // 反射折射比

			{ // Russian roulette 避免递归分支过多
				F P = .25 + 0.5 * Re; // 直接按Re做P会出现极端情况，缩放一下，保证有上下界[0.25,0.75]
				if (rnd(1, st) < P) {
					ray = Ray(x, ray.d - 2 * Vec::dot(ray.d, nl) * nl);   
					fres = fres * Re / P;
				} else {
					ray = Ray(x, rd);
					fres = fres * Tr / (1-P);
				}
			}
		}
	}
} 

#endif // PT_TRACING