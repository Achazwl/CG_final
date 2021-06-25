#ifndef PT_TRACING
#define PT_TRACING

#include "ray.h"
#include "hit.h"
#include "../utils/rnd.h"
#include "../utils/math.h"
#include "../object/group.h"
#include "../vecs/vector3f.h"

inline __device__ Vec Lambert_Diffuse(Vec Kd) {
	return Kd * M_1_PI;
}

inline __device__ Vec Disney_Burley_Diffuse(Vec Kd, F roughness, F O_N, F I_N, F O_H) {
	F FD90 = 0.5 + 2 * sqr(O_H) * roughness;
	F FdV = 1 + (FD90 - 1) * pow5( 1 - O_N );
	F FdL = 1 + (FD90 - 1) * pow5( 1 - I_N );
	return Kd * ( M_1_PI * FdV * FdL );
}

inline __device__ F D_GGX(F roughness, F H_N) {
	F a = sqr(roughness);
	F a2 = sqr(a);
	F d = sqr(H_N) * (a2 - 1) + 1;
	return a2 / ( M_PI * sqr(d) );
}

inline __device__ F D_Blinn(F roughness, F H_N) {
	F a = sqr(roughness);
	F a2 = sqr(a);
	F n = 2 / a2 - 2;
	return (n+2) / (2*M_PI) * pow(H_N, n);
}

inline __device__ Vec F_Schlick(Vec Ks, F O_H) {
	F Fc = pow5(1 - O_H);
	return (1 - Fc) * Ks + Fc;
}

inline __device__ F Vis_Schlick(F roughness, F O_N, F I_N) { // combine G and normal denominator
	F k = sqr(roughness * 0.5 + 1) * 0.5;
	F Vis_SchlickV = O_N * (1 - k) + k;
	F Vis_SchlickL = I_N * (1 - k) + k;
	return 0.25 / ( Vis_SchlickV * Vis_SchlickL );
}

inline __device__ Vec UE4_Specular(Vec Ks, F roughness, F I_N, F O_N, F H_N, F O_H, F I_H) {
	return D_GGX(roughness, H_N) * F_Schlick(Ks, O_H) * Vis_Schlick(roughness, O_N, I_N);
}

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
		F p = Kd.max(); // TODO why only consider Kd?

		eres = eres + fres * m->e;

		if (++depth > 5) {  // || !p
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
		Vec wh = (wi + wo).normal();
		F I_N = Vec::dot(wi, nl), O_N = Vec::dot(wo, nl), H_N = Vec::dot(wh, nl);
		F O_H = Vec::dot(wo, wh), I_H = Vec::dot(wi, wh);
		// Vec fs = UE4_Specular(m->Ks, m->roughness, I_N, O_N, H_N, O_H, I_H);
		Vec fd = Lambert_Diffuse(Kd);
		Vec f = fd; // TODO fs+fd skew bug
		fres = fres * f * M_PI;
		ray = Ray(x, wi);
	}
} 

#endif // PT_TRACING