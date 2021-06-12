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
	// return Vec(0.9, 0.2, 0.2);

	while (true) {
		Hit hit(1e20);
		if (!group->intersect(ray, hit)) return eres;

		Vec x = ray.At(hit.t);
		bool into = Vec::dot(hit.n, ray.d) < 0;
		Vec nl = into ? hit.n : -hit.n;
		Material* m = hit.m;
		RGB f = m->getColor(hit.tex); // TODO: texture 
		F p = f.max();

		eres = eres + fres * m->e;

		if (++depth > 5 || !p) { // TODO: bigger decay limit than 5?
			if (rnd(1, st) < p) f = f / p; // expeted = p * (f / p) + (1 - p) * 0 = f
			else return eres;
		}

		// if (m->refl == Refl::DIFFUSE) { // TODO fix
		// 	F a, b, c; rndCosWeightedHSphere(a, b, c, st);
		// 	Vec u, v, w = nl; Vec::orthoBase(w, u, v);
		// 	Vec wo = -ray.d;
		// 	Vec wi = (u * a + v * b + w * c).normal();
		// 	Vec wh = (wi + wo).normal();
		// 	F cosi = Vec::dot(wi, nl), coso = Vec::dot(wo, nl), cosh = Vec::dot(wh, nl);
		// 	// Fresnel(wi, wh)
		// 		Vec F0 = m->Ks;
		// 		Vec Fr = F0 + (Vec(1,1,1) - F0) * pow5(1-Vec::dot(wi, wh));
		// 	// D(wh)
		// 		F ax = 0.8, ay = 0.7;
		// 		F sinh = sqrt(1 - sqr(cosh));
		// 		F e = (sqr(Vec::dot(wh, u) / ax) + sqr(Vec::dot(wh, v) / ay)) * sqr(sinh/cosh);
		// 		F D = 1 / (M_PI * ax * ay * pow4(cosh) * sqr(1+e));
		// 	// G(wo, wi)
		// 		auto Lambda = [=](const Vec& w) {
		// 			F cosw = Vec::dot(w, nl);
		// 			F sinw = sqrt(1 - sqr(cosw));
		// 			F tanw = sinw / cosw;
		// 			F alpha = sqrt(sqr(Vec::dot(w, u) * ax) + sqr(Vec::dot(w, v) * ay));
		// 			F a = 1 / (alpha * tanw);
		// 			return a >= 1.6 ? 0 : (1 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
		// 		};
		// 		F G = 1 / (1 + Lambda(wo) + Lambda(wi));
		// 	auto fr = D * G * Fr / (4 * cosi * coso);
		// 	fres = fres * fr * M_PI;
		// 	ray = Ray(x, wi);
		// }
		if (m->refl == Refl::DIFFUSE) {
			F a, b, c; rndCosWeightedHSphere(a, b, c, st);
			Vec u, v, w = nl; Vec::orthoBase(w, u, v);
			Vec wi = (u * a + v * b + w * c).normal();
			fres = fres * f;
			ray = Ray(x, wi);
		}
		else if (m->refl == Refl::GLASS) {
			F a, b, c; rndCosWeightedHSphere(a, b, c, st);
			Vec u, v, w = nl; Vec::orthoBase(w, u, v);
			Vec wo = -ray.d;
			Vec wi = (u * a + v * b + w * c).normal();
			Vec wh = (wi + wo).normal();
			F cosi = Vec::dot(wi, nl), coso = Vec::dot(wo, nl), cosh = Vec::dot(wh, nl);

			Vec diffuse = (28. * M_1_PI / 23.) * f * (Vec(1,1,1) - m->Ks) * (1-pow5(1-0.5*cosi)) * (1-pow5(1-0.5*coso));
			// Fresnel(wi, wh)
				Vec F0 = m->Ks;
				Vec Fr = F0 + (Vec(1,1,1) - F0) * pow5(1-Vec::dot(wi, wh));
			// D(wh)
				F ax = 0.1, ay = 0.1; // TODO
				F sinh = sqrt(1 - sqr(cosh));
				F e = (sqr(Vec::dot(wh, u) / ax) + sqr(Vec::dot(wh, v) / ay)) * sqr(sinh/cosh);
				F D = 1 / (M_PI * ax * ay * pow4(cosh) * sqr(1+e));
			Vec specular = D / ( 4 * Vec::dot(wi, wh) * max(cosi, coso) ) * Fr;
			auto fr = diffuse + specular;
			fres = fres * fr * M_PI;
			ray = Ray(x, wi);
		}
	}
} 

#endif // PT_TRACING