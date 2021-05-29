#ifndef PT_TRACING
#define PT_TRACING

#include "ray.h"
#include "../utils/rnd.h"
#include "../utils/math.h"
#include "../object/group.h"
#include "../vecs/vector3f.h"

inline Vec tracing(const Group &group, const Ray &ray, int depth, int E = 1) {
	static constexpr int DEPTH_DECAY = 5; // TODO: bigger ?

	Hit hit(1e20);
	if (!group.intersect(ray, hit)) return Vec();

	Vec x = ray.At(hit.t);
	bool into = Vec::dot(hit.n, ray.d) < 0;
	Vec nl = into ? hit.n : -hit.n;
	Material* m = hit.o->material;
	RGB f = hit.o->getColor(x); 
	F p = f.max();

	if (++depth > DEPTH_DECAY || !p) {
		if (rnd() < p) f = f / p; // expeted = p * (f / p) + (1 - p) * 0 = f
		else return m->e;
	}

	if (m->refl == Refl::DIFFUSE) { // TODO fix
		auto [a, b, c] = rndCosWeightedHSphere();
		auto [u, v, w] = Vec::orthoBase(nl);
		Vec wo = -ray.d;
		Vec wi = (u * a + v * b + w * c).normal();
		Vec wh = (wi + wo).normal();
		F cosi = Vec::dot(wi, nl), coso = Vec::dot(wo, nl), cosh = Vec::dot(wh, nl);

		// Fresnel(wi, wh)
			Vec F0 = m->Ks;
			Vec Fr = F0 + (Vec(1,1,1) - F0) * pow5(1-Vec::dot(wi, wh));
		// D(wh)
			F ax = 0.8, ay = 0.7;
			F sinh = sqrt(1 - sqr(cosh));
			F e = (sqr(Vec::dot(wh, u) / ax) + sqr(Vec::dot(wh, v) / ay)) * sqr(sinh/cosh);
			F D = 1 / (M_PI * ax * ay * pow4(cosh) * sqr(1+e));
		// G(wo, wi)
			auto Lambda = [=](const Vec& w) {
				F cosw = Vec::dot(w, nl);
				F sinw = sqrt(1 - sqr(cosw));
				F tanw = sinw / cosw;
				F alpha = sqrt(sqr(Vec::dot(w, u) * ax) + sqr(Vec::dot(w, v) * ay));
				F a = 1 / (alpha * tanw);
				return a >= 1.6 ? 0 : (1 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
			};
			F G = 1 / (1 + Lambda(wo) + Lambda(wi));
		auto fr = f + D * G * Fr / (4 * cosi * coso);
		return m->e + fr * tracing(group, Ray(x, wi), depth, 1);
	}
	else if (m->refl == Refl::GLASS) {
		auto [a, b, c] = rndCosWeightedHSphere();
		auto [u, v, w] = Vec::orthoBase(nl);
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
		Vec specular = D / ( 4 * Vec::dot(wi, wh) * std::max(cosi, coso) ) * Fr;
		auto fr = diffuse + specular;
		return m->e + fr * tracing(group, Ray(x, wi), depth, 1) * M_PI;
	}
	return Vec{};
} 

#endif // PT_TRACING