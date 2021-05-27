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

	if (m->refl == Refl::DIFFUSE)
	{                  
		auto [a, b, c] = rndHSphere();
		auto [u, v, w] = Vec::orthoBase(nl);
		Vec d = (u * a + v * b + w * c).normal();
		return m->e + f * tracing(group, Ray(x, d), depth, 1);
	}
	else {
		Ray reflRay(x, ray.d - 2 * Vec::dot(ray.d, nl) * nl);   
		if (m->refl == Refl::MIRROR)
		{           
			return m->e + f * tracing(group, reflRay, depth); 
		}
		else if (m->refl == Refl::GLASS)
		{
			F na = into ? 1 : 1.5, nb = into ? 1.5 : 1;
			F cosi = -Vec::dot(ray.d, nl);
			F nn = na / nb;
			F sinr2 = sqr(nn) * (1 - sqr(cosi)); // sin(r)^2 = 1 - (na/nb sin(i))^2
			if (sinr2 > 1) { // 超过临界角，全反射, 按照MIRROR的方式算
				return m->e + f * tracing(group, reflRay, depth); 
			}
			F cosr = sqrt(1-sinr2);
			Vec rd = nn*(ray.d + nl * cosi) + (-nl) * cosr;

			F F0 = sqr(nn-1)/sqr(nn+1);
			F Re = F0 + (1-F0) * pow(1-cosi, 5), Tr = 1-Re; // 反射折射比

			if (depth > 2) { // Russian roulette 避免递归分支过多
				F P = .25 + 0.5 * Re; // 直接按Re做P会出现极端情况，缩放一下，保证有上下界[0.25,0.75]
				return m->e + f * (
					rnd()<P ? tracing(group, reflRay,depth) * Re/P : tracing(group, Ray(x, rd), depth) * Tr/(1-P)
				);
			}
			else {
				return m->e + f * (
					tracing(group, reflRay,depth) * Re + tracing(group, Ray(x, rd), depth) * Tr
				);
			}
		}
	}
	return {};
} 

#endif // PT_TRACING