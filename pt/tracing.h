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
	RGB f = hit.m->col; 
	double p = f.max();

	if (++depth > DEPTH_DECAY || !p) {
		if (rnd() < p) f = f / p; // expeted = p * (f / p) + (1 - p) * 0 = f
		else return hit.m->e;
	}

	if (hit.m->refl == Refl::DIFFUSE)
	{                  
		double r1=rnd(2*M_PI), r2=rnd(), r2s=sqrt(r2); 
		Vec w = nl, u = w.ortho(), v = Vec::cross(w, u); 
		Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).normal(); 
		return hit.m->e + f * tracing(group, Ray(x, d), depth, 1);
	}
	else {
		Ray reflRay(x, ray.d - 2 * Vec::dot(ray.d, nl) * nl);   
		if (hit.m->refl == Refl::MIRROR)
		{           
			return hit.m->e + f * tracing(group, reflRay, depth); 
		}
		else if (hit.m->refl == Refl::GLASS)
		{
			double na = into ? 1 : 1.5, nb = into ? 1.5 : 1;
			double cosi = -Vec::dot(ray.d, nl);
			double nn = na / nb;
			double sinr2 = sqr(nn) * (1 - sqr(cosi)); // sin(r)^2 = 1 - (na/nb sin(i))^2
			if (sinr2 > 1) { // 超过临界角，全反射, 按照MIRROR的方式算
				return hit.m->e + f * tracing(group, reflRay, depth); 
			}
			double cosr = sqrt(1-sinr2);
			Vec rd = nn*(ray.d + nl * cosi) + (-nl) * cosr;

			double F0 = sqr(nn-1)/sqr(nn+1);
			double Re = F0 + (1-F0) * pow(1-cosi, 5), Tr = 1-Re; // 反射折射比

			if (depth > 2) { // Russian roulette 避免递归分支过多
				double P = .25 + 0.5 * Re; // 直接按Re做P会出现极端情况，缩放一下，保证有上下界[0.25,0.75]
				return hit.m->e + f * (
					rnd()<P ? tracing(group, reflRay,depth) * Re/P : tracing(group, Ray(x, rd), depth) * Tr/(1-P)
				);
			}
			else {
				return hit.m->e + f * (
					tracing(group, reflRay,depth) * Re + tracing(group, Ray(x, rd), depth) * Tr
				);
			}
		}
	}
	return {};
} 

#endif // PT_TRACING