#ifndef PT_TRACING
#define PT_TRACING

#include "ray.h"
#include "../utils/rnd.h"
#include "../utils/math.h"
#include "../object/refl.h"
#include "../object/group.h"
#include "../vecs/vector3f.h"

inline Vec tracing(const Group &group, const Ray &ray, int depth, int E = 1) {
	static constexpr int DEPTH_DECAY = 5; // TODO: bigger ?

	double t; int id = 0;
	if (!group.intersect(ray, t, id)) return Vec();
	const Sphere *obj = dynamic_cast<Sphere*>(group[id]);

	Vec x = ray.o + ray.d * t;
	Vec nl = (x - obj->c).normal();
	nl = Vec::dot(nl, ray.d) < 0 ? nl : -nl;
	RGB f = obj->col; 
	double p = f.max();

	if (++depth > DEPTH_DECAY || !p) {
		if (rnd() < p) f = f / p; // expeted = p * (f / p) + (1 - p) * 0 = f
		else return obj->e;
	}

	if (obj->refl == Refl::DIFFUSE)
	{                  
		double r1=rnd(2*M_PI), r2=rnd(), r2s=sqrt(r2); 
		Vec w = nl, u = w.ortho(), v = Vec::cross(w, u); 
		Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).normal(); 

		Vec e{};
		for (int i = 0; i < group.size(); ++i){
			Sphere *s = dynamic_cast<Sphere*>(group[i]);
			if (s->e.max()<=0) continue;
			
			Vec sw = (s->c-x).normal(), su = sw.ortho(), sv = Vec::cross(sw, su).normal();
			double cos_a_max = sqrt(1 - sqr(s->r) / (s->c-x).norm2()); // 切线张角
			double cos_a = 1 - (1 - cos_a_max) * rnd(); // 不均匀的，比a_max小的a张角
			double sin_a = sqrt(1 - sqr(cos_a*cos_a));
			double phi = rnd(2*M_PI);
			Vec l = su*cos(phi)*sin_a + sv*sin(phi)*sin_a + sw*cos_a;

			if (group.intersect(Ray(x,l), t, id) && id==i){  // 无遮挡大前提
				double omega = 2*M_PI*(1-cos_a_max); // 圆锥立体角
				e = e + Vec::mult(f, s->e * Vec::dot(l, nl) * omega) * M_1_PI;  // 1/pi for brdf
			}
		}

		return obj->e * E + e + Vec::mult(f, tracing(group, Ray(x, d), depth, 0)); // E置0，下一层如果随机恰能反射到光源，那就不重复计算了
	}
	else {
		Ray reflRay(x, ray.d - 2 * Vec::dot(ray.d, nl) * nl);   
		if (obj->refl == Refl::MIRROR)
		{           
			return obj->e + Vec::mult(f, tracing(group, reflRay, depth)); 
		}
		else if (obj->refl == Refl::GLASS)
		{
			bool into = (x-obj->c).dot(nl) > 0; // 是否从物体外侧面射入
			double na = into ? 1 : 1.5, nb = into ? 1.5 : 1;
			double cosi = -Vec::dot(ray.d, nl);
			double nn = na / nb;
			double sinr2 = sqr(nn) * (1 - sqr(cosi)); // sin(r)^2 = 1 - (na/nb sin(i))^2
			if (sinr2 > 1) { // 超过临界角，全反射, 按照MIRROR的方式算
				return obj->e + Vec::mult(f, tracing(group, reflRay, depth)); 
			}
			double cosr = sqrt(1-sinr2);
			Vec rd = nn*(ray.d + nl * cosi) + (-nl) * cosr;

			double F0 = sqr(nn-1)/sqr(nn+1);
			double Re = F0 + (1-F0) * pow(1-cosi, 5), Tr = 1-Re; // 反射折射比

			if (depth > 2) { // Russian roulette 避免递归分支过多
				double P = .25 + 0.5 * Re; // 直接按Re做P会出现极端情况，缩放一下，保证有上下界[0.25,0.75]
				return obj->e + Vec::mult(
					f,
					rnd()<P ? tracing(group, reflRay,depth) * Re/P : tracing(group, Ray(x, rd), depth) * Tr/(1-P)
				);
			}
			else {
				return obj->e + Vec::mult(
					f,
					tracing(group, reflRay,depth) * Re + tracing(group, Ray(x, rd), depth) * Tr
				);
			}
		}
	}
} 

#endif // PT_TRACING