#ifndef OBJ_MATERIAL
#define OBJ_MATERIAL

#define STB_IMAGE_IMPLEMENTATION
#include "../images/stb_image.h"

enum class Refl { // reflect types
	DIFFUSE,
	MIRROR,
	GLASS,
}; 

class Material {
public:
	Material() = default;
	Material(Vec e, Vec Kd, Vec Ks, Refl refl): e(e), Kd(Kd), Ks(Ks), refl(refl) { }
	Material(const Material &rhs) = default;

public:
	Vec e, Kd, Ks;
	Refl refl;
};

#endif // OBJ_MATERIAL