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
	Material(Vec e, Vec Kd, Vec Ks, Refl refl, const char *filename=nullptr): e(e), Kd(Kd), Ks(Ks), refl(refl) {
		if (filename != nullptr) { // load texture
			img = stbi_load(filename, &w, &h, &c, 0);
		} else {
			img = nullptr;
		}	
	}
	Material(const Material &rhs) = default;

	__device__ Vec getColor(F u, F v) const {
		if (img == nullptr) return Kd;
		int pw = (int(u * w) % w + w) % w, ph = (int((1-v) * h) % h + h) % h;
		int idx = (ph * w + pw) * c;
		int x = img[idx + 0], y = img[idx + 1], z = img[idx + 2];
		return Vec(x, y, z) / 255.0;
	}

	__host__ Material* to() const {
		Material* mat = new Material(*this);
		if (img == nullptr) return mat;
		cudaMalloc((void**)&mat->img, w*h*c*sizeof(u_char));
		cudaMemcpy(mat->img, img, w*h*c*sizeof(u_char), cudaMemcpyHostToDevice);
		return mat;
	}

public:
	Vec e, Kd, Ks;
	Refl refl;
	int w, h, c; u_char *img;
};

#endif // OBJ_MATERIAL