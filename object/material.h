#ifndef OBJ_MATERIAL
#define OBJ_MATERIAL

#define STB_IMAGE_IMPLEMENTATION
#include "../images/stb_image.h"
#include "../vecs/vector2f.h"

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

	__device__ Vec getColor(Tex tex) const { // TODO cudaTexture pool
		if (img == nullptr) return Kd;
		int pw = (int(tex.y * w) % w + w) % w, ph = (int((1-tex.x) * h) % h + h) % h; // TODO which direction
		int idx = (ph * w + pw) * c;
		int x = img[idx + 0], y = img[idx + 1], z = img[idx + 2];
		return Vec(x, y, z) / 255.0;
	}

	Material* to() const {
		Material* mat = new Material(*this);
		if (img != nullptr) {
			cudaMalloc((void**)&mat->img, w*h*c*sizeof(u_char));
			cudaMemcpy(mat->img, img, w*h*c*sizeof(u_char), cudaMemcpyHostToDevice);
		}

		Material *device;
		cudaMalloc((void**)&device, sizeof(Material));
		cudaMemcpy(device, mat, sizeof(Material), cudaMemcpyHostToDevice);
		return device;
	}

public:
	Vec e, Kd, Ks;
	Refl refl;
	int w, h, c; u_char *img;
};

#endif // OBJ_MATERIAL