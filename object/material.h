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
    Material(Vec e, Vec col, Refl refl, const char *filename=nullptr): e(e), col(col), refl(refl), filename(filename) {
		if (filename != nullptr) { // load texture
			img = stbi_load(filename, &w, &h, &c, 0);
		} else {
			img = nullptr;
		}
	}

	Vec getcol(double a, double b) const {
		int pw = (int(a * w) % w + w) % w, ph = (int((1-b) * h) % h + h) % h;
		int idx = ph * w * c + pw * c;
		int x = img[idx + 0], y = img[idx + 1], z = img[idx + 2];
		return Vec(x, y, z) / 255.0;
	}

	bool useTexture() const {
		return filename != nullptr;
	}

public:
	Vec e, col;
	Refl refl;
	const char* filename;
	u_char *img; int w, h, c;
};

#endif // OBJ_MATERIAL