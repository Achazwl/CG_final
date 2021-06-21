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
	Material(Vec e, Vec Kd, Vec Ks, F roughness, Refl refl, const char *filename=nullptr)
	: e(e), Kd(Kd), Ks(Ks), roughness(roughness), refl(refl) {
		if (filename == nullptr) { // no texture
			c = -1;
			return;
		}
		// load texture
		u_char *raw = stbi_load(filename, &w, &h, &c, 0);

		for (int j = 0; j < c; ++j) {
			auto img = new F[w*h];
			for (int i = 0; i < w*h; ++i) {
				img[i] = (F)raw[i*c + j] / 255.0;
			}

			// code below are loading texture to cuda texture pool
			auto channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			cudaMallocArray(&cuArray, &channelDesc, w, h);
			cudaMemcpy2DToArray(cuArray, 0, 0, img, w*sizeof(F), w*sizeof(F), h, cudaMemcpyHostToDevice);

			delete[] img; // TODO delete this comment

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cuArray;

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(cudaTextureDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 1;

			cudaCreateTextureObject(&texObj[j], &resDesc, &texDesc, NULL);
		}
		stbi_image_free(raw);
	}
	Material(const Material &rhs) = default;
	~Material() { // TODO uncomment this
		// cudaDestroyTextureObject(texObj);
		// cudaFreeArray(cuArray);
	}

	__device__ Vec getTexById(F u, F v) const {
		return Vec(
			tex2D<F>(texObj[0], u, v),
			tex2D<F>(texObj[1], u, v),
			tex2D<F>(texObj[2], u, v)
		);
	}

	__device__ Vec getColor(Tex tex) const { // TODO cudaTexture pool
		if (c == -1) return Kd;
		return getTexById(tex.x, tex.y);
	}

	Material* to() const {
		Material* mat = new Material(*this);
		// if (c != -1) {
		// 	cudaMalloc((void**)&mat->img, w*h*c*sizeof(F));
		// 	cudaMemcpy(mat->img, img, w*h*c*sizeof(F), cudaMemcpyHostToDevice);
		// }

		Material *device;
		cudaMalloc((void**)&device, sizeof(Material));
		cudaMemcpy(device, mat, sizeof(Material), cudaMemcpyHostToDevice);
		return device;
	}

public:
	Vec e, Kd, Ks;
	F roughness;
	Refl refl;
	int w, h, c;
	cudaArray_t cuArray;
	cudaTextureObject_t texObj[3];
};

#endif // OBJ_MATERIAL