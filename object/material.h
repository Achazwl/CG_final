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
private:
	void loadTexture(const char *texname, int &w, int &h, int &c) { // load texture
		if (texname == nullptr) {
			c = -1;
			return;
		}
		u_char *raw = stbi_load(texname, &w, &h, &c, 0);
		// printf("%s %d %d %d\n", texname, w, h, c); // debug

		for (int j = 0; j < c; ++j) {
			auto img = new F[w*h];
			for (int i = 0; i < w*h; ++i) {
				img[i] = raw[i*c + j] / 255.0;
			}

			// code below are loading texture to cuda texture pool
			auto channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			cudaMallocArray(&texArray, &channelDesc, w, h);
			cudaMemcpy2DToArray(texArray, 0, 0, img, w*sizeof(F), w*sizeof(F), h, cudaMemcpyHostToDevice);

			delete[] img;

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = texArray;

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
		// printf("load to cuda texture memory success!\n"); // debug
	}

	void loadBump(const char *bumpname, int &w, int &h, int &c) {
		if (bumpname == nullptr) {
			cb = -1;
			return;
		}
		u_char *raw = stbi_load(bumpname, &w, &h, &c, 0);
		// printf("%s %d %d %d\n", bumpname, w, h, c); // debug

		for (int j = 0; j < 3; ++j) {
			auto img = new F[w*h];
			for (int i = 0; i < w*h; ++i) {
				img[i] = (raw[i*c + j] / 255.0 - 0.5) * 2;
				/*
				X: -1 to +1 :  Red:     0 to 255
				Y: -1 to +1 :  Green:   0 to 255
				Z:  0 to 1 :  Blue:  128 to 255
				*/
			}

			// code below are loading texture to cuda texture pool
			auto channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			cudaMallocArray(&bumpArray, &channelDesc, w, h);
			cudaMemcpy2DToArray(bumpArray, 0, 0, img, w*sizeof(F), w*sizeof(F), h, cudaMemcpyHostToDevice);
		
			delete[] img;

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = bumpArray;

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(cudaTextureDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 1;

			cudaCreateTextureObject(&bumpObj[j], &resDesc, &texDesc, NULL);
		}
		stbi_image_free(raw);
		// printf("load to cuda texture memory success!\n"); // debug
	}
public:
	Material() = default;
	Material(Vec e, Vec Kd, Vec Ks, F roughness, Refl refl, const char *texname=nullptr, const char *bumpname=nullptr)
	: e(e), Kd(Kd), Ks(Ks), roughness(roughness), refl(refl) {
		loadTexture(texname, w, h, c);
		loadBump(bumpname, wb, hb, cb);
	}
	Material(const Material &rhs) = default;

	__device__ Vec getTexById(F u, F v) const {
		return Vec(
			tex2D<F>(texObj[0], u, v),
			tex2D<F>(texObj[1], u, v),
			tex2D<F>(texObj[2], u, v)
		);
	}

	__device__ Vec getBumpById(F u, F v) const {
		return Vec(
			tex2D<F>(bumpObj[0], u, v),
			tex2D<F>(bumpObj[1], u, v),
			tex2D<F>(bumpObj[2], u, v)
		);
	}

	__device__ Vec getColor(const Tex& tex, Vec& p, Vec& n, const Vec& pu, const Vec& pv) const {
		if (cb != -1) {
			// printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", tex.x, tex.y, p.x, p.y, p.z, pu.x, pu.y, pu.z, pv.x, pv.y, pv.z); // debug
			Vec B = getBumpById(tex.x, tex.y);
			n = (B.z * n + B.x * pu + B.y * pv).normal();
		}
		return c == -1 ? Kd : getTexById(tex.x, 1-tex.y);
	}

	Material* to() const {
		Material* mat = new Material(*this);

		Material *device;
		cudaMalloc((void**)&device, sizeof(Material));
		cudaMemcpy(device, mat, sizeof(Material), cudaMemcpyHostToDevice);
		return device;
	}

public:
	Vec e, Kd, Ks;
	F roughness;
	Refl refl;

	int w, h, c, wb, hb, cb;
	cudaArray_t texArray, bumpArray;
	cudaTextureObject_t texObj[3];
	cudaTextureObject_t bumpObj[3];
};

#endif // OBJ_MATERIAL