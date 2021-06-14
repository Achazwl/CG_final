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
		img = new F[w*h*c];
		for (int i = 0; i < w*h; ++i)
		for (int j = 0; j < c; ++j)
			img[j*w*h + i] = (F)raw[i*c + j] / 255.0;
		stbi_image_free(raw);

		// auto extent = make_cudaExtent(w, h, c);

		// auto formatDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		// cudaMalloc3DArray(&arr, &formatDesc, extent, cudaArrayLayered);

		// cudaMemcpy3DParms params;
		// memset(&params, 0, sizeof(params));
		// params.srcPos = params.dstPos = make_cudaPos(0, 0, 0);
		// params.srcPtr = make_cudaPitchedPtr(img, w * sizeof(F), w, h);
		// params.dstArray = arr;
		// params.extent = extent;
		// params.kind = cudaMemcpyHostToDevice;
		// cudaMemcpy3D(&params);

		// delete[] img;

		// cudaResourceDesc res_desc;
		// memset(&res_desc, 0, sizeof(cudaResourceDesc));
		// res_desc.resType = cudaResourceTypeArray;
		// res_desc.res.array.array = arr;
		// cudaTextureDesc tex_desc;
		// memset(&tex_desc, 0, sizeof(cudaTextureDesc));
		// tex_desc.filterMode = cudaFilterModeLinear;
		// tex_desc.addressMode[0] = cudaAddressModeWrap;
		// tex_desc.addressMode[1] = cudaAddressModeWrap;
		// tex_desc.addressMode[2] = cudaAddressModeWrap;
		// tex_desc.readMode = cudaReadModeElementType;
		// tex_desc.normalizedCoords = true;
		// cudaCreateTextureObject(&texture_obj, &res_desc, &tex_desc, NULL);
	}
	Material(const Material &rhs) = default;
	// ~Material() { // TODO
	// 	cudaDestroyTextureObject(texture_obj);
	// 	cudaFreeArray(arr);
	// }

	__device__ Vec getTexById(int idx) const {
		return Vec(img[idx], img[w*h+idx], img[w*h*2+idx]);
	}

	__device__ Vec getColor(Tex tex) const { // TODO cudaTexture pool
		if (c == -1) return Kd;
		F pw = tex.x * w, ph = tex.y * h;
		while (pw < eps) pw += w;
		while (pw > w-1-eps) pw -= w;
		if (pw < 1) pw += 1;
		while (ph < eps) ph += h;
		while (ph > h-1-eps) ph -= h;
		if (ph < 1) ph += 1;
		F a = pw - int(pw), b = ph - int(ph);
		return
			((1-a) * getTexById(int(ph  )*w+int(pw)) + a * getTexById(int(ph  )*w+int(pw+1))) * (1-b) +
			((1-a) * getTexById(int(ph+1)*w+int(pw)) + a * getTexById(int(ph+1)*w+int(pw+1))) * b;
	}

	Material* to() const {
		Material* mat = new Material(*this);
		if (c != -1) {
			cudaMalloc((void**)&mat->img, w*h*c*sizeof(F));
			cudaMemcpy(mat->img, img, w*h*c*sizeof(F), cudaMemcpyHostToDevice);
		}

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
	F *img;
	// cudaArray_t arr;
	// cudaTextureObject_t texture_obj;
};

#endif // OBJ_MATERIAL