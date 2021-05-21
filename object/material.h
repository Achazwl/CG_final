#ifndef OBJ_MATERIAL
#define OBJ_MATERIAL

enum class Refl { // reflect types
	DIFFUSE,
	MIRROR,
	GLASS,
}; 

class Material {
public:
    Material(Vec e, Vec col, Refl refl): e(e), col(col), refl(refl) {}
	Material(const Material& rhs) : e(rhs.e), col(rhs.col), refl(rhs.refl) {}
public:
	Vec e, col;
	Refl refl;
};

#endif // OBJ_MATERIAL