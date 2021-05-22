#ifndef MESH_H
#define MESH_H

#include "base.h"
#include "triangle.h"
#include "../vecs/vector3f.h"

class Mesh : public Object3D {

public:
    Mesh(const char *filename, Material *m);

    struct TriangleIndex {
        TriangleIndex() {
            x[0] = 0; x[1] = 0; x[2] = 0;
        }
        int &operator[](const int i) { return x[i]; }
        // By Computer Graphics convention, counterclockwise winding is front face
        int x[3]{};
    };

    std::vector<Vector3f> v; // nodes
    std::vector<TriangleIndex> t; // which 3 nodes index above form a triangle

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        bool result = false;
        for (int triId = 0; triId < (int) t.size(); ++triId) {
            TriangleIndex& triIndex = t[triId];
            Triangle triangle(v[triIndex[0]], v[triIndex[1]], v[triIndex[2]], material);
            result |= triangle.intersect(r, h, tmin);
        }
        return result;
    }
};

#endif
