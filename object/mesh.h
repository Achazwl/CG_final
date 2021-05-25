#ifndef OBJ_MESH
#define OBJ_MESH

#include "base.h"
#include "triangle.h"
#include "../vecs/vector3f.h"

#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>

class Mesh : public Object3D {

public:
    Mesh(const Vec& offset, double scale, const char *filename, Material *material) : Object3D(material) {
        std::ifstream f;
        f.open(filename);
        if (!f.is_open()) {
            std::cout << "Cannot open " << filename << "\n";
            return;
        }
        std::string line;
        std::string vTok("v");
        std::string fTok("f");
        std::string texTok("vt");
        char bslash = '/', space = ' ';
        std::string tok;
        int texID;
        while (true) {
            std::getline(f, line);
            if (f.eof()) {
                break;
            }
            if (line.size() < 3) {
                continue;
            }
            if (line.at(0) == '#') {
                continue;
            }
            std::stringstream ss(line);
            ss >> tok;
            if (tok == vTok) {
                Vec vec;
                ss >> vec.x >> vec.y >> vec.z;
                v.push_back(vec * scale + offset);
            } else if (tok == fTok) {
                if (line.find(bslash) != std::string::npos) {
                    std::replace(line.begin(), line.end(), bslash, space);
                    std::stringstream facess(line);
                    TriangleIndex trig;
                    facess >> tok;
                    for (int ii = 0; ii < 3; ii++) {
                        facess >> trig[ii] >> texID;
                        trig[ii]--;
                    }
                    t.push_back(trig);
                } else {
                    TriangleIndex trig;
                    for (int ii = 0; ii < 3; ii++) {
                        ss >> trig[ii];
                        trig[ii]--;
                    }
                    t.push_back(trig);
                }
            } else if (tok == texTok) {
                std::array<float,2> texcoord;
                ss >> texcoord[0];
                ss >> texcoord[1];
            }
        }
        f.close();

        for (auto& triIndex : t) {
            triangles.emplace_back(new Triangle{v[triIndex[0]], v[triIndex[1]], v[triIndex[2]], material});
        }
        for (auto& triangle : triangles) {
            bound = bound + triangle->bound;
        }
    }

    bool intersect(const Ray &ray, Hit &hit) const override {
        bool hav = false;
        for (auto& triangle: triangles)
            hav |= triangle->intersect(ray, hit);
        return hav;
    }
    std::vector<Triangle*> triangles;

protected:
    std::vector<Vec> v; // nodes

    using TriangleIndex = std::array<int, 3>; // counterclockwise winding is front face
    std::vector<TriangleIndex> t; // which 3 nodes index above form a triangle
};

#endif // OBJ_MESH
