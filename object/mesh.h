#ifndef OBJ_MESH
#define OBJ_MESH

#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_DOUBLE
#include "objloader.h"

#include "base.h"
#include "triangle.h"
#include "../vecs/vector3f.h"

#include <cstdio>
#include <iostream>
#include <vector>
#include <array>

class MeshFile {

public:
    MeshFile(std::string filename) {
        std::string basepth = std::string("mesh")+'/'+filename+'/';
        std::string inputfile = basepth+filename+".obj";
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = basepth; // Path to material files

        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(inputfile, reader_config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyObjReader: " << reader.Error();
            }
            exit(1);
        }

        if (!reader.Warning().empty()) {
            std::cout << "TinyObjReader: " << reader.Warning();
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        for (const auto& material : materials) {
            Vec Ke = Vec(
                material.emission[0],
                material.emission[1],
                material.emission[2]
            );
            Vec Kd = Vec(
                material.diffuse[0],
                material.diffuse[1],
                material.diffuse[2]
            );
            Vec Ks = Vec(
                material.specular[0],
                material.specular[1],
                material.specular[2]
            );
            this->materials.emplace_back(
                Ke, Kd, Ks, material.roughness,
                Refl::GLASS
                ,
                material.diffuse_texname.empty() 
                    ? nullptr
                    : (basepth+material.diffuse_texname).c_str()
            );
        }

		for (const auto& shape : shapes) {
            const auto& vec_ids = shape.mesh.indices;
            const auto& mat_ids = shape.mesh.material_ids;
            for (int i = 0; i < mat_ids.size(); ++i) {
                const auto& mat_id = mat_ids[i];

                std::vector<Vec> v;
                std::vector<Vec> vn;
                std::vector<Tex> vt;
                for (int k = 0; k < 3; ++k) {
                    const auto& vec_id = vec_ids[3*i+k];
                    v.emplace_back(
                        attrib.vertices[3*vec_id.vertex_index+0],
                        attrib.vertices[3*vec_id.vertex_index+1],
                        attrib.vertices[3*vec_id.vertex_index+2]
                    );
                    vn.emplace_back(
                        attrib.normals[3*vec_id.normal_index+0],
                        attrib.normals[3*vec_id.normal_index+1],
                        attrib.normals[3*vec_id.normal_index+2]
                    );
                    vt.emplace_back(
                        attrib.texcoords[2*vec_id.texcoord_index+0],
                        attrib.texcoords[2*vec_id.texcoord_index+1]
                    );
                }
                this->triangles.emplace_back(
                    v[0], v[1], v[2],
                    vn[0], vn[1], vn[2],
                    vt[0], vt[1], vt[2]
                );
                this->material_ids.emplace_back(
                    mat_id
                );
                // this->triangles.back().debug();
            }
		}
    }
public:
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<int> material_ids;
};

#endif // OBJ_MESH
