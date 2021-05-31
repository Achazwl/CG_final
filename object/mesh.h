#ifndef OBJ_MESH
#define OBJ_MESH

#include "base.h"
#include "triangle.h"
#include "../vecs/vector3f.h"
#include <stdio.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>

class MeshFile {

public:
    __host__ MeshFile(const char *filename) {
        FILE *f = freopen(filename, "r", stdin);
        if (f == nullptr) {
            printf("Cannot open file : %s\n", filename);
            return;
        }
        char tok[5];
        while (true) {
            if (scanf("%s", tok) != 1) {
                break;
            }
            if (tok[0] == '#') {
                scanf("%*s[^\n]%*c");
                continue;
            }
            if (tok[0] == 'v' && tok[1] == 0) {
                Vec vec;
                scanf("%lf %lf %lf", &vec.x, &vec.y, &vec.z);
                v.push_back(vec);
            }
            else if (tok[0] == 'f') {
                // if (line.find(bslash) != std::string::npos) {
                //     std::replace(line.begin(), line.end(), bslash, space);
                //     std::stringstream facess(line);
                //     Index trig;
                //     Index texId;
                //     facess >> tok;
                //     int texID[3];
                //     for (int ii = 0; ii < 3; ii++) {
                //         scanf("%d/%d", trig[ii], texId[ii]);
                //         trig[ii]--;
                //     }
                //     t.push_back({trig, texId});
                // }
                // else {
                    Index trig;
                    scanf("%d", &thrust::get<0>(trig)); thrust::get<0>(trig)--;
                    scanf("%d", &thrust::get<1>(trig)); thrust::get<1>(trig)--;
                    scanf("%d", &thrust::get<2>(trig)); thrust::get<2>(trig)--;
                    t.push_back({trig, Index()});
                // }
            }
            // else if (tok[0] == 'v' && tok[1] == 't') {
            //     TextureCoord texcoord;
            //     scanf("%lf %lf", &texcoord[0], &texcoord[1]);
            //     tex.push_back(texcoord);
            // }
        }
        fclose(f);

        // for (auto node : v)
        //     printf("%lf %lf %lf\n", node.x, node.y, node.z);
        // for (auto node : t)
        //     printf("%d %d %d\n", node.first[0], node.first[1], node.first[2]);
    }

public:
    using Index = thrust::tuple<int, int, int>; // counterclockwise winding is front face
    // using TextureCoord = Array<F, 2>;

    thrust::device_vector<Vec> v; // nodes
    thrust::device_vector<thrust::pair<Index, Index>> t; // which 3 nodes index above form a triangle
    // Vector<TextureCoord> tex; // textures
};

#endif // OBJ_MESH
