#ifndef OBJ_MESH
#define OBJ_MESH

#include "base.h"
#include "triangle.h"
#include "../vecs/vector3f.h"
#include <stdio.h>
#include <vector>
#include <array>

class MeshFile {

public:
    MeshFile(const char *filename) {
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
            if (tok[0] == 'v') {
                Vec vec;
                scanf("%lf %lf %lf", &vec.x, &vec.y, &vec.z);
                v.push_back(vec);
            }
            else if (tok[0] == 'f') {
                Index trig;
                for (int ii = 0; ii < 3; ++ii) {
                    scanf("%d", &trig[ii]);
                    trig[ii]--;
                }
                t.push_back(trig);
            }
        }
        fclose(f);
    }

public:
    using Index = std::array<int, 3>; // counterclockwise winding is front face

    std::vector<Vec> v; // nodes
    std::vector<Index> t; // which 3 nodes index above form a triangle
};

#endif // OBJ_MESH
