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
        char tok[20];
        while (true) {
            int eof = scanf("%s", tok) != 1;
            if (eof) break;
            if (tok[0] == '#') {
                eof = scanf("%*s[^\n]%*c");
            }
            else if (tok[0] == 'v' && tok[1] == 0) {
                Vec vec;
                eof = scanf("%lf %lf %lf", &vec.x, &vec.y, &vec.z);
                v.push_back(vec);
            }
            else if (tok[0] == 'f') {
                Index trig;
                for (int ii = 0; ii < 3; ++ii) {
                    eof = scanf("%d", &trig[ii]);
                    trig[ii]--;
                }
                t.push_back(trig);
            }
            else if (tok[0] == 'v' && tok[1] == 't') {
                F u, v;
                eof = scanf("%lf %lf", &u, &v);
            }
            else if (tok[0] == 'v' && tok[1] == 'n') {
                Vec vec;
                eof = scanf("%lf %lf %lf", &vec.x, &vec.y, &vec.z);
            }
            else {
                static char tmp[100007], *unused;
                printf("unknown type in mesh file: %s\n", tok);
                unused = fgets(tmp, 100000, stdin);
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
