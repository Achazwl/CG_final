#ifndef VECS_MATRIX3F
#define VECS_MATRIX3F

#include "vector3f.h"

struct Matrix {
    static double det(Vec a, Vec b, Vec c) {
        return Matrix::det(
            a.x, a.y, a.z,
            b.x, b.y, b.z,
            c.x, c.y, c.z
        )
    } 

    static double det(  float m00, float m01, float m02,
                        float m10, float m11, float m12,
                        float m20, float m21, float m22 )
    {
        return
            m00 * ( m11 * m22 - m12 * m21 ) -
			m01 * ( m10 * m22 - m12 * m20 ) +
			m02 * ( m10 * m21 - m11 * m20 );
    }
};

#endif // VECS_MATRIX3F