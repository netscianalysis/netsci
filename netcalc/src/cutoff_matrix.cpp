//
// Created by astokely on 8/28/23.
//

#include "cutoff_matrix.h"

void netcalc::cutoffMatrix(
        CuArray<float> *A,
        CuArray<float> *Z,
        int m,
        int n,
        int d,
        float cutoff,
        int platform
) {
    if (platform == 0) {
        cutoffMatrixGpu(A,
                        Z,
                        m,
                        n,
                        d,
                        cutoff);
    }
}