//
// Created by astokely on 8/28/23.
//

#ifndef NETSCI_CUTOFF_MATRIX_H
#define NETSCI_CUTOFF_MATRIX_H

#include "cuarray.h"

namespace netcalc {

    void cutoffMatrixGpu(
            CuArray<float> *A,
            CuArray<float> *Z,
            int m,
            int n,
            int d,
            float cutoff
    );

    void cutoffMatrix(
            CuArray<float> *A,
            CuArray<float> *Z,
            int m,
            int n,
            int d,
            float cutoff,
            int platform
    );
}

#endif //NETSCI_CUTOFF_MATRIX_H
