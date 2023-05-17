//
// Created by astokely on 5/16/23.
//

#ifndef NETSCI_MATH_UTILS_H
#define NETSCI_MATH_UTILS_H

#include "cuarray.h"

void mean(
        CuArray<float> *a,
        CuArray<float> *u,
        int m,
        int n,
        int platform
);

void meanGpu(
        CuArray<float> *a,
        CuArray<float> *u,
        int m,
        int n
);

void standardDeviation(
        CuArray<float> *a,
        CuArray<float> *u,
        CuArray<float> *sigma,
        int m,
        int n,
        int platform
);

void standardDeviationGpu(
        CuArray<float> *a,
        CuArray<float> *u,
        CuArray<float> *sigma,
        int m,
        int n
);

#endif //NETSCI_MATH_UTILS_H
