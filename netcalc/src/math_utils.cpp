//
// Created by astokely on 5/16/23.
//
#include "math_utils.h"

void mean(
        CuArray<float> *a,
        CuArray<float> *u,
        int m,
        int n,
        int platform
) {

    m = a->size() / n;
    u->init(1, m);
    if (platform == 0) {
        meanGpu(a, u, m, n);
    }
}

void standardDeviation(
        CuArray<float> *a,
        CuArray<float> *u,
        CuArray<float> *sigma,
        int m,
        int n,
        int platform
) {

    m = a->size() / n;
    u->init(1, m);
    sigma->init(1, m);
    if (platform == 0) {
        standardDeviationGpu(a, u, sigma, m, n);
    }
}
