//
// Created by astokely on 5/16/23.
//
#include "math_utils.h"

void mean(
        const std::unique_ptr<CuArray<float>> &a,
        const std::unique_ptr<CuArray<float>> &u,
        int m,
        const int n,
        const int platform) {

    m = a->size() / n;
    u->init(1, m);
    if (platform == 0) {
        meanGpu(a, u, m, n);
    }
}

void standardDeviation(
        const std::unique_ptr<CuArray<float>> &a,
        const std::unique_ptr<CuArray<float>> &u,
        const std::unique_ptr<CuArray<float>> &sigma,
        int m,
        const int n,
        const int platform) {

    m = a->size() / n;
    u->init(1, m);
    sigma->init(1, m);
    if (platform == 0) {
        standardDeviationGpu(a, u, sigma, m, n);
    }
}
