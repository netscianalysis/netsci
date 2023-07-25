//
// Created by astokely on 5/16/23.
//
#include <stdexcept>
#include "math_utils.h"

void mean(
        CuArray<float> *a,
        CuArray<float> *u,
        int m,
        int n,
        const std::string &platform
) {

    m = a->size() / n;
    u->init(1, m);
    void (*meanFunction)(CuArray<float> *, CuArray<float> *, int, int);
    if (platform == "gpu") {
        meanFunction = meanGpu;
    }
    else {
        throw std::runtime_error("Invalid platform");
    }
    meanFunction(a, u, m, n);
}

void standardDeviation(
        CuArray<float> *a,
        CuArray<float> *u,
        CuArray<float> *sigma,
        int m,
        int n,
        const std::string &platform
) {

    m = a->size() / n;
    u->init(1, m);
    sigma->init(1, m);
    void (*standardDeviationFunction)(CuArray<float> *, CuArray<float> *, CuArray<float> *, int, int);
    if (platform == "gpu") {
        standardDeviationFunction = standardDeviationGpu;
    }
    else {
        throw std::runtime_error("Invalid platform");
    }
    standardDeviationFunction(a, u, sigma, m, n);
}
