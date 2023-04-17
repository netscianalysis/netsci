//
// Created by andy on 4/17/23.
//
#include "generalized_correlation.h"
#include "mutual_information.h"
#include <cmath>

float gpuGeneralizedCorrelation(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n,
        int xd,
        int d
) {
    float mutualInformation =
            gpuMutualInformation(
                    Xa,
                    Xb,
                    k,
                    n,
                    xd,
                    d
            );
    return (float) std::sqrt(
            1.0 - (float) exp(-(2.0 / (float) d) * mutualInformation)
    );
}

