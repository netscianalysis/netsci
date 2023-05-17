//
// Created by andy on 4/17/23.
//
#include "generalized_correlation.h"
#include "mutual_information.h"
#include <cmath>

float generalizedCorrelationGpu(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n,
        int xd,
        int d
) {
    float mutualInformation =
            mutualInformationGpu(
                    Xa,
                    Xb,
                    k,
                    n,
                    xd,
                    d
            );
    if (mutualInformation <= 0.0) {
        return 0.0;
    } else {
        return (float) std::sqrt(
                1.0 - (float) exp(-(2.0 / (float) d) * mutualInformation)
        );
    }
}

