//
// Created by andy on 4/17/23.
//
#include "generalized_correlation.h"
#include "mutual_information.h"
#include <cmath>

float netcalc::generalizedCorrelationCpu(
        const std::unique_ptr<CuArray<float>> &Xa,
        const std::unique_ptr<CuArray<float>> &Xb,
        int k,
        int n,
        int xd,
        int d) {
    float mutualInformation = netcalc::mutualInformationCpu(
            Xa, Xb, k, n, xd, d);
    if (mutualInformation <= 0.0) {
        return 0.0;
    } else {
        return static_cast<float>(std::sqrt(
                1.0 - static_cast<float>(std::exp(-(2.0 / static_cast<float>(d)) * mutualInformation))));
    }
}

