//
// Created by andy on 4/17/23.
//
#include <stdexcept>
#include "mutual_information.h"
#include "generalized_correlation.h"

void netcalc::generalizedCorrelation(
        CuArray<float> *X,
        CuArray<float> *R,
        CuArray<int> *ab,
        int k,
        int n,
        int xd,
        int d,
        const std::string &platform
) {
    R->init(
            1,
            ab->m()
    );
    float (*generalizedCorrelationFunction)(
            CuArray<float> *,
            CuArray<float> *,
            int,
            int,
            int,
            int
    );
    if (platform == "gpu") {
        generalizedCorrelationFunction = netcalc::generalizedCorrelationGpu;
    } else if (platform == "cpu") {
        generalizedCorrelationFunction = netcalc::generalizedCorrelationCpu;
    } else {
        throw std::runtime_error("Invalid platform");
    }
    for (int i = 0; i < ab->m(); i++) {
        int a = ab->get(i,
                        0);
        int b = ab->get(i,
                        1);
        auto Xa = new CuArray<float>;
        auto Xb = new CuArray<float>;
        Xa->fromCuArrayShallowCopy(
                X,
                a,
                a,
                1,
                X->n()
        );
        Xb->fromCuArrayShallowCopy(
                X,
                b,
                b,
                1,
                X->n()
        );
        R->set(
                generalizedCorrelationFunction(
                        Xa,
                        Xb,
                        k,
                        n,
                        xd,
                        d
                ),
                0,
                i
        );
        delete Xa;
        delete Xb;
    }
}

