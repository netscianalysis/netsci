//
// Created by andy on 4/11/23.
//
#include <iostream>
#include "mutual_information.h"

void netcalc::mutualInformation(
        CuArray<float> *X,
        CuArray<float> *I,
        CuArray<int> *ab,
        int k,
        int n,
        int xd,
        int d,
        const std::string &platform
) {
    float (*mutualInformationFunction)(
            CuArray<float> *,
            CuArray<float> *,
            int,
            int,
            int,
            int
    );
    if (platform == "gpu") {
        mutualInformationFunction = &netcalc::mutualInformationGpu;
    } else if (platform == "cpu") {
        mutualInformationFunction = &netcalc::mutualInformationCpu;
    } else {
        throw std::runtime_error("Invalid platform");
    }
    I->init(
            1,
            ab->m()
    );

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
            I->set(
                    mutualInformationFunction(
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
