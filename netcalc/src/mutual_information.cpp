//
// Created by andy on 4/11/23.
//
#include <iostream>
#include "mutual_information.h"

int mutualInformation(
        CuArray<float> *X,
        CuArray<float> *I,
        CuArray<int> *ab,
        int k,
        int n,
        int xd,
        int d,
        int platform
) {
    I->init(
            1, ab->m()
    );

    for (int i = 0; i < ab->m(); i++) {
        int a = ab->get(i, 0);
        int b = ab->get(i, 1);
        std::cout << a << " " << b << std::endl;
        auto Xa = new CuArray<float>;
        auto Xb = new CuArray<float>;
        Xa->fromCuArrayDeepCopy(
                X,
                a,
                a,
                1, X->n()
        );
        Xb->fromCuArrayDeepCopy(
                X,
                b,
                b,
                1, X->n()
        );
        if (platform == 0)
            I->set(
                    gpuMutualInformation(
                            Xa, Xb, k, n, xd, d
                    ),
                    0, i
            );
        else if (platform == 1)
            I->set(
                    cpuMutualInformation(
                            Xa, Xb, k, n, xd, d
                    ),
                    0, i
            );
        else {
            throw std::runtime_error("Invalid platform");
        }
        delete Xa;
        delete Xb;
    }
    return platform;
}
