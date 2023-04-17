//
// Created by andy on 4/17/23.
//

#ifndef NETSCI_GENERALIZED_CORRELATION_H
#define NETSCI_GENERALIZED_CORRELATION_H
#include "cuarray.h"

int generalizedCorrelation(
        CuArray<float> *X,
        CuArray<float> *R,
        CuArray<int> *ab,
        int k,
        int n,
        int xd,
        int d,
        const std::string &platform
);

float gpuGeneralizedCorrelation(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n,
        int xd,
        int d
);

float cpuGeneralizedCorrelation(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n,
        int xd,
        int d
);


#endif //NETSCI_GENERALIZED_CORRELATION_H
