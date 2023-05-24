//
// Created by andy on 4/17/23.
//

#ifndef NETSCI_GENERALIZED_CORRELATION_H
#define NETSCI_GENERALIZED_CORRELATION_H
#include "cuarray.h"

namespace netcalc {
    int generalizedCorrelation(
            CuArray<float> *X,
            CuArray<float> *R,
            CuArray<int> *ab,
            int k,
            int n,
            int xd,
            int d,
            int platform
    );

    float generalizedCorrelationGpu(
            CuArray<float> *Xa,
            CuArray<float> *Xb,
            int k,
            int n,
            int xd,
            int d
    );

    float generalizedCorrelationCpu(
            CuArray<float> *Xa,
            CuArray<float> *Xb,
            int k,
            int n,
            int xd,
            int d
    );
}


#endif //NETSCI_GENERALIZED_CORRELATION_H
