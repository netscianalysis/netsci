//
// Created by andy on 3/17/23.
//

#ifndef MUTUAL_INFORMATION_SHARED_MEMORY_MUTUAL_INFORMATION_H
#define MUTUAL_INFORMATION_SHARED_MEMORY_MUTUAL_INFORMATION_H

#include "cuarray.h"

namespace netcalc {
    int mutualInformation(
            CuArray<float> *X,
            CuArray<float> *I,
            CuArray<int> *ab,
            int k,
            int n,
            int xd,
            int d,
            int platform
    );

    float mutualInformationGpu(
            CuArray<float> *Xa,
            CuArray<float> *Xb,
            int k,
            int n,
            int xd,
            int d
    );


    float mutualInformationCpu(
            CuArray<float> *Xa,
            CuArray<float> *Xb,
            int k,
            int n,
            int xd,
            int d
    );
}

#endif //MUTUAL_INFORMATION_SHARED_MEMORY_MUTUAL_INFORMATION_H
