//
// Created by andy on 3/17/23.
//

#ifndef MUTUAL_INFORMATION_SHARED_MEMORY_MUTUAL_INFORMATION_H
#define MUTUAL_INFORMATION_SHARED_MEMORY_MUTUAL_INFORMATION_H

#include "cuarray.h"

float gpuMutualInformation2X1D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
);

float gpuMutualInformation2X2D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
);

float gpuGeneralizedCorrelation2X1D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
);

float gpuGeneralizedCorrelation2X2D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
);

float cpuMutualInformation2X1D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
);

float cpuMutualInformation2X2D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
);

float cpuGeneralizedCorrelation2X1D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
);

float cpuGeneralizedCorrelation2X2D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
);



#endif //MUTUAL_INFORMATION_SHARED_MEMORY_MUTUAL_INFORMATION_H
