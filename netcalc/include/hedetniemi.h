//
// Created by astokely on 5/10/23.
//

#ifndef NETCALC_HEDETNIEMI_H
#define NETCALC_HEDETNIEMI_H

#include "cuarray.h"

void hedetniemiShortestPaths(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        int maxPathLength,
        int platform
);

void hedetniemiShortestPathLengths(
        CuArray<float> *A,
        CuArray<float> *H,
        int maxPathLength,
        int platform
);

void hedetniemiShortestPathsGpu(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        int maxPathLength
);

void hedetniemiShortestPathLengthsGpu(
        CuArray<float> *A,
        CuArray<float> *H,
        int maxPathLength
);

void correlationToAdjacency(
        CuArray<float> *A,
        CuArray<float> *C,
        int n,
        int platform
);


void correlationToAdjacencyGpu(
        CuArray<float> *A,
        CuArray<float> *C,
        int n
);


#endif //NETCALC_HEDETNIEMI_H
