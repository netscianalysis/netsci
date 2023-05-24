//
// Created by astokely on 5/10/23.
//

#ifndef NETCALC_HEDETNIEMI_H
#define NETCALC_HEDETNIEMI_H

#include "cuarray.h"

namespace netcalc {
    void hedetniemiShortestPaths(
            CuArray<float> *A,
            CuArray<float> *H,
            CuArray<int> *paths,
            float tolerance,
            int platform
    );

    void hedetniemiShortestPathsGpu(
            CuArray<float> *A,
            CuArray<float> *H,
            CuArray<int> *paths,
            float tolerance
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

    int longestShortestPathNodeCount(
            CuArray<int> *paths
    );
}


#endif //NETCALC_HEDETNIEMI_H
