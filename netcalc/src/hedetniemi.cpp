//
// Created by astokely on 5/10/23.
//
#include <iostream>
#include <vector>
#include "hedetniemi.h"
#include <limits>

int hedetniemiShortestPaths(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        int platform
) {
    int maxPathLength = 0;
    H->init(A->m(),
            A->n());
    if (platform == 0) {
        maxPathLength = hedetniemiShortestPathsGpu(A,
                                                   H,
                                                   paths
        );

    }
    return maxPathLength;
}

void correlationToAdjacency(
        CuArray<float> *A,
        CuArray<float> *C,
        int n,
        int platform
) {
    A->init(n,
            n);
    if (platform == 0) {
        correlationToAdjacencyGpu(A,
                                  C,
                                  n);
    }

}
