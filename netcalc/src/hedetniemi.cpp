//
// Created by astokely on 5/10/23.
//
#include "hedetniemi.h"

void netcalc::hedetniemiShortestPaths(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        float tolerance,
        int platform
) {
    H->init(A->m(),
            A->n());
    if (platform == 0) {
        netcalc::hedetniemiShortestPathsGpu(A,
                                            H,
                                            paths,
                                            tolerance
        );

    }
}

void netcalc::correlationToAdjacency(
        CuArray<float> *A,
        CuArray<float> *C,
        int n,
        int platform
) {
    A->init(n,
            n);
    if (platform == 0) {
        netcalc::correlationToAdjacencyGpu(A,
                                           C,
                                           n);
    }
}

int netcalc::longestShortestPathNodeCount(CuArray<int> *paths) {
    int numNodes = paths->m();
    return paths->n() / numNodes;
}
