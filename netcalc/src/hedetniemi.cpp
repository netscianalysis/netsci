//
// Created by astokely on 5/10/23.
//
#include <iostream>
#include <vector>
#include "hedetniemi.h"
#include <limits>

void hedetniemiShortestPaths(
        CuArray<float> *A,
        CuArray<float> *H, CuArray<int> *paths,
        int maxPathLength, int platform
) {
    paths->init(A->m(), A->n() * maxPathLength);
    for (int i = 0; i < A->m(); i++) {
        for (int j = 0; j < maxPathLength * A->n(); j++) {
            paths->set(-1, i, j);
        }
    }
    H->init(A->m(), A->n());
    if (platform == 0) {
        hedetniemiShortestPathsGpu(A, H, paths, maxPathLength);
    }
}

void hedetniemiShortestPathLengths(
        CuArray<float> *A,
        CuArray<float> *H,
        int maxPathLength, int platform
) {
    H->init(A->m(), A->n());
    if (platform == 0) {
        hedetniemiShortestPathLengthsGpu(A, H, maxPathLength);
    }
}

void correlationToAdjacency(
        CuArray<float> *A,
        CuArray<float> *C,
        int n,
        int platform
) {
    A->init(n, n);
    if (platform == 0) {
        correlationToAdjacencyGpu(A, C, n);
    }

}
