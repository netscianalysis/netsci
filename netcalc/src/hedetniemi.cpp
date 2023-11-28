//
// Created by astokely on 5/10/23.
//
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "hedetniemi.h"

void netcalc::hedetniemiAllShortestPaths(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        int maxPathLength,
        int platform
) {
    H->init(A->m(),
            A->n());
    if (platform == 0) {
        netcalc::hedetniemiAllShortestPathsGpu(A,
                                               H,
                                               paths,
                                               maxPathLength
        );
    } else if (platform == 1) {
        netcalc::hedetniemiAllShortestPathsCpu(A,
                                               H,
                                               paths,
                                               maxPathLength
        );
    } else {
        throw std::invalid_argument("Invalid platform");
    }
}

void netcalc::hedetniemiAllShortestPathLengths(
        CuArray<float> *A,
        CuArray<float> *H,
        int maxPathLength,
        int platform
) {
    H->init(A->m(),
            A->n());
    if (platform == 0) {
        netcalc::hedetniemiAllShortestPathLengthsGpu(A,
                                                     H,
                                                     maxPathLength
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

void netcalc::recoverSingleShortestPath(
        int **NUMPY_ARRAY,
        int **NUMPY_ARRAY_DIM1,
        CuArray<int> *paths,
        int maxPathLength,
        int i,
        int j
) {
    *(NUMPY_ARRAY_DIM1) = new int[1];
    (*NUMPY_ARRAY_DIM1)[0] = 0;
    std::vector<int> path;
    path.push_back(i);
    int n = (int) std::sqrt(paths->n() / maxPathLength);
    for (int k = 0; k < maxPathLength; k++) {
        auto node = paths->get(0,
                               i * maxPathLength * n +
                               j * maxPathLength + k);
        if (std::find(
                path.begin(),
                path.end(),
                node) == path.end()) {
            path.push_back(node);
        }
    }
    if (std::find(
            path.begin(),
            path.end(),
            j) == path.end()) {
        path.push_back(j);
    }
    (*NUMPY_ARRAY_DIM1)[0] = (int) path.size();
    *NUMPY_ARRAY = new int[(*NUMPY_ARRAY_DIM1)[0]];
    std::copy(
            path.begin(),
            path.end(),
            *NUMPY_ARRAY
    );
}
