//
// Created by astokely on 5/10/23.
//
#include <stdexcept>
#include "hedetniemi.h"

void netcalc::hedetniemiShortestPaths(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        float tolerance,
        const std::string &platform
) {
    H->init(A->m(),
            A->n());
    void (*hedetniemiShortestPathsFunction)(
            CuArray<float> *,
            CuArray<float> *,
            CuArray<int> *,
            float
    );
    if (platform == "gpu") {
        hedetniemiShortestPathsFunction = netcalc::hedetniemiShortestPathsGpu;
    } else {
        throw std::runtime_error("Invalid platform");
    }
    hedetniemiShortestPathsFunction(A,
                                    H,
                                    paths,
                                    tolerance
    );
}

void netcalc::correlationToAdjacency(
        CuArray<float> *A,
        CuArray<float> *C,
        int n,
        const std::string &platform
) {
    A->init(n,
            n);
    void (*correlationToAdjacencyFunction)(
            CuArray<float> *,
            CuArray<float> *,
            int
    );
    if (platform == "gpu") {
        correlationToAdjacencyFunction = netcalc::correlationToAdjacencyGpu;
    } else {
        throw std::runtime_error("Invalid platform");
    }
    correlationToAdjacencyFunction(A,
                                   C,
                                   n
    );
}

int netcalc::longestShortestPathNodeCount(CuArray<int> *paths) {
    int numNodes = paths->m();
    return paths->n() / numNodes;
}

void netcalc::pathFromPathsCuArray(
        int **NUMPY_ARRAY,
        int **NUMPY_ARRAY_DIM1,
        CuArray<int> *paths,
        int i,
        int j
) {
    auto longestPath = netcalc::longestShortestPathNodeCount(paths);
    *(NUMPY_ARRAY_DIM1) = new int[1];
    (*NUMPY_ARRAY_DIM1)[0] = 0;
    for (int k = 0; k < longestPath; k++) {
        auto node = paths->get(i,
                               j * longestPath + k);
        if (node != -1) {
            (*NUMPY_ARRAY_DIM1)[0] += 1;
        } else {
            break;
        }
    }
    *NUMPY_ARRAY = new int[(*NUMPY_ARRAY_DIM1)[0]];
    std::copy(
            paths->host() + i * paths->n()
            + j * longestPath,
            paths->host() + i * paths->n()
            + j * longestPath + (*NUMPY_ARRAY_DIM1)[0],
            *NUMPY_ARRAY
    );
}
