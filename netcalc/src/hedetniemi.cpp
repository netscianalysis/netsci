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


void netcalc::pathFromPathsCuArray(
        int **NUMPY_ARRAY,
        int **NUMPY_ARRAY_DIM1,
        CuArray<int> *paths,
        int i,
        int j
) {
    auto longestPath = 5;
    *(NUMPY_ARRAY_DIM1) = new int[1];
    (*NUMPY_ARRAY_DIM1)[0] = 0;
    for (int k = 0; k < longestPath; k++) {
        auto node = paths->get(i,
                               j * longestPath + k);
        if (node != -1) {
            (*NUMPY_ARRAY_DIM1)[0] += 1;
        } else {
            (*NUMPY_ARRAY_DIM1)[0] += 1;
            paths->set(
                    j, i, j*longestPath + k
                    );
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
