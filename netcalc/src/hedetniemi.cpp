//
// Created by astokely on 5/10/23.
//
#include <stdexcept>
#include <algorithm>
#include <iostream>
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

template<typename ForwardIterator>
ForwardIterator remove_duplicates(
        ForwardIterator first,
        ForwardIterator last
) {
    auto new_last = first;

    for (auto current = first; current != last; ++current) {
        if (std::find(first,
                      new_last,
                      *current) == new_last) {
            if (new_last != current) *new_last = *current;
            ++new_last;
        }
    }

    return new_last;
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
    std::vector<int> tmp;
    for (int k = 0; k < maxPathLength; k++) {
        auto node = paths->get(i*maxPathLength + k,
                               j );
        if (node != -1) {
            tmp.push_back(node);
        }
    }
    //Reverse tmp
    std::reverse(tmp.begin(),
                 tmp.end());
    tmp.push_back(j);
    (*NUMPY_ARRAY_DIM1)[0] = (int)tmp.size();
    *NUMPY_ARRAY = new int[(*NUMPY_ARRAY_DIM1)[0]];
    std::copy(
            tmp.begin(),
            tmp.end(),
            *NUMPY_ARRAY
    );
}
