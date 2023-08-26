//
// Created by astokely on 5/10/23.
//

#ifndef NETCALC_HEDETNIEMI_H
#define NETCALC_HEDETNIEMI_H

#include <vector>
#include "cuarray.h"
#include "platform.h"

namespace netcalc {
    /*!
     * \brief Compute the shortest paths and shortest path lengths between all pairs of nodes in a graph.
     *
     * \param A         Adjacency matrix of the graph.
     * \param H         NxN matrix of shortest path lengths, where N is the number of nodes in the graph.
     *                  Each matrix element represents the shortest path length between the nodes corresponding
     *                  to the row and column indices of the element.
     * \param paths     Nx(N*k) matrix of shortest paths, where N is the number of nodes in the graph and k is
     *                  the number of nodes in the longest shortest path. Each row of the matrix contains the shortest
     *                  path between the node corresponding to the row index and all other nodes. Each path has k elements,
     *                  where the first element is the node corresponding to the row index and the last element is the node
     *                  corresponding to the column. Unless the path is the longest shortest path, the remaining elements are -1.
     * \param tolerance A very small floating-point number (usually less than 1e-4) used in one of the algorithm steps to
     *                  determine if a node belongs to the shortest path being calculated. If this value is too large,
     *                  the shortest path returned may be incorrect, and if it is too small, the algorithm may never terminate.
     *                  In all testing, 1e-5 has been a good value.
     * \param maxPathLength The maximum number of nodes in a shortest
     *                      allowed in a shortest path. The maximum
     *                      number of edges allowed in a shortest path
     *                      is equal to maxPathLength - 1.
     * \param platform  Platform used to perform the calculation. Use 0 for GPU and 1 for CPU.
     */
    void hedetniemiShortestPaths(
            CuArray<float>* A,
            CuArray<float>* H,
            CuArray<int>* paths,
            float tolerance,
            int maxPathLength,
            int platform
    );

    /*!
     * \brief Compute the shortest paths and shortest path lengths between all pairs of nodes in a graph on the GPU.
     *
     * \param A         Adjacency matrix of the graph.
     *
     * \param H         NxN matrix of shortest path lengths, where N is the number of nodes in the graph.
     *                  Each matrix element represents the shortest path length between the nodes corresponding
     *                  to the row and column indices of the element.
     *
     * \param paths     Nx(N*k) matrix of shortest paths, where N is the number of nodes in the graph and k is
     *                  the number of nodes in the longest shortest path. Each row of the matrix contains the shortest
     *                  path between the node corresponding to the row index and all other nodes. Each path has k elements,
     *                  where the first element is the node corresponding to the row index and the last element is the node
     *                  corresponding to the column. Unless the path is the longest shortest path, the remaining elements are -1.
     *
     * \param maxPathLength The maximum number of nodes in a shortest
     *                      allowed in a shortest path. The maximum
     *                      number of edges allowed in a shortest path
     *                      is equal to maxPathLength - 1.
     *
     * \param tolerance A very small floating-point number (usually less than 1e-4) used in one of the algorithm steps to
     *                  determine if a node belongs to the shortest path being calculated. If this value is too large,
     *                  the shortest path returned may be incorrect, and if it is too small, the algorithm may never terminate.
     *                  In all testing, 1e-5 has been a good value.
     */
    void hedetniemiShortestPathsGpu(
            CuArray<float>* A,
            CuArray<float>* H,
            CuArray<int>* paths,
            int maxPathLength,
            float tolerance
    );

    /*!
     * \brief Converts a correlation matrix from a generalized correlation calculation to an adjacency matrix.
     *        Each element represents the edge weight between two nodes, where the weight is equal to 1/rij, where rij
     *        is the correlation between node i and node j if rij > 0. If rij <= 0, the edge weight is set to infinity.
     *        All diagonal elements are set to 0.
     *
     * \param A The adjacency matrix.
     * \param C The correlation matrix.
     * \param n The number of nodes in the graph.
     * \param platform Platform used to perform the calculation. Use 0 for GPU and 1 for CPU.
     */
    void correlationToAdjacency(
            CuArray<float>* A,
            CuArray<float>* C,
            int n,
            int platform
    );

    /*!
     * \brief Converts a correlation matrix from a generalized correlation calculation to an adjacency matrix on the GPU.
     *        Each element represents the edge weight between two nodes, where the weight is equal to 1/rij, where rij
     *        is the correlation between node i and node j if rij > 0. If rij <= 0, the edge weight is set to infinity.
     *        All diagonal elements are set to 0.
     *
     * \param A The adjacency matrix.
     * \param C The correlation matrix.
     * \param n The number of nodes in the graph.
     */
    void correlationToAdjacencyGpu(
            CuArray<float>* A,
            CuArray<float>* C,
            int n
    );

    /*!
     * \brief Returns the shortest path between node i and node j.
     *
     * \param paths     Nx(N*k) matrix of shortest paths, where N is the number of nodes in the graph and k is
     *                  the number of nodes in the longest shortest path.
     * \param maxPathLength The maximum number of nodes in a shortest
     *                      path, which is equal to the maximum number
     *                      of edges allowed plus 1.
     * \param i         The index of the first node in the path.
     * \param j         The index of the last node in the path.
     */
    void pathFromPathsCuArray(
            int** NUMPY_ARRAY,
            int** NUMPY_ARRAY_DIM1,
            CuArray<int>* paths,
            int maxPathLength,
            int i,
            int j
    );
}

#endif // NETCALC_HEDETNIEMI_H
