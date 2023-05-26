//
// Created by astokely on 5/16/23.
//

#ifndef NETSCI_MATH_UTILS_H
#define NETSCI_MATH_UTILS_H

#include "cuarray.h"

/**
 * \brief Computes the mean of the rows in a matrix.
 *
 * This function calculates the mean of each row in the input matrix 'a'
 * and stores the result in the output array 'u'.
 *
 * \param a         Pointer to the input matrix of type CuArray<float>.
 * \param u         Pointer to the output array where the mean will be stored, of type CuArray<float>.
 * \param m         Number of rows in the matrix.
 * \param n         Number of columns in the matrix.
 * \param platform  Platform used for computation. Use 0 for GPU and 1 for CPU.
 */
void mean(
        CuArray<float>* a,
        CuArray<float>* u,
        int m,
        int n,
        int platform
);

/**
 * \brief Computes the mean of the rows in a matrix on the GPU.
 *
 * This function calculates the mean of each row in the input matrix 'a'
 * on the GPU and stores the result in the output array 'u'.
 *
 * \param a         Pointer to the input matrix of type CuArray<float>.
 * \param u         Pointer to the output array where the mean will be stored, of type CuArray<float>.
 * \param m         Number of rows in the matrix.
 * \param n         Number of columns in the matrix.
 */
void meanGpu(
        CuArray<float>* a,
        CuArray<float>* u,
        int m,
        int n
);

/**
 * \brief Computes the standard deviation of each row in a matrix.
 *
 * This function calculates the standard deviation of the elements in each row
 * of the input matrix 'a' and stores the result in the output array 'sigma'.
 * The mean of each row is also calculated and stored in the array 'u'.
 *
 * \param a         Pointer to the input matrix of type CuArray<float>.
 * \param u         Pointer to the array where the mean will be stored, of type CuArray<float>.
 * \param sigma     Pointer to the output array where the standard deviation will be stored, of type CuArray<float>.
 * \param m         Number of rows in the matrix.
 * \param n         Number of columns in the matrix.
 * \param platform  Platform used for computation. Use 0 for GPU and 1 for CPU.
 */
void standardDeviation(
        CuArray<float>* a,
        CuArray<float>* u,
        CuArray<float>* sigma,
        int m,
        int n,
        int platform
);

/**
 * \brief Computes the standard deviation of each row in a matrix on the GPU.
 *
 * This function calculates the standard deviation of the elements in each row
 * of the input matrix 'a' on the GPU and stores the result in the output array 'sigma'.
 * The mean of each row is also calculated and stored in the array 'u'.
 *
 * \param a         Pointer to the input matrix of type CuArray<float>.
 * \param u         Pointer to the array where the mean will be stored, of type CuArray<float>.
 * \param sigma     Pointer to the output array where the standard deviation will be stored, of type CuArray<float>.
 * \param m         Number of rows in the matrix.
 * \param n         Number of columns in the matrix.
 */
void standardDeviationGpu(
        CuArray<float>* a,
        CuArray<float>* u,
        CuArray<float>* sigma,
        int m,
        int n
);

#endif // NETSCI_MATH_UTILS_H