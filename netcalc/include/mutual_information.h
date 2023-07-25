//
// Created by andy on 3/17/23.
//

#ifndef MUTUAL_INFORMATION_SHARED_MEMORY_MUTUAL_INFORMATION_H
#define MUTUAL_INFORMATION_SHARED_MEMORY_MUTUAL_INFORMATION_H

#include "cuarray.h"

namespace netcalc {
    /*!
     * \brief Computes the mutual information between all pairs of random variables listed in 'ab'.
     *
     * \param X       Mx(d*N) matrix of M d-dimensional random variables with N samples.
     * \param I       Vector that stores the mutual information between pairs of random variables listed in 'ab'.
     * \param ab      Vector of pairs of random variables for which mutual information is computed.
     * \param k       K value used in mutual information calculation.
     * \param n       Number of samples.
     * \param xd      The dimension of the joint random variable. Only 2D-joint random variables are supported.
     * \param d       The dimension of each random variable. Only 1, 2, and 3-dimensional random variables are supported.
     * \param platform Platform (CPU or GPU) used for computation.
     *                 Use "gpu" for GPU, and "cpu" for CPU.
     *
     */
    void mutualInformation(
            CuArray<float>* X,
            CuArray<float>* I,
            CuArray<int>* ab,
            int k,
            int n,
            int xd,
            int d,
            const std::string &platform
    );

    /*!
     * \brief Computes the mutual information between two random variables Xa and Xb on the GPU.
     *
     * \param Xa  CuArray representing the first random variable.
     * \param Xb  CuArray representing the second random variable.
     * \param k   K value used in mutual information calculation.
     * \param n   Number of samples.
     * \param xd  The dimension of the joint random variable. Only 2D-joint random variables are supported.
     * \param d   The dimension of each random variable. Only 1, 2, and 3-dimensional random variables are supported.
     *
     * \return The computed mutual information value.
     */
    float mutualInformationGpu(
            CuArray<float>* Xa,
            CuArray<float>* Xb,
            int k,
            int n,
            int xd,
            int d
    );


    /*!
     * \brief Computes the mutual information between two random variables Xa and Xb on the CPU.
     *
     * \param Xa  CuArray representing the first random variable.
     * \param Xb  CuArray representing the second random variable.
     * \param k   K value used in mutual information calculation.
     * \param n   Number of samples.
     * \param xd  The dimension of the joint random variable. Only 2D-joint random variables are supported.
     * \param d   The dimension of each random variable. Only 1, 2, and 3-dimensional random variables are supported.
     *
     * \return The computed mutual information value.
     */
    float mutualInformationCpu(
            CuArray<float>* Xa,
            CuArray<float>* Xb,
            int k,
            int n,
            int xd,
            int d
    );
}

#endif // MUTUAL_INFORMATION_SHARED_MEMORY_MUTUAL_INFORMATION_H
