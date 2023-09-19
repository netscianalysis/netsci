#ifndef NETSCI_GENERALIZED_CORRELATION_H
#define NETSCI_GENERALIZED_CORRELATION_H

#include "cuarray.h"
#include "platform.h"

namespace netcalc {
    /*!
     * \brief Computes the generalized correlation between all pairs of random variables listed in 'ab'.
     *
     * \param X       Mx(d*N) matrix of M d-dimensional random variables with N samples.
     * \param R       Vector that stores the generalized correlation between pairs of random variables listed in 'ab'.
     * \param ab      Vector of pairs of random variables for which generalized correlation is computed.
     * \param k       K value used in generalized correlation calculation.
     * \param n       Number of samples.
     * \param xd      The dimension of the joint random variable. Only 2D-joint random variables are supported.
     * \param d       The dimension of each random variable. Only 1, 2, and 3-dimensional random variables are supported.
     * \param platform Platform (CPU or GPU) used for computation. Use 0 for GPU, and 1 for CPU.
     *
     * \return 0 if successful, 1 otherwise.
     */
    int generalizedCorrelation(
            CuArray<float>* X,
            CuArray<float>* R,
            CuArray<int>* ab,
            int k,
            int n,
            int xd,
            int d,
            int platform,
            int checkpointFrequency,
            std::string checkpointFileName
    );

    int generalizedCorrelation(
            CuArray<float>* X,
            CuArray<float>* R,
            CuArray<int>* ab,
            int k,
            int n,
            int xd,
            int d,
            int platform
    );

    /*!
     * \brief Computes the generalized correlation between two random variables Xa and Xb on the GPU.
     *
     * \param Xa  CuArray representing the first random variable.
     * \param Xb  CuArray representing the second random variable.
     * \param k   K value used in generalized correlation calculation.
     * \param n   Number of samples.
     * \param xd  The dimension of the joint random variable. Only 2D-joint random variables are supported.
     * \param d   The dimension of each random variable. Only 1, 2, and 3-dimensional random variables are supported.
     *
     * \return The computed generalized correlation value.
     */
    float generalizedCorrelationGpu(
            CuArray<float>* Xa,
            CuArray<float>* Xb,
            int k,
            int n,
            int xd,
            int d
    );

    /*!
     * \brief Computes the generalized correlation between two random variables Xa and Xb on the CPU.
     *
     * \param Xa  CuArray representing the first random variable.
     * \param Xb  CuArray representing the second random variable.
     * \param k   K value used in generalized correlation calculation.
     * \param n   Number of samples.
     * \param xd  The dimension of the joint random variable. Only 2D-joint random variables are supported.
     * \param d   The dimension of each random variable. Only 1, 2, and 3-dimensional random variables are supported.
     *
     * \return The computed generalized correlation value.
     */
    float generalizedCorrelationCpu(
            CuArray<float>* Xa,
            CuArray<float>* Xb,
            int k,
            int n,
            int xd,
            int d
    );
}

#endif // NETSCI_GENERALIZED_CORRELATION_H
