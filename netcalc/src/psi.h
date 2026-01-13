//
// Created by andy on 3/24/23.
//

#ifndef MUTUAL_INFORMATION_SHARED_MEMORY_PSI_H
#define MUTUAL_INFORMATION_SHARED_MEMORY_PSI_H

#include "cuarray.h"
#include <vector>

#include <memory>

/*!
 * \brief Generates the diagamma array used in the mutual information calculation.
 *
 * This function generates the diagamma array and stores it in the provided output array 'psi'.
 *
 * \param psi Pointer to the output array where the diagamma array will be stored.
 * \param n   Number of observations in each random variable.
 */
void generatePsi(
        const std::unique_ptr<CuArray<float>> &psi,
        int n);

#endif // MUTUAL_INFORMATION_SHARED_MEMORY_PSI_H
