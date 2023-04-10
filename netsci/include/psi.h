//
// Created by andy on 3/24/23.
//

#ifndef MUTUAL_INFORMATION_SHARED_MEMORY_PSI_H
#define MUTUAL_INFORMATION_SHARED_MEMORY_PSI_H

#include <vector>
#include "cuarray.h"

void generatePsi(
        CuArray<float> *psi,
        int n
);

#endif //MUTUAL_INFORMATION_SHARED_MEMORY_PSI_H
