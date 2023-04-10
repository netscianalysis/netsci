//
// Created by andy on 3/24/23.
//
#include "psi.h"

void generatePsi(
        CuArray<float> *psi,
        int n
) {
    (*psi)[1] = -0.57721566490153;
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            auto inversePsiIndex = (float)(1.0 / static_cast<float>(i));
            (*psi)[i + 1] = (*psi)[i] + inversePsiIndex;
        }
    }
}