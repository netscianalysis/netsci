#include <complex>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "mutual_information.h"
#include "psi.h"

float cpuMutualInformation2X1D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
) {
    auto *psi = new CuArray<float>;
    psi->init(1, n + 1);
    generatePsi(psi, n);
    auto *nXa = new CuArray<int>;
    nXa->init(1, n);
    auto *nXb = new CuArray<int>;
    nXb->init(1, n);
    float averageXaXbPsiK = 0.0;
    for (int i = 0; i < n; i++) {
        (*nXa)[i] = 0;
        (*nXb)[i] = 0;
        auto *dZ = new CuArray<float>;
        dZ->init(1, n);
        for (int j = 0; j < n; j++) {
            (*dZ)[j] = std::max(
                    std::abs((*Xa)[i] - (*Xa)[j]),
                    std::abs((*Xb)[i] - (*Xb)[j])
            );
        }
        std::vector<int> argMin(n);
        std::iota(argMin.begin(), argMin.end(), 0);
        std::sort(argMin.begin(), argMin.end(), [&](
                int i,
                int j
        ) { return (*dZ)[i] < (*dZ)[j]; });
        float epsXa = 0.0;
        float epsXb = 0.0;
        for (int ki = 1; ki < k + 1; ki++) {
            epsXa = std::max(
                    epsXa,
                    (float) std::abs((*Xa)[i] - (*Xa)[argMin[ki]])
            );
            epsXb = std::max(
                    epsXb,
                    (float) std::abs((*Xb)[i] - (*Xb)[argMin[ki]])
            );
        }
        for (int j = 0; j < n; j++) {
            if (std::abs((*Xa)[i] - (*Xa)[j]) <= epsXa) {
                (*nXa)[i]++;
            }
            if (std::abs((*Xb)[i] - (*Xb)[j]) <= epsXb) {
                (*nXb)[i]++;
            }
        }
        (*nXa)[i] -= 1;
        (*nXb)[i] -= 1;
        averageXaXbPsiK += ((*psi)[(*nXa)[i]] + (*psi)[(*nXb)[i]]);
        delete dZ;
    }
    averageXaXbPsiK /= (float) n;
    delete nXa;
    delete nXb;
    float mutualInformation = (*psi)[n] + (*psi)[k] -
            (float)(1.0 / (float) k) - averageXaXbPsiK;
    delete psi;
    return mutualInformation;
}

float cpuMutualInformation2X2D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
) {
    auto *psi = new CuArray<float>;
    psi->init(1, n + 1);
    generatePsi(psi, n);
    auto *nXa = new CuArray<int>;
    nXa->init(1, n);
    auto *nXb = new CuArray<int>;
    nXb->init(1, n);
    float averageXaXbPsiK = 0.0;
    for (int i = 0; i < n; i++) {
        (*nXa)[i] = 0;
        (*nXb)[i] = 0;
        auto *dZ = new CuArray<float>;
        dZ->init(1, n);
        for (int j = 0; j < n; j++) {
            float dXa = std::abs((*Xa)[i] - (*Xa)[j]);
            dXa = std::max(
                    dXa,
                    std::abs((*Xa)[i + n] - (*Xa)[j + n])
            );
            float dXb = std::abs((*Xb)[i] - (*Xb)[j]);
            dXb = std::max(
                    dXb,
                    std::abs((*Xb)[i + n] - (*Xb)[j + n])
            );
            (*dZ)[j] = std::max(dXa, dXb);
        }
        std::vector<int> argMin(n);
        std::iota(argMin.begin(), argMin.end(), 0);
        std::sort(argMin.begin(), argMin.end(), [&](
                int i,
                int j
        ) { return (*dZ)[i] < (*dZ)[j]; });
        float epsXa = 0.0;
        float epsXb = 0.0;
        for (int ki = 1; ki < k + 1; ki++) {
            float dXa = std::abs((*Xa)[i] - (*Xa)[argMin[ki]]);
            dXa = std::max(
                    dXa,
                    std::abs((*Xa)[i + n] - (*Xa)[argMin[ki] + n])
            );
            float dXb = std::abs((*Xb)[i] - (*Xb)[argMin[ki]]);
            dXb = std::max(
                    dXb,
                    std::abs((*Xb)[i + n] - (*Xb)[argMin[ki] + n])
            );
            epsXa = std::max(epsXa, dXa);
            epsXb = std::max(epsXb, dXb);
        }
        for (int j = 0; j < n; j++) {
            float dXa = std::abs((*Xa)[i] - (*Xa)[j]);
            dXa = std::max(
                    dXa,
                    std::abs((*Xa)[i + n] - (*Xa)[j + n])
            );
            float dXb = std::abs((*Xb)[i] - (*Xb)[j]);
            dXb = std::max(
                    dXb,
                    std::abs((*Xb)[i + n] - (*Xb)[j + n])
            );
            if (dXa <= epsXa) {
                (*nXa)[i]++;
            }
            if (dXb <= epsXb) {
                (*nXb)[i]++;
            }
        }
        (*nXa)[i] -= 1;
        (*nXb)[i] -= 1;
        averageXaXbPsiK += ((*psi)[(*nXa)[i]] + (*psi)[(*nXb)[i]]);
        delete dZ;
    }
    averageXaXbPsiK /= (float) n;
    delete nXa;
    delete nXb;
    float mutualInformation = (*psi)[n] + (*psi)[k] -
            (float)(1.0 / (float) k) - averageXaXbPsiK;
    delete psi;
    return mutualInformation;
}

float cpuGeneralizedCorrelation2X1D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
) {
    float mutualInformation = cpuMutualInformation2X1D(Xa, Xb, k, n);
    return (float) std::sqrt(
            1.0 - (float) std::exp(-2.0 * mutualInformation)
    );
}

float cpuGeneralizedCorrelation2X2D(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n
) {
    float mutualInformation = cpuMutualInformation2X2D(Xa, Xb, k, n);
    return (float) std::sqrt(
            1.0 - std::exp(-1.0 * mutualInformation)
    );
}

