#include <numeric>
#include <algorithm>
#include "mutual_information.h"
#include "psi.h"

float cpuMutualInformation(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n,
        int xd,
        int d
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
        nXa->set(0, 0, i);
        nXb->set(0, 0, i);
        auto *dZ = new CuArray<float>;
        dZ->init(1, n);
        for (int j = 0; j < n; j++) {
            float dXa = std::abs(Xa->get(0, i)
                     - Xa->get(0, j));
            for (int di = 1; di < d; di++) {
                dXa = std::max(
                        dXa,
                        std::abs(Xa->get(0, i + (di * n))
                                 - Xa->get(0, j + (di * n)))
                );
            }
            float dXb = std::abs(Xb->get(0, i)
                     - Xb->get(0, j));
            for (int di = 1; di < d; di++) {
                dXb = std::max(
                        dXb,
                        std::abs(Xb->get(0, i + (di * n))
                                 - Xb->get(0, j + (di * n)))
                );
            }
            dZ->set(std::max(dXa, dXb), 0, j);
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
            float dXa = std::abs(Xa->get(0, i)
                     - Xa->get(0, argMin[ki]));
            for (int di = 1; di < d; di++) {
                dXa = std::max(
                        dXa,
                        std::abs(Xa->get(0, i + (di * n))
                                 - Xa->get(0, argMin[ki] + (di * n)))
                );
            }
            float dXb = std::abs(Xb->get(0, i)
                     - Xb->get(0, argMin[ki]));
            for (int di = 1; di < d; di++) {
                dXb = std::max(
                        dXb,
                        std::abs(Xb->get(0, i + (di * n))
                                 - Xb->get(0, argMin[ki] + (di * n)))
                );
            }
            epsXa = std::max(epsXa, dXa);
            epsXb = std::max(epsXb, dXb);
        }
        for (int j = 0; j < n; j++) {
            float dXa = std::abs(Xa->get(0, i)
                     - Xa->get(0, j));
            for (int di = 1; di < d; di++) {
                dXa = std::max(
                        dXa,
                        std::abs(Xa->get(0, i + (di * n))
                                 - Xa->get(0, j + (di * n)))
                );
            }
            float dXb = std::abs(Xb->get(0, i)
                     - Xb->get(0, j));
            for (int di = 1; di < d; di++) {
                dXb = std::max(
                        dXb,
                        std::abs(Xb->get(0, i + (di * n))
                                 - Xb->get(0, j + (di * n)))
                );
            }
            if (dXa <= epsXa) {
                nXa->set(
                        nXa->get(0, i) + 1,
                        0, i);
            }
            if (dXb <= epsXb) {
                nXb->set(
                        nXb->get(0, i) + 1,
                        0, i);
            }
        }
        nXa->set(nXa->get(0, i) - 1, 0, i);
        nXb->set(nXb->get(0, i) - 1, 0, i);
        averageXaXbPsiK += ((*psi)[(*nXa)[i]] + (*psi)[(*nXb)[i]]);
        delete dZ;
    }
    averageXaXbPsiK /= (float) n;
    delete nXa;
    delete nXb;
    float mutualInformation = psi->get(0, n) + psi->get(0, k) -
                              (float) (1.0 / (float) k) -
                              averageXaXbPsiK;
    delete psi;
    return mutualInformation;
}

