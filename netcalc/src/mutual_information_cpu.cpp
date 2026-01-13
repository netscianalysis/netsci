#include <numeric>
#include <algorithm>
#include <memory>

#include "mutual_information.h"
#include "psi.h"

float netcalc::mutualInformationCpu(
    const std::unique_ptr<CuArray<float>>& Xa,
    const std::unique_ptr<CuArray<float>>& Xb,
    int k,
    int n,
    int xd,
    int d
) {
    // psi array
    auto psi = std::make_unique<CuArray<float>>();
    psi->init(1, n + 1);
    generatePsi(psi, n);

    // neighbor counts
    auto nXa = std::make_unique<CuArray<int>>();
    nXa->init(1, n);

    auto nXb = std::make_unique<CuArray<int>>();
    nXb->init(1, n);

    float averageXaXbPsiK = 0.0f;

    for (int i = 0; i < n; i++) {
        nXa->set(0, 0, i);
        nXb->set(0, 0, i);

        // distance array
        auto dZ = std::make_unique<CuArray<float>>();
        dZ->init(1, n);

        for (int j = 0; j < n; j++) {
            float dXa = std::abs(Xa->get(0, i) - Xa->get(0, j));
            for (int di = 1; di < d; di++) {
                dXa = std::max(
                    dXa,
                    std::abs(
                        Xa->get(0, i + di * n) -
                        Xa->get(0, j + di * n)
                    )
                );
            }

            float dXb = std::abs(Xb->get(0, i) - Xb->get(0, j));
            for (int di = 1; di < d; di++) {
                dXb = std::max(
                    dXb,
                    std::abs(
                        Xb->get(0, i + di * n) -
                        Xb->get(0, j + di * n)
                    )
                );
            }

            dZ->set(std::max(dXa, dXb), 0, j);
        }

        std::vector<int> argMin(n);
        std::iota(argMin.begin(), argMin.end(), 0);

        std::sort(argMin.begin(), argMin.end(),
            [&](int a, int b) {
                return (*dZ)[a] < (*dZ)[b];
            });

        float epsXa = 0.0f;
        float epsXb = 0.0f;

        for (int ki = 1; ki <= k; ki++) {
            int j = argMin[ki];

            float dXa = std::abs(Xa->get(0, i) - Xa->get(0, j));
            for (int di = 1; di < d; di++) {
                dXa = std::max(
                    dXa,
                    std::abs(
                        Xa->get(0, i + di * n) -
                        Xa->get(0, j + di * n)
                    )
                );
            }

            float dXb = std::abs(Xb->get(0, i) - Xb->get(0, j));
            for (int di = 1; di < d; di++) {
                dXb = std::max(
                    dXb,
                    std::abs(
                        Xb->get(0, i + di * n) -
                        Xb->get(0, j + di * n)
                    )
                );
            }

            epsXa = std::max(epsXa, dXa);
            epsXb = std::max(epsXb, dXb);
        }

        for (int j = 0; j < n; j++) {
            float dXa = std::abs(Xa->get(0, i) - Xa->get(0, j));
            for (int di = 1; di < d; di++) {
                dXa = std::max(
                    dXa,
                    std::abs(
                        Xa->get(0, i + di * n) -
                        Xa->get(0, j + di * n)
                    )
                );
            }

            float dXb = std::abs(Xb->get(0, i) - Xb->get(0, j));
            for (int di = 1; di < d; di++) {
                dXb = std::max(
                    dXb,
                    std::abs(
                        Xb->get(0, i + di * n) -
                        Xb->get(0, j + di * n)
                    )
                );
            }

            if (dXa <= epsXa) {
                nXa->set(nXa->get(0, i) + 1, 0, i);
            }
            if (dXb <= epsXb) {
                nXb->set(nXb->get(0, i) + 1, 0, i);
            }
        }

        nXa->set(nXa->get(0, i) - 1, 0, i);
        nXb->set(nXb->get(0, i) - 1, 0, i);

        averageXaXbPsiK +=
            (*psi)[(*nXa)[i]] + (*psi)[(*nXb)[i]];
    }

    averageXaXbPsiK /= static_cast<float>(n);

    float mutualInformation =
        psi->get(0, n) +
        psi->get(0, k) -
        (1.0f / static_cast<float>(k)) -
        averageXaXbPsiK;

    return mutualInformation;
}
