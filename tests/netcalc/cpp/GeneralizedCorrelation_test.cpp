#include <ut.hpp>
#include <cmath>
#include <vector>

#include "cuarray.h"
#include "psi.h"
#include "generalized_correlation.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

using namespace boost::ut;

int main() {

    test("GeneralizedCorrelation_2X1D_1000n4k_GpuCpu") = [] {
        int n = 1000;
        int k = 4;

        auto* Xa = new CuArray<float>;
        auto* Xb = new CuArray<float>;
        Xa->init(1, n);
        Xb->init(1, n);

        float incr = M_PI / static_cast<float>(n);
        std::vector<float> domain(n);
        domain[0] = 0.001f;
        for (int i = 1; i < n; i++) {
            domain[i] = domain[i - 1] + incr;
        }

        for (int i = 0; i < n; i++) {
            (*Xa)[i] = std::sin(domain[i]);
            (*Xb)[i] = std::cos(domain[i]);
        }

        float cpu =
            netcalc::generalizedCorrelationCpu(Xa, Xb, k, n, 2, 1);
        float gpu =
            netcalc::generalizedCorrelationGpu(Xa, Xb, k, n, 2, 1);

        expect(approx(cpu, gpu, 1e-6f));

        delete Xa;
        delete Xb;
    };

    test("GeneralizedCorrelation_2X1D_2000n4k_GpuCpu") = [] {
        int n = 2000;
        int k = 4;

        auto* Xa = new CuArray<float>;
        auto* Xb = new CuArray<float>;
        Xa->init(1, n);
        Xb->init(1, n);

        float incr = M_PI / static_cast<float>(n);
        std::vector<float> domain(n);
        domain[0] = 0.001f;
        for (int i = 1; i < n; i++) {
            domain[i] = domain[i - 1] + incr;
        }

        for (int i = 0; i < n; i++) {
            (*Xa)[i] = std::sin(domain[i]);
            (*Xb)[i] = std::cos(domain[i]);
        }

        float cpu =
            netcalc::generalizedCorrelationCpu(Xa, Xb, k, n, 2, 1);
        float gpu =
            netcalc::generalizedCorrelationGpu(Xa, Xb, k, n, 2, 1);

        expect(approx(cpu, gpu, 1e-6f));

        delete Xa;
        delete Xb;
    };

    test("GeneralizedCorrelation_2X2D_1000n4k_GpuCpu") = [] {
        int n = 1000;
        int k = 4;

        auto* Xa = new CuArray<float>;
        auto* Xb = new CuArray<float>;
        Xa->init(1, 2 * n);
        Xb->init(1, 2 * n);

        float incr = M_PI / static_cast<float>(n);
        std::vector<float> domain(n);
        domain[0] = 0.001f;
        for (int i = 1; i < n; i++) {
            domain[i] = domain[i - 1] + incr;
        }

        for (int i = 0; i < n; i++) {
            (*Xa)[i]       = std::sin(domain[i]);
            (*Xa)[i + n]   = std::cos(domain[i]);
            (*Xb)[i]       = std::cos(domain[i]);
            (*Xb)[i + n]   = 2.0f * domain[i];
        }

        float cpu =
            netcalc::generalizedCorrelationCpu(Xa, Xb, k, n, 2, 2);
        float gpu =
            netcalc::generalizedCorrelationGpu(Xa, Xb, k, n, 2, 2);

        expect(approx(cpu, gpu, 1e-6f));

        delete Xa;
        delete Xb;
    };

    test("GeneralizedCorrelation_2X2D_2000n4k_GpuCpu") = [] {
        int n = 2000;
        int k = 4;

        auto* Xa = new CuArray<float>;
        auto* Xb = new CuArray<float>;
        Xa->init(1, 2 * n);
        Xb->init(1, 2 * n);

        float incr = M_PI / static_cast<float>(n);
        std::vector<float> domain(n);
        domain[0] = 0.001f;
        for (int i = 1; i < n; i++) {
            domain[i] = domain[i - 1] + incr;
        }

        for (int i = 0; i < n; i++) {
            (*Xa)[i]       = std::sin(domain[i]);
            (*Xa)[i + n]   = std::cos(domain[i]);
            (*Xb)[i]       = domain[i];
            (*Xb)[i + n]   = 2.0f * domain[i];
        }

        float cpu =
            netcalc::generalizedCorrelationCpu(Xa, Xb, k, n, 2, 2);
        float gpu =
            netcalc::generalizedCorrelationGpu(Xa, Xb, k, n, 2, 2);

        expect(approx(cpu, gpu, 1e-6f));

        delete Xa;
        delete Xb;
    };

    test("GeneralizedCorrelation_2X3D_1000n4k_GpuCpu") = [] {
        int n = 1000;
        int k = 4;

        auto* Xa = new CuArray<float>;
        auto* Xb = new CuArray<float>;
        Xa->init(1, 3 * n);
        Xb->init(1, 3 * n);

        float incr = M_PI / static_cast<float>(n);
        std::vector<float> domain(n);
        domain[0] = 0.001f;
        for (int i = 1; i < n; i++) {
            domain[i] = domain[i - 1] + incr;
        }

        for (int i = 0; i < n; i++) {
            (*Xa)[i]           = std::sin(domain[i]);
            (*Xa)[i + n]       = std::cos(domain[i]);
            (*Xa)[i + 2 * n]   = std::log(domain[i]);
            (*Xb)[i]           = std::cos(domain[i]);
            (*Xb)[i + n]       = 2.0f * domain[i];
            (*Xb)[i + 2 * n]   = static_cast<float>(std::log(domain[i] * 2.0));
        }

        float cpu =
            netcalc::generalizedCorrelationCpu(Xa, Xb, k, n, 2, 3);
        float gpu =
            netcalc::generalizedCorrelationGpu(Xa, Xb, k, n, 2, 3);

        expect(approx(cpu, gpu, 1e-6f));

        delete Xa;
        delete Xb;
    };

    test("GeneralizedCorrelation_2X3D_2000n4k_GpuCpu") = [] {
        int n = 2000;
        int k = 4;

        auto* Xa = new CuArray<float>;
        auto* Xb = new CuArray<float>;
        Xa->init(1, 3 * n);
        Xb->init(1, 3 * n);

        float incr = M_PI / static_cast<float>(n);
        std::vector<float> domain(n);
        domain[0] = 0.001f;
        for (int i = 1; i < n; i++) {
            domain[i] = domain[i - 1] + incr;
        }

        for (int i = 0; i < n; i++) {
            (*Xa)[i]           = std::sin(domain[i]);
            (*Xa)[i + n]       = std::cos(domain[i]);
            (*Xa)[i + 2 * n]   = std::log(domain[i]);
            (*Xb)[i]           = std::cos(domain[i]);
            (*Xb)[i + n]       = 2.0f * domain[i];
            (*Xb)[i + 2 * n]   = static_cast<float>(std::log(domain[i] * 2.0));
        }

        float cpu =
            netcalc::generalizedCorrelationCpu(Xa, Xb, k, n, 2, 3);
        float gpu =
            netcalc::generalizedCorrelationGpu(Xa, Xb, k, n, 2, 3);

        expect(approx(cpu, gpu, 1e-6f));

        delete Xa;
        delete Xb;
    };

    test("GeneralizedCorrelation_UsedCpuPlatform") = [] {
        int n = 1000;
        int k = 4;

        auto* X  = new CuArray<float>;
        auto* R  = new CuArray<float>;
        auto* ab = new CuArray<int>;

        X->init(2, n);

        float incr = M_PI / static_cast<float>(n);
        std::vector<float> domain(n);
        domain[0] = 0.001f;
        for (int i = 1; i < n; i++) {
            domain[i] = domain[i - 1] + incr;
        }

        for (int i = 0; i < n; i++) {
            X->set(std::sin(domain[i]), 0, i);
            X->set(std::cos(domain[i]), 1, i);
        }

        ab->init(1, 2);
        ab->set(0, 0, 0);
        ab->set(1, 0, 1);

        expect(
            netcalc::generalizedCorrelation(
                X, R, ab, k, n, 2, 1, netcalc::CPU_PLATFORM
            ) == 1
        );

        delete X;
        delete R;
        delete ab;
    };

    test("GeneralizedCorrelation_UsedGpuPlatform") = [] {
        int n = 1000;
        int k = 4;

        auto* X  = new CuArray<float>;
        auto* R  = new CuArray<float>;
        auto* ab = new CuArray<int>;

        X->init(2, n);

        float incr = M_PI / static_cast<float>(n);
        std::vector<float> domain(n);
        domain[0] = 0.001f;
        for (int i = 1; i < n; i++) {
            domain[i] = domain[i - 1] + incr;
        }

        for (int i = 0; i < n; i++) {
            X->set(std::sin(domain[i]), 0, i);
            X->set(std::cos(domain[i]), 1, i);
        }

        ab->init(1, 2);
        ab->set(0, 0, 0);
        ab->set(1, 0, 1);

        expect(
            netcalc::generalizedCorrelation(
                X, R, ab, k, n, 2, 1, netcalc::GPU_PLATFORM
            ) == 0
        );

        delete X;
        delete R;
        delete ab;
    };

    return 0;
}
