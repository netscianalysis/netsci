//
// Created by andy on 4/4/23.
//
#include "gtest/gtest.h"
#include "cuarray.h"
#include "psi.h"
#include <cmath>
#include "generalized_correlation.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

TEST(
        GeneralizedCorrelation,
        GeneralizedCorrelation_2X1D_1000n4k_GpuCpu
) {
    int n = 1000;
    int k = 4;
    auto *Xa = new CuArray<float>;
    Xa->init(
            1, n
    );
    auto *Xb = new CuArray<float>;
    Xb->init(
            1, n
    );
    float incr = M_PI / (float) n;
    std::vector<float> domain(n);
    domain[0] = 0.001;
    for (
            int i = 1;
            i < n;
            i++) {
        domain[i] = domain[i - 1] +
                    incr;
    }
    for (
            int i = 0;
            i < n;
            i++) {
        (*Xa)[i] =
                std::sin(domain[i]);
        (*Xb)[i] =
                std::cos(domain[i]);
    }
    float cpuGeneralizedCorrelation_ = cpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            1
    );
    float gpuGeneralizedCorrelation_ = gpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            1
    );
    EXPECT_FLOAT_EQ(cpuGeneralizedCorrelation_,
                    gpuGeneralizedCorrelation_
    );
    delete
            Xa;
    delete
            Xb;
}

TEST(
        GeneralizedCorrelation,
        GeneralizedCorrelation_2X1D_2000n4k_GpuCpu
) {
    int n = 1000;
    int k = 4;
    auto *Xa = new CuArray<float>;
    Xa->init(
            1, n
    );
    auto *Xb = new CuArray<float>;
    Xb->init(
            1, n
    );
    float incr = M_PI / (float) n;
    std::vector<float> domain(n);
    domain[0] = 0.001;
    for (int i = 1; i < n; i++) {
        domain[i] = domain[i - 1] + incr;
    }
    for (int i = 0; i < n; i++) {
        (*Xa)[i] = std::sin(domain[i]);
        (*Xb)[i] = std::cos(domain[i]);
    }
    float cpuGeneralizedCorrelation_ = cpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            1
    );
    float gpuGeneralizedCorrelation_ = gpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            1
    );
    EXPECT_FLOAT_EQ(cpuGeneralizedCorrelation_,
                    gpuGeneralizedCorrelation_);
    delete Xa;
    delete Xb;
}

TEST(
        GeneralizedCorrelation,
        GeneralizedCorrelation_2X2D_1000n4k_GpuCpu
) {
    int n = 1000;
    int k = 4;
    auto *Xa = new CuArray<float>;
    Xa->init(
            1, 2 * n
    );
    auto *Xb = new CuArray<float>;
    Xb->init(
            1, 2 * n
    );
    float incr = M_PI / (float) n;
    std::vector<float> domain(n);
    domain[0] = 0.001;
    for (
            int i = 1;
            i < n;
            i++) {
        domain[i] = domain[i - 1] +
                    incr;
    }
    for (
            int i = 0;
            i < n;
            i++) {
        (*Xa)[i] =
                std::sin(domain[i]);
        (*Xa)[i + n] = std::cos(domain[i]);
        (*Xb)[i] =
                std::cos(domain[i]);
        (*Xb)[i + n] = 2 * domain[i];
    }
    float cpuGeneralizedCorrelation_ = cpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            2
    );
    float gpuGeneralizedCorrelation_ = gpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            2
    );
    EXPECT_FLOAT_EQ(cpuGeneralizedCorrelation_,
                    gpuGeneralizedCorrelation_
    );
    delete
            Xa;
    delete
            Xb;
}

TEST(
        GeneralizedCorrelation,
        GeneralizedCorrelation_2X2D_2000n4k_GpuCpu
) {
    int n = 2000;
    int k = 4;
    auto *Xa = new CuArray<float>;
    Xa->init(
            1, 2 * n
    );
    auto *Xb = new CuArray<float>;
    Xb->init(
            1, 2 * n
    );
    float incr = M_PI / (float) n;
    std::vector<float> domain(n);
    domain[0] = 0.001;
    for (int i = 1; i < n; i++) {
        domain[i] = domain[i - 1] + incr;
    }
    for (int i = 0; i < n; i++) {
        (*Xa)[i] = std::sin(domain[i]);
        (*Xa)[i + n] = std::cos(domain[i]);
        (*Xb)[i] = domain[i];
        (*Xb)[i + n] = 2 * domain[i];
    }
    float cpuGeneralizedCorrelation_ = cpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            2
    );
    float gpuGeneralizedCorrelation_ = gpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            2
    );
    EXPECT_FLOAT_EQ(cpuGeneralizedCorrelation_,
                    gpuGeneralizedCorrelation_);
    delete Xa;
    delete Xb;
}

TEST(
        GeneralizedCorrelation,
        GeneralizedCorrelation_2X3D_1000n4k_GpuCpu
) {
    int n = 1000;
    int k = 4;
    auto Xa = new CuArray<float>;
    Xa->init(
            1,
            3 * n
    );
    auto Xb = new CuArray<float>;
    Xb->init(
            1,
            3 * n
    );
    float incr = M_PI / (float) n;
    std::vector<float> domain(n);
    domain[0] = 0.001;
    for (int i = 1; i < n; i++) {
        domain[i] = domain[i - 1] + incr;
    }
    for (int i = 0; i < n; i++) {
        (*Xa)[i] = std::sin(domain[i]);
        (*Xa)[i + n] = std::cos(domain[i]);
        (*Xa)[i + 2 * n] = std::log(domain[i]);
        (*Xb)[i] = std::cos(domain[i]);
        (*Xb)[i + n] = 2 * domain[i];
        (*Xb)[i + 2 * n] = (float) std::log(domain[i] * 2.0);
    }
    float cpuGeneralizedCorrelation_ = cpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            3
    );
    float gpuGeneralizedCorrelation_ = gpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            3
    );
    EXPECT_FLOAT_EQ(
            cpuGeneralizedCorrelation_,
            gpuGeneralizedCorrelation_
    );
    delete Xa;
    delete Xb;
}

TEST(
        GeneralizedCorrelation,
        GeneralizedCorrelation_2X3D_2000n4k_GpuCpu
) {
    int n = 2000;
    int k = 4;
    auto Xa = new CuArray<float>;
    Xa->init(
            1, 3 * n
    );
    auto Xb = new CuArray<float>;
    Xb->init(
            1, 3 * n
    );
    float incr = M_PI / (float) n;
    std::vector<float> domain(n);
    domain[0] = 0.001;
    for (int i = 1; i < n; i++) {
        domain[i] = domain[i - 1] + incr;
    }
    for (int i = 0; i < n; i++) {
        (*Xa)[i] = std::sin(domain[i]);
        (*Xa)[i + n] = std::cos(domain[i]);
        (*Xa)[i + 2 * n] = std::log(domain[i]);
        (*Xb)[i] = std::cos(domain[i]);
        (*Xb)[i + n] = 2 * domain[i];
        (*Xb)[i + 2 * n] = (float) std::log(domain[i] * 2.0);
    }
    float cpuGeneralizedCorrelation_ = cpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            3
    );
    float gpuGeneralizedCorrelation_ = gpuGeneralizedCorrelation(
            Xa,
            Xb,
            k,
            n,
            2,
            3
    );
    EXPECT_FLOAT_EQ(cpuGeneralizedCorrelation_,
                    gpuGeneralizedCorrelation_
    );

    delete Xa;
    delete Xb;
}

TEST(
        GeneralizedCorrelation,
        GeneralizedCorrelation_UsedCpuPlatform
) {
    int n = 1000;
    int k = 4;
    auto *X = new CuArray<float>;
    auto *R = new CuArray<float>;
    auto *ab = new CuArray<int>;

    X->init(
            2, n
    );
    float incr = M_PI / (float) n;
    std::vector<float> domain(n);
    domain[0] = 0.001;
    for (
            int i = 1;
            i < n;
            i++) {
        domain[i] = domain[i - 1] +
                    incr;
    }
    for (
            int i = 0;
            i < n;
            i++) {
        X->set(
                std::sin(domain[i]),
                0,
                i
        );
        X->set(
                std::cos(domain[i]),
                1,
                i
        );

    }
    ab->init(
            1, 2
    );
    ab->set(
            0, 0, 0
    );
    ab->set(
            1, 0, 1
    );
    ASSERT_EQ(
            generalizedCorrelation(
                    X, R, ab, k, n, 2, 1, 1
            ), 1);
    delete X;
    delete R;
    delete ab;
}

TEST(
        GeneralizedCorrelation,
        GeneralizedCorrelation_UsedGpuPlatform
) {
    int n = 1000;
    int k = 4;
    auto *X = new CuArray<float>;
    auto *R = new CuArray<float>;
    auto *ab = new CuArray<int>;

    X->init(
            2, n
    );
    float incr = M_PI / (float) n;
    std::vector<float> domain(n);
    domain[0] = 0.001;
    for (
            int i = 1;
            i < n;
            i++) {
        domain[i] = domain[i - 1] +
                    incr;
    }
    for (
            int i = 0;
            i < n;
            i++) {
        X->set(
                std::sin(domain[i]),
                0,
                i
        );
        X->set(
                std::cos(domain[i]),
                1,
                i
        );

    }
    ab->init(
            1, 2
    );
    ab->set(
            0, 0, 0
    );
    ab->set(
            1, 0, 1
    );
    ASSERT_EQ(
            generalizedCorrelation(
                    X, R, ab, k, n, 2, 1, 1
            ), 0);
    delete X;
    delete R;
    delete ab;
}

int main(
        int argc,
        char **argv
) {
    ::testing::InitGoogleTest(
            &argc,
            argv
    );
    return RUN_ALL_TESTS(
    );
}



