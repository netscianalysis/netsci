//
// Created by andy on 4/4/23.
//
#include "gtest/gtest.h"
#include "cuarray.h"
#include "psi.h"
#include <cmath>
#include "mutual_information.h"

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
    float cpuGeneralizedCorrelation = cpuGeneralizedCorrelation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    float gpuGeneralizedCorrelation = gpuGeneralizedCorrelation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    EXPECT_FLOAT_EQ(cpuGeneralizedCorrelation,
                    gpuGeneralizedCorrelation
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
    float cpuGeneralizedCorrelation = cpuGeneralizedCorrelation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    float gpuGeneralizedCorrelation = gpuGeneralizedCorrelation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    EXPECT_FLOAT_EQ(cpuGeneralizedCorrelation,
                    gpuGeneralizedCorrelation);
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
    float cpuGeneralizedCorrelation = cpuGeneralizedCorrelation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    float gpuGeneralizedCorrelation = gpuGeneralizedCorrelation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    EXPECT_FLOAT_EQ(cpuGeneralizedCorrelation,
                    gpuGeneralizedCorrelation
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
    float cpuGeneralizedCorrelation = cpuGeneralizedCorrelation2X2D(
            Xa,
            Xb,
            k,
            n
    );
    float gpuGeneralizedCorrelation = gpuGeneralizedCorrelation2X2D(
            Xa,
            Xb,
            k,
            n
    );
    EXPECT_FLOAT_EQ(cpuGeneralizedCorrelation,
                    gpuGeneralizedCorrelation);
    delete Xa;
    delete Xb;
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



