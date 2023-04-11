//
// Created by andy on 4/5/23.
//
#include <gtest/gtest.h>
#include "mutual_information.h"
#include "psi.h"
#include "cnpy.h"

TEST(
        MutualInformation,
        MutualInformation2X1D_1000n4k09covGaussian_GpuCpu
) {
    int n = 1000;
    int k = 4;
    auto Xnp = cnpy::npy_load(
            "data/2X_1D_1000_4.npy"
    );
    auto X = Xnp.data<double>();
    auto Xa = new CuArray<float>;
    Xa->init(
            1, n
    );
    auto Xb = new CuArray<float>;
    Xb->init(
            1, n
    );
    for (int i = 0; i < n; i++) {
        (*Xa)[i] = X[i];
        (*Xb)[i] = X[i + n];
    }
    float cpuMutualInformation = cpuMutualInformation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    float gpuMutualInformation = gpuMutualInformation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    EXPECT_FLOAT_EQ(cpuMutualInformation, gpuMutualInformation);
    delete Xa;
    delete Xb;
}

TEST(
        MutualInformation,
        MutualInformation2X1D_2000n4k09covGaussian_GpuCpu
) {
    int n = 2000;
    int k = 4;
    auto Xnp = cnpy::npy_load(
            "data/2X_1D_2000_4.npy"
    );
    auto X = Xnp.data<double>();
    auto Xa = new CuArray<float>;
    Xa->init(
            1,
            n
    );
    auto Xb = new CuArray<float>;
    Xb->init(
            1,
            n
    );
    for (int i = 0; i < n; i++) {
        (*Xa)[i] = X[i];
        (*Xb)[i] = X[i + n];
    }
    float cpuMutualInformation = cpuMutualInformation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    float gpuMutualInformation = gpuMutualInformation2X1D(
            Xa,
            Xb,
            k,
            n
    );
    EXPECT_FLOAT_EQ(cpuMutualInformation, gpuMutualInformation);
    delete Xa;
    delete Xb;
}


#include <cmath>

TEST(
        MutualInformation,
        MutualInformation2X2D_1000n4k_GpuCpu
) {
    int n = 1000;
    int k = 4;
    auto Xa = new CuArray<float>;
    Xa->init(
            1,
            2 * n
    );
    auto Xb = new CuArray<float>;
    Xb->init(
            1,
            2 * n
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
    float cpuMutualInformation = cpuMutualInformation2X2D(
            Xa,
            Xb,
            k,
            n
    );
    float gpuMutualInformation = gpuMutualInformation2X2D(
            Xa,
            Xb,
            k,
            n
    );
    EXPECT_FLOAT_EQ(cpuMutualInformation, gpuMutualInformation);
    delete Xa;
    delete Xb;
}


TEST(
        MutualInformation,
        MutualInformation2X2D_2000n4k_GpuCpu
) {
    int n = 2000;
    int k = 4;
    auto Xa = new CuArray<float>;
    Xa->init(
            1, 2 * n
    );
    auto Xb = new CuArray<float>;
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
    float cpuMutualInformation = cpuMutualInformation2X2D(
            Xa,
            Xb,
            k,
            n
    );
    float gpuMutualInformation = gpuMutualInformation2X2D(
            Xa,
            Xb,
            k,
            n
    );
    EXPECT_FLOAT_EQ(cpuMutualInformation, gpuMutualInformation);

    delete Xa;
    delete Xb;
}

int main(
        int argc,
        char **argv
) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}







