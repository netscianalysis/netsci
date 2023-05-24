//
// Created by andy on 4/5/23.
//
#include <gtest/gtest.h>
#include <cmath>
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
    float mutualInformationCpu_ = netcalc::mutualInformationCpu(
            Xa,
            Xb,
            k,
            n,
            2,
            1
    );
    float mutualInformationGpu_ = netcalc::mutualInformationGpu(
            Xa,
            Xb,
            k,
            n,
            2,
            1
    );
    EXPECT_FLOAT_EQ(mutualInformationCpu_, mutualInformationGpu_);
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
    float cpuMutualInformation_ = netcalc::mutualInformationCpu(
            Xa,
            Xb,
            k,
            n,
            2,
            1
    );
    float gpuMutualInformation_ = netcalc::mutualInformationGpu(
            Xa,
            Xb,
            k,
            n,
            2,
            1
    );
    EXPECT_FLOAT_EQ(cpuMutualInformation_, gpuMutualInformation_);
    delete Xa;
    delete Xb;
}


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
    float cpuMutualInformation_ = netcalc::mutualInformationCpu(
            Xa,
            Xb,
            k,
            n,
            2,
            2
    );
    float gpuMutualInformation_ = netcalc::mutualInformationGpu(
            Xa,
            Xb,
            k,
            n,
            2,
            2
    );
    EXPECT_FLOAT_EQ(cpuMutualInformation_, gpuMutualInformation_);
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
    float cpuMutualInformation_ = netcalc::mutualInformationCpu(
            Xa,
            Xb,
            k,
            n,
            2,
            2
    );
    float gpuMutualInformation_ = netcalc::mutualInformationGpu(
            Xa,
            Xb,
            k,
            n,
            2,
            2
    );
    EXPECT_FLOAT_EQ(cpuMutualInformation_, gpuMutualInformation_);

    delete Xa;
    delete Xb;
}

TEST(
        MutualInformation,
        MutualInformation2X3D_1000n4k_GpuCpu
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
        (*Xb)[i + 2 * n] = (float)std::log(domain[i]*2.0);
    }
    float cpuMutualInformation_ = netcalc::mutualInformationCpu(
            Xa,
            Xb,
            k,
            n,
            2,
            3
    );
    float gpuMutualInformation_ = netcalc::mutualInformationGpu(
            Xa,
            Xb,
            k,
            n,
            2,
            3
    );
    EXPECT_FLOAT_EQ(
            cpuMutualInformation_,
            gpuMutualInformation_
    );
    delete Xa;
    delete Xb;
}


TEST(
        MutualInformation,
        MutualInformation2X3D_2000n4k_GpuCpu
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
    float cpuMutualInformation_ = netcalc::mutualInformationCpu(
            Xa,
            Xb,
            k,
            n,
            2,
            3
    );
    float gpuMutualInformation_ = netcalc::mutualInformationGpu(
            Xa,
            Xb,
            k,
            n,
            2,
            3
    );
    EXPECT_FLOAT_EQ(
            cpuMutualInformation_,
            gpuMutualInformation_
    );

    delete Xa;
    delete Xb;
}

TEST(
        MutualInformation,
       MutualInformation_UsedCpuPlatform
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
            netcalc::mutualInformation(
                    X, R, ab, k, n, 2, 1, 1
            ), 1);
    delete X;
    delete R;
    delete ab;
}

TEST(
        MutualInformation,
       MutualInformation_UsedGpuPlatform
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
            netcalc::mutualInformation(
                    X, R, ab, k, n, 2, 1, 0
            ), 0);
    delete X;
    delete R;
    delete ab;
}

int main(
        int argc,
        char **argv
) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}







