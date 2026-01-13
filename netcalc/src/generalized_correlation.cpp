//
// Created by andy on 4/17/23.
//
#include "generalized_correlation.h"
#include "mutual_information.h"
#include <stdexcept>

int netcalc::generalizedCorrelation(
        const std::unique_ptr<CuArray<float>> &X,
        const std::unique_ptr<CuArray<float>> &R,
        const std::unique_ptr<CuArray<int>> &ab,
        const int k,
        const int n,
        const int xd,
        const int d,
        const int platform,
        const int checkpointFrequency,
        std::string checkpointFileName) {
    if (checkpointFileName.size() > 4 && checkpointFileName.substr(checkpointFileName
                                                                                   .size() -
                                                                           4,
                                                                   4) == ".npy")
        checkpointFileName = checkpointFileName.substr(0,
                                                       checkpointFileName.size() -
                                                               4);
    ab->save(
            checkpointFileName + "_ab.npy");
    R->init(
            1,
            ab->m());

    for (int i = 0; i < ab->m(); i++) {
        int a = ab->get(i,
                        0);
        int b = ab->get(i,
                        1);
        auto Xa = std::make_unique<CuArray<float>>();
        auto Xb = std::make_unique<CuArray<float>>();
        Xa->fromCuArrayShallowCopy(
                X.get(),
                a,
                a,
                1,
                X->n());
        Xb->fromCuArrayShallowCopy(
                X.get(),
                b,
                b,
                1,
                X->n());
        if (platform == 0)
            R->set(
                    netcalc::generalizedCorrelationGpu(
                            Xa,
                            Xb,
                            k,
                            n,
                            xd,
                            d),
                    0,
                    i);
        else if (platform == 1)
            R->set(
                    netcalc::generalizedCorrelationCpu(
                            Xa,
                            Xb,
                            k,
                            n,
                            xd,
                            d),
                    0,
                    i);
        else {
            throw std::runtime_error("Invalid platform");
        }

        if (i % checkpointFrequency == 0) {
            std::remove(
                    (checkpointFileName + "_" +
                     std::to_string(i - checkpointFrequency) + ""
                                                               ".npy")
                            .c_str());
            R->save(
                    checkpointFileName + "_" + std::to_string(i) + ""
                                                                   ".npy");
        }
    }
    return platform;
}

int netcalc::generalizedCorrelation(
        const std::unique_ptr<CuArray<float>> &X,
        const std::unique_ptr<CuArray<float>> &R,
        const std::unique_ptr<CuArray<int>> &ab,
        const int k,
        const int n,
        const int xd,
        const int d,
        const int platform) {
    R->init(
            1,
            ab->m());

    for (int i = 0; i < ab->m(); i++) {
        const int a = ab->get(i,
                              0);
        const int b = ab->get(i,
                        1);
        auto Xa = std::make_unique<CuArray<float>>();
        auto Xb = std::make_unique<CuArray<float>>();
        Xa->fromCuArrayShallowCopy(
                X.get(),
                a,
                a,
                1,
                X->n());
        Xb->fromCuArrayShallowCopy(
                X.get(),
                b,
                b,
                1,
                X->n());
        if (platform == 0)
            R->set(
                    netcalc::generalizedCorrelationGpu(
                            Xa, Xb,
                            k,
                            n,
                            xd,
                            d),
                    0,
                    i);
        else if (platform == 1)
            R->set(
                    netcalc::generalizedCorrelationCpu(
                            Xa, Xb,
                            k,
                            n,
                            xd,
                            d),
                    0,
                    i);
        else {
            throw std::runtime_error("Invalid platform");
        }
    }
    return platform;
}
