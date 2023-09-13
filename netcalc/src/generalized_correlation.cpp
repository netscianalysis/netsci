//
// Created by andy on 4/17/23.
//
#include <stdexcept>
#include "mutual_information.h"
#include "generalized_correlation.h"

int netcalc::generalizedCorrelation(
        CuArray<float> *X,
        CuArray<float> *R,
        CuArray<int> *ab,
        int k,
        int n,
        int xd,
        int d,
        int platform,
        int checkpointFrequency,
        std::string checkpointFileName
) {
    if (checkpointFileName.size() > 4 && checkpointFileName.substr
            (checkpointFileName
                     .size() - 4,
             4) == ".npy")
        checkpointFileName = checkpointFileName.substr(0,
                                                       checkpointFileName.size() -
                                                       4);
    ab->save(
            checkpointFileName + "_ab.npy"
    );
    R->init(
            1,
            ab->m()
    );

    for (int i = 0; i < ab->m(); i++) {
        int a = ab->get(i,
                        0);
        int b = ab->get(i,
                        1);
        auto Xa = new CuArray<float>;
        auto Xb = new CuArray<float>;
        Xa->fromCuArrayShallowCopy(
                X,
                a,
                a,
                1,
                X->n()
        );
        Xb->fromCuArrayShallowCopy(
                X,
                b,
                b,
                1,
                X->n()
        );
        if (platform == 0)
            R->set(
                    netcalc::generalizedCorrelationGpu(
                            Xa,
                            Xb,
                            k,
                            n,
                            xd,
                            d
                    ),
                    0,
                    i
            );
        else if (platform == 1)
            R->set(
                    netcalc::generalizedCorrelationCpu(
                            Xa,
                            Xb,
                            k,
                            n,
                            xd,
                            d
                    ),
                    0,
                    i
            );
        else {
            throw std::runtime_error("Invalid platform");
        }

        if (i % checkpointFrequency == 0) {
            std::remove(
                    (checkpointFileName + "_" +
                     std::to_string(i - checkpointFrequency)
                     + ""
                       ".npy").c_str()
            );
            R->save(
                    checkpointFileName + "_" + std::to_string(i) + ""
                                                                   ".npy"
            );
        }
        delete Xa;
        delete Xb;
    }
    return platform;
}

int netcalc::generalizedCorrelation(
        CuArray<float> *X,
        CuArray<float> *R,
        int k,
        int n,
        int xd,
        int d,
        int platform,
        int checkpointFrequency,
        std::string checkpointFileName,
        const std::string &restartRFileName,
        const std::string &restartAbFileName
) {
    if (checkpointFileName.size() > 4 && checkpointFileName.substr
            (checkpointFileName
                     .size() - 4,
             4) == ".npy")
        checkpointFileName = checkpointFileName.substr(0,
                                                       checkpointFileName.size() -
                                                       4);
    R->load(
            restartRFileName
    );
    auto ab = new CuArray<int>;
    ab->load(
            restartAbFileName
    );
    auto restartAb = new CuArray<int>;
    auto restartIndex = netcalc::generateRestartAbFromCheckpointFile(
            ab,
            restartAb,
            restartRFileName
    ) + 1;
    for (int i = 0; i < restartAb->m(); i++) {
        int a = restartAb->get(i,
                        0);
        int b = restartAb->get(i,
                        1);
        auto Xa = new CuArray<float>;
        auto Xb = new CuArray<float>;
        Xa->fromCuArrayShallowCopy(
                X,
                a,
                a,
                1,
                X->n()
        );
        Xb->fromCuArrayShallowCopy(
                X,
                b,
                b,
                1,
                X->n()
        );
        if (platform == 0)
            R->set(
                    netcalc::generalizedCorrelationGpu(
                            Xa,
                            Xb,
                            k,
                            n,
                            xd,
                            d
                    ),
                    0,
                   restartIndex
            );
        else if (platform == 1)
            R->set(
                    netcalc::generalizedCorrelationCpu(
                            Xa,
                            Xb,
                            k,
                            n,
                            xd,
                            d
                    ),
                    0,
                   restartIndex
            );
        else {
            throw std::runtime_error("Invalid platform");
        }

        if (restartIndex % checkpointFrequency == 0) {
            std::remove(
                    (checkpointFileName + "_" +
                     std::to_string(restartIndex - checkpointFrequency)
                     + ""
                       ".npy").c_str()
            );
            R->save(
                    checkpointFileName + "_" + std::to_string(restartIndex)
                    + ""
                                                                   ".npy"
            );
        }
        restartIndex++;
        delete Xa;
        delete Xb;
    }
    delete ab;
    delete restartAb;
    return platform;
}

int netcalc::generalizedCorrelation(
        CuArray<float> *X,
        CuArray<float> *R,
        CuArray<int> *ab,
        int k,
        int n,
        int xd,
        int d,
        int platform
) {
    R->init(
            1,
            ab->m()
    );

    for (int i = 0; i < ab->m(); i++) {
        int a = ab->get(i,
                        0);
        int b = ab->get(i,
                        1);
        auto Xa = new CuArray<float>;
        auto Xb = new CuArray<float>;
        Xa->fromCuArrayShallowCopy(
                X,
                a,
                a,
                1,
                X->n()
        );
        Xb->fromCuArrayShallowCopy(
                X,
                b,
                b,
                1,
                X->n()
        );
        if (platform == 0)
            R->set(
                    netcalc::generalizedCorrelationGpu(
                            Xa,
                            Xb,
                            k,
                            n,
                            xd,
                            d
                    ),
                    0,
                    i
            );
        else if (platform == 1)
            R->set(
                    netcalc::generalizedCorrelationCpu(
                            Xa,
                            Xb,
                            k,
                            n,
                            xd,
                            d
                    ),
                    0,
                    i
            );
        else {
            throw std::runtime_error("Invalid platform");
        }
        delete Xa;
        delete Xb;
    }
    return platform;
}

