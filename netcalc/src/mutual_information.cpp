//
// Created by andy on 4/11/23.
//
#include <iostream>
#include <regex>
#include "mutual_information.h"

int netcalc::mutualInformation(
        CuArray<float> *X,
        CuArray<float> *I,
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
    I->init(
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
            I->set(
                    netcalc::mutualInformationGpu(
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
            I->set(
                    netcalc::mutualInformationCpu(
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
            I->save(
                    checkpointFileName + "_" + std::to_string(i) + ""
                                                                   ".npy"
            );
        }
        delete Xa;
        delete Xb;
    }
    return platform;
}

int netcalc::mutualInformation(
        CuArray<float> *X,
        CuArray<float> *I,
        CuArray<int> *ab,
        int k,
        int n,
        int xd,
        int d,
        int platform
) {
    I->init(
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
            I->set(
                    netcalc::mutualInformationGpu(
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
            I->set(
                    netcalc::mutualInformationCpu(
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

void netcalc::generateRestartAbFromCheckpointFile(
        CuArray<int> *ab,
        CuArray<int> *restartAb,
        const std::string &checkpointFileName
) {
    size_t lastSlashPos = checkpointFileName.find_last_of('/');
    int lastAbIndex;
    if (lastSlashPos != std::string::npos) {
        // Extract the file name from the path
        std::string fileName = checkpointFileName.substr(
                lastSlashPos + 1);
        size_t lastUnderscorePos = fileName.find_last_of('_');
        std::string lastAbIndexStr = fileName.substr(lastUnderscorePos + 1);
        lastAbIndex = std::stoi(lastAbIndexStr);
    }
    else {
        size_t lastUnderscorePos = checkpointFileName.find_last_of('_');
        std::string lastAbIndexStr = checkpointFileName.substr(lastUnderscorePos + 1);
        lastAbIndex = std::stoi(lastAbIndexStr);
    }
    restartAb->fromCuArrayDeepCopy(
            ab,
            lastAbIndex + 1,
            ab->m() - 1,
            ab->m() - lastAbIndex - 1,
            2
    );
}
