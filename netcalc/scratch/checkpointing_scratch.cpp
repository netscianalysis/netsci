#include <iostream>
#include "mutual_information.h"

int main() {
    int n = 290;
    auto ab = new CuArray<int>;
    ab->init(n*n, 2);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 290; j++) {
            ab->set(i, i*n+j, 0);
            ab->set(j, i*n+j, 1);
        }
    }
    std::string checkpointFileName = "test_84000.npy";
    auto restartAb = new CuArray<int>;
    netcalc::generateRestartAbFromCheckpointFile(
            ab,
            restartAb,
            checkpointFileName
            );

    for (int i = 0; i < restartAb->m(); i++) {
        std::cout
        << restartAb->get(i, 0) << " "
        << restartAb->get(i, 1) << std::endl;
    }
    delete ab;
    delete restartAb;
    return 0;
}