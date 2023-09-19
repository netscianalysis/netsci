#include <string>
#include <iostream>
#include "network.h"
#include "cuarray.h"

int main() {
    std::string dcdFile =
            "/home/astokely/CLionProjects/netsci/tests/netchem/cpp"
            "/data/test.dcd";
    std::string pdbFile = "/home/astokely/CLionProjects/netsci/tests/netchem/cpp"
                          "/data/test.pdb";
    int firstFrame = 0;
    int lastFrame = 9;
    int stride = 3;
    auto network3 = new Network();
    auto network1 = new Network();
    network3->init(
            dcdFile,
            pdbFile,
            firstFrame,
            lastFrame,
            stride
    );
    network1->init(
            dcdFile,
            pdbFile,
            firstFrame,
            lastFrame,
            1
    );
    auto nodeCoordinates3 = network3->nodeCoordinates();
    auto nodeCoordinates1 = network1->nodeCoordinates();
    for (int i = 0; i < 290; i++) {
        for (int j = 0; j < network3->numFrames(); j++) {
            std::cout << nodeCoordinates3->get(i, j) << " ";
            std::cout << nodeCoordinates3->get(i, j + network3->numFrames()) <<
            " ";
            std::cout << nodeCoordinates3->get(i, j + 2*network3->numFrames()) <<
            " ";
            std::cout << j * stride << std::endl;
        }
        std::cout << std::endl;
        for (int j = 0; j < network1->numFrames(); j++) {
            std::cout << nodeCoordinates1->get(i, j) << " ";
            std::cout << nodeCoordinates1->get(i, j + network1->numFrames()) <<
            " ";
            std::cout << nodeCoordinates1->get(i, j + 2*network1->numFrames()) <<
            " ";
            std::cout << j * 1 << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
    return 0;
}