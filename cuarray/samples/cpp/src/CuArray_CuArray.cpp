#include <iostream>
#include "cuarray.h"

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

    /* Creates a new float CuArray instance */
    auto *cuArray = new CuArray<float>();

    /* Free memory */
    delete cuArray;

    return 0;
}