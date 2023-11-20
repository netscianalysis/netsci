#include <cuarray.h>
#include <iostream>
#include <random>

int main() {

    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Creates a new float CuArray instance */
    CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
    auto rows = 300;
    auto cols = 300;
    cuArray->init(rows,
                  cols);

/* Allocate device memory. */
    cuArray->allocateDevice();

/* Check if device memory is allocated. If it is,
 * allocatedDevice() will return 1, other wise it
 * will return 0. This is convenient for boolean checks.*/
    auto deviceMemoryAllocated = cuArray->allocatedDevice();

/* Print whether or not device memory is allocated. */
    std::cout
            << "Device memory allocated: "
            << deviceMemoryAllocated
            << std::endl;

    delete cuArray;

    return 0;
}