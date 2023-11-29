#include <cuarray.h>
#include <random>
#include <iostream>

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

/* Check if host memory is allocated. If it is,
 * allocatedHost() will return 1, other wise it
 * will return 0. This is convenient for boolean checks.*/
    auto hostMemoryAllocated = cuArray->allocatedHost();

/* Print whether or not host memory is allocated. */
    std::cout
            << "Host memory allocated: "
            << hostMemoryAllocated
            << std::endl;

    delete cuArray;
    return 0;
}