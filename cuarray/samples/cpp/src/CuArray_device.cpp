#include <cuarray.h>
#include <iostream>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Creates a new float CuArray instance */
    CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 3 rows and 3 columns */
    cuArray->init(3,
                  3);

/*Set each i, j element equal to i*3 + j */
    for (int i = 0; i < 9; i++) {
        cuArray->host()[i] = i;
    }

/* Allocate device memory. */
    cuArray->allocateDevice();

/* Copy data from host to device. */
    cuArray->toDevice();

/* Set deviceArray equal to cuArray's device data via the
 * device() method, */
    auto deviceArray = cuArray->device();
/* which can be used in CUDA kernels.
 * Eg.) <<<1, 1>>>kernel(deviceArray)*/


/* delete frees both host and device memory. */
    delete cuArray;
    return 0;
}