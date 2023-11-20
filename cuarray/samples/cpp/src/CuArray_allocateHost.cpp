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

/* Fill the CuArray with random values */
    for (int i = 0; i < cuArray->m(); i++) {
        for (int j = 0; j < cuArray->n(); j++) {
            cuArray->host()[i * cuArray->n() + j] =
                    static_cast<float>(rand() / (float) RAND_MAX);
        }
    }
/* Allocate device memory. */
    cuArray->allocateDevice();

/* Copy data from host to device. */
    cuArray->toDevice();

/* Free host memory, since it is no longer needed.*/
    cuArray->deallocateHost();

/*Do some complicated GPU calculations
* and then allocate host memory when you need it again.
* Also, this is extremely wasteful, it's just an example of
* how to use this method. Realistically, most users will never have
* to manually allocate host memory as that is handled by the
* init methods. If memory allocation is successful, allocateHost
 * returns 0*/
    auto err = cuArray->allocateHost();

    /* Check if host memory allocation was successful. */
    if (err == 0) {
        std::cout
                << "Host memory allocated successfully."
                << std::endl;
    } else {
        std::cout
                << "Host memory allocation failed."
                << std::endl;
    }

/* Copy data from device to host. */
    cuArray->toHost();

    delete cuArray;
    return 0;
}