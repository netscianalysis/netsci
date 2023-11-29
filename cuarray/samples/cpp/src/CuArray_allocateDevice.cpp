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
/* Allocate device memory. If successful, allocateDevice returns 0.*/
    auto err = cuArray->allocateDevice();

    /* Check if device memory allocation was successful. */
    if (err == 0) {
        std::cout
                << "Device memory allocated successfully."
                << std::endl;
    } else {
        std::cout
                << "Device memory allocation failed."
                << std::endl;
    }

/* Frees host and device memory. */
    delete cuArray;
    return 0;
}