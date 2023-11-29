#include <iostream>
#include "cuarray.h"

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Creates a new float CuArray instance */
    auto *cuArray = new CuArray<float>();

/*
 * Initializes the CuArray with 10 rows and 5 columns
 * and allocates memory on host.
 */
    cuArray->init(10,
                  5);

    /* Print the cuArray */
    for (int i = 0; i < cuArray->m(); i++) {
        for (int j = 0; j < cuArray->n(); j++) {
            std::cout << cuArray->get(i,
                                      j) << " ";
        }
        std::cout << std::endl;
    }

/* Free the memory allocated on host and device */
    delete cuArray;

    return 0;
}