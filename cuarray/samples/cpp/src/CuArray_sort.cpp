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

/* Create a new CuArray that contains the sorted data from the
 * 8th row of the original CuArray. */
    auto sortedCuArray = cuArray->sort(7);

/* Print the sorted CuArray. */
    for (int j = 0; j < sortedCuArray->n(); j++) {
        std::cout
                << sortedCuArray->get(0,
                                      j)
                << std::endl;
    }

/* Cleanup time. */
    delete cuArray;
    delete sortedCuArray;
    return 0;
}