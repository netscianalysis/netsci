#include "cuarray.h"
#include <iostream>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Creates a new float CuArray instance */
    CuArray<float> *cuArray = new CuArray<float>();

/*
 * Initializes the CuArray with 10 rows and 5 columns
 * and allocates memory on host.
 */
    cuArray->init(10,
                  5);

/* Create a 50-element float vector and fill it with random values */
    auto a = new float[50];
    for (int i = 0; i < 50; i++) {
        a[i] = static_cast<float>(rand() / (float) RAND_MAX);
    }

/* Initialize the CuArray with data from "a", preserving
 * overall size while setting new dimensions
 * (similar to NumPy's reshape method). */
    cuArray->init(a,
                  10,
                  5);

/* Print each element in cuArray's host memory.
 * The host data is linear and stored in row major order. To
 * access element i,j you would use the linear index
 * i*n+j, where n is the number of columns.*/
    for (int i = 0; i < cuArray->m(); i++) {
        for (int j = 0; j < cuArray->n(); j++) {
            std::cout << cuArray->get(i,
                                      j) << " ";
            std::cout << a[i * cuArray->n() + j] << std::endl;
        }
        std::cout << std::endl;
    }

/* Delete "a" and cuArray */
    delete[] a;
    delete cuArray;
    return 0;
}