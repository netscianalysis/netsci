#include "cuarray.h"
#include <iostream>
#include <random>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Creates a new float CuArray instance that will have 10 rows
 * and 10 columns*/
    CuArray<float> *cuArray = new CuArray<float>();
    int m = 10; /* Number of rows */
    int n = 10; /* Number of columns */
    cuArray->init(m,
                  n);

/* Fill the CuArray with random values */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cuArray->set((float) rand() / (float) RAND_MAX,
                         i,
                         j);
        }
    }

/* As it's name implies, get(i, j) returns the value at the
 * specified position (i, j) in the CuArray. */

/* Use the get method to print the value at each position in the CuArray. */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout
                    << cuArray->get(i,
                                    j)
                    << " ";
        }
        std::cout
                << std::endl;
    }

/* Free the CuArray. */
    delete cuArray;
    return 0;
}