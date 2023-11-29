#include "cuarray.h"
#include <iostream>

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

/* As it's name implies, set(value, i, j) sets the value at the
 * specified position (i, j) in the CuArray. */

/* Use the set method to set the value at each position in the CuArray
 * to a random number.*/
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cuArray->set((float) rand() / (float) RAND_MAX,
                         i,
                         j);
        }
    }

/* Print the CuArray. */
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