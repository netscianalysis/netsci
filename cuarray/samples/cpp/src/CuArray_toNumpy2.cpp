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

/* Create a double pointer to a float array. It will
 * store the data from the CuArray. */
    auto NUMPY_ARRAY = new float *[1];

/* Create two double pointer int arrays that will store
 * the number rows and columns in the CuArray.
 * Btw this is what the NumPy C backend is doing every time
 * you create a numpy array in Python*/
    auto rows = new int *[1];
    auto cols = new int *[1];

/* Fill the CuArray with random values */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cuArray->set((float) rand() / (float) RAND_MAX,
                         i,
                         j);
        }
    }

/* Copy the CuArray data into the NUMPY_ARRAY. The
 * NUMPY_ARRAY has the same dimensions as the CuArray. */
    cuArray->toNumpy(
            NUMPY_ARRAY,
            rows,
            cols
    );

/* Print the NUMPY_ARRAY data and the CuArray data. */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout
                    << cuArray->get(i,
                                    j)
                    << " ";
            std::cout
                    << (*(NUMPY_ARRAY))[i * m + j]
                    << std::endl;
        }
        std::cout
                << std::endl;
    }

/* Clean this mess up. Makes you appreciate std::vectors :).*/
    delete cuArray;
    delete[] NUMPY_ARRAY[0];
    delete[] NUMPY_ARRAY;
    delete[] rows[0];
    delete[] rows;
    delete[] cols[0];
    delete[] cols;
    return 0;
}