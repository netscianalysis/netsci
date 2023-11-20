#include "cuarray.h"
#include <iostream>
#include <random>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;
/* Creates a new float CuArray instance 1 row and 10 columns*/
    CuArray<float> *cuArray = new CuArray<float>();
    int m = 1; /* Number of rows */
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
            cols
    );

/* Print the NUMPY_ARRAY data and the CuArray data. */
    for (int i = 0; i < n; i++) {
        std::cout
                << cuArray->get(0,
                                i)
                << " ";
        std::cout
                << (*(NUMPY_ARRAY))[i]
                << std::endl;
    }

/* Clean this mess up. Makes you appreciate std::vectors :).*/
    delete cuArray;
    delete[] NUMPY_ARRAY[0];
    delete[] NUMPY_ARRAY;
    delete[] cols[0];
    delete[] cols;
    return 0;
}