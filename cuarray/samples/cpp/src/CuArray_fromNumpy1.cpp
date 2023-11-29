#include <cuarray.h>
#include <iostream>
#include <random>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Creates a new float CuArray instance */
    CuArray<float> *cuArray = new CuArray<float>();

/* Create a float vector with 10 elements.*/
    auto *NUMPY_ARRAY = new float[10];
    int rows = 10;

/* Fill the NUMPY_ARRAY with random values */
    for (int i = 0; i < rows; i++) {
        NUMPY_ARRAY[i] =
                (float) rand() / (float) RAND_MAX;
    }

/* Copy the NUMPY_ARRAY data into the CuArray. The
 * CuArray has the same dimensions as the NUMPY_ARRAY. */
    cuArray->fromNumpy(
            NUMPY_ARRAY,
            rows
    );

/* Print the CuArray. */
    for (int i = 0; i < rows; i++) {
        std::cout
                << cuArray->host()[i]
                << " ";
    }
    std::cout
            << std::endl;


/* Free the NUMPY_ARRAY and CuArray. */
    delete cuArray;
    delete[] NUMPY_ARRAY;
    return 0;
}