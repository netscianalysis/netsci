#include <cuarray.h>
#include <iostream>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Creates a new float CuArray instance */
    CuArray<float> *cuArray = new CuArray<float>();

/* Create a linear float array that has 10 rows and 10 columns.*/
    auto *NUMPY_ARRAY = new float[100];
    int rows = 10;
    int cols = 10;

/* Fill the NUMPY_ARRAY with random values */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            NUMPY_ARRAY[i * cols + j] =
                    (float) rand() / (float) RAND_MAX;
        }
    }

/* Copy the NUMPY_ARRAY data into the CuArray. The
 * CuArray has the same dimensions as the NUMPY_ARRAY. */
    cuArray->fromNumpy(
            NUMPY_ARRAY,
            rows,
            cols
    );

/* Print the CuArray. */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout
                    << cuArray->host()[i * cols + j]
                    << " ";
        }
        std::cout
                << std::endl;
    }
    std::cout
            << std::endl;


/* Free the NUMPY_ARRAY and CuArray. */
    delete cuArray;
    delete[] NUMPY_ARRAY;
    return 0;
}