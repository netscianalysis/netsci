#include <cuarray.h>
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

/* Get the total number of values in the CuArray */
    int size = cuArray->size();

/* Print the total number of values in cuArray. */
    std::cout
            << "Number of values: "
            << size
            << std::endl;
/* Output:
 * Number of values: 50
 */

    delete cuArray;
    return 0;
}