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

/* Get the number of columns in the CuArray */
    int n = cuArray->n();

/* Print the number of columns */
    std::cout
            << "Number of columns: "
            << n
            << std::endl;
/* Output:
 * Number of columns: 5
 */

    delete cuArray;
    return 0;
}