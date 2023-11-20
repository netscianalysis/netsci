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
 * Initializes the CuArray with 10 rows and 5 rows
 * and allocates memory on host.
 */
    cuArray->init(10,
                  5);

/* Get the number of rows in the CuArray */
    int m = cuArray->m();

/* Print the number of rows */
    std::cout
            << "Number of rows: "
            << m
            << std::endl;
/* Output:
 * Number of rows: 10
 */

    delete cuArray;
    return 0;
}