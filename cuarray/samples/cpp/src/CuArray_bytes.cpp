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

/* Get the number of bytes the CuArray data occupies */
    auto bytes_ = cuArray->bytes();

/* Print the total number of bytes in cuArray. */
    std::cout
            << "Number of bytes: "
            << bytes_
            << std::endl;
/* Output:
 * Number of bytes: 200
 */

    delete cuArray;
    return 0;
}