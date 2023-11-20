#include <cuarray.h>
#include <iostream>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;
/* Creates a new float CuArray instance */
    CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 3 rows and 3 columns */
    cuArray->init(3,
                  3);

/*Set each i, j element equal to i*3 + j */
    for (int i = 0; i < 9; i++) {
        cuArray->host()[i] = i;
    }

/* Print each element in cuArray's host memory.
 * The host data is linear and stored in row major order. To
 * access element i,j you would use the linear index
 * i*n+j, where n is the number of columns.*/
    for (int i = 0; i < cuArray->m(); i++) {
        for (int j = 0; j < cuArray->n(); j++) {
            std::cout << cuArray->host()[i * cuArray->n() + j] << " ";
        }
        std::cout << std::endl;
    }
/* Output:
 * 0 1 2
 * 3 4 5
 * 6 7 8
 */

    delete cuArray;
    return 0;
}