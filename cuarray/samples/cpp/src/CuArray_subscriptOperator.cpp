#include <cuarray.h>
#include <iostream>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Create a new float CuArray instance */
    auto cuArray = new CuArray<float>;

/* Initialize the CuArray with 3 rows and 3 columns */
    cuArray->init(3,
                  3);

/*Set each i, j element equal to i*3 + j */
    for (int i = 0; i < 9; i++) {
        cuArray->host()[i] = i;
    }

/* Calculate the linear index that
 * retrieves the 3rd element in the 2nd row of the CuArray. */
    int i = 1;
    int j = 2;
    int linearIndex = i * cuArray->n() + j;
    auto ijLinearVal = (*(cuArray))[linearIndex];
    auto ijVal = cuArray->get(i,
                              j);

/* Print the values at the linear index and the (i, j) index. */
    std::cout
            << ijLinearVal
            << " "
            << ijVal
            << std::endl;

/*Deallocate memory*/
    delete cuArray;
    return 0;
}