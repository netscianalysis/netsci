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

/*
 * Create a float 'CuArray' that
 * will be a deep copy of the last two cuArray rows
 */
    auto cuArray2x3Copy = new CuArray<float>;
    cuArray2x3Copy->init(2,
                         3);

/* First row to copy from cuArray into cuArray2x3Copy */
    int startRowIndex = 1;

/* Last row to copy from cuArray into cuArray2x3Copy */
    int endRowIndex = 2;

    cuArray2x3Copy->fromCuArrayDeepCopy(
            cuArray, /*Source for copying data into cuArray2x3Copy. This method is
            * significantly safer than its shallow copy equivalent. However, it is also
            * slower, which can impact performance if it's called a lot.*/
            startRowIndex, /* First row to copy from cuArray into cuArray2x3Copy */
            endRowIndex, /* Last row to copy from cuArray into cuArray2x3Copy */
            cuArray2x3Copy->m(), /* Number of rows in cuArray2x3Copy */
            cuArray2x3Copy->n() /* Number of columns in cuArray2x3Copy */
    );

/* Print each element in cuArray2x3Copy */
    for (int i = 0; i < cuArray2x3Copy->m(); i++) {
        for (int j = 0; j < cuArray2x3Copy->n(); j++) {
            std::cout << cuArray2x3Copy->get(i,
                                             j) << " ";
        }
        std::cout << std::endl;
    }
/* Output:
 * 3 4 5
 * 6 7 8
 */

/* Both cuArray and cuArray2x3Copy own their data.*/
    std::cout
            << cuArray->owner() << " "
            << cuArray2x3Copy->owner()
            << std::endl;
/* Output:
 * 1 1
 */

    delete cuArray2x3Copy;
    delete cuArray;
    return 0;
}