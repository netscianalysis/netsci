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
 * will be a shallow copy of the last two cuArray rows
 */
    auto cuArray2x3Copy = new CuArray<float>;
    cuArray2x3Copy->init(2,
                         3);

/* First row to copy from cuArray into cuArray2x3Copy */
    int startRowIndex = 1;

/* Last row to copy from cuArray into cuArray2x3Copy */
    int endRowIndex = 2;

    cuArray2x3Copy->fromCuArrayShallowCopy(
            cuArray, /* Source for copying data into cuArray2x3Copy.
            * Both cuArray and cuArray2x3Copy will point to the same
            * data, which helps with
            * performance at the expense of being extremely dangerous. As an
            * attempt to make this method somewhat safe, there is an "owner"
            * attribute that is set to 1 if the CuArray owns the data and 0
            * otherwise. Logic is implemented in the destructor to check for ownership
            * and only delete data if the CuArray owns the data. As of now, this method has
            * passed all real life stress tests, and CUDA-MEMCHECK doesn't hate it,
            * but it still shouldn't be used in the vast majority of cases.
            * The legitimate reason this should ever be called is when you have to
            * pass the CuArray data as a double pointer to a function that
            * cannot itself take a CuArray object. Eg.) A CUDA kernel.*/
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
    delete cuArray2x3Copy;
    delete cuArray;
    return 0;
}