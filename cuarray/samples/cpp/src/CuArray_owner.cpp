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
            cuArray, /* Source for copying data into cuArray2x3Copy. See
                  * CuArray::fromCuArrayShallowCopy for more info. */
            startRowIndex, /* First row to copy from cuArray into cuArray2x3Copy */
            endRowIndex, /* Last row to copy from cuArray into cuArray2x3Copy */
            cuArray2x3Copy->m(), /* Number of rows in cuArray2x3Copy */
            cuArray2x3Copy->n() /* Number of columns in cuArray2x3Copy */
    );

/* Now make another CuArray that is a deep copy of cuArray2x3Copy */
    auto cuArray2x3DeepCopy = new CuArray<float>;
    cuArray2x3DeepCopy->init(2,
                             3);
    cuArray2x3DeepCopy->fromCuArrayDeepCopy(
            cuArray, /* Source for copying data into cuArray2x3DeepCopy. See
                  * CuArray::fromCuArrayDeepCopy for more info. */
            startRowIndex, /* First row to copy from cuArray into cuArray2x3DeepCopy */
            endRowIndex, /* Last row to copy from cuArray into cuArray2x3DeepCopy */
            cuArray2x3DeepCopy->m(), /* Number of rows in cuArray2x3DeepCopy */
            cuArray2x3DeepCopy->n() /* Number of columns in cuArray2x3DeepCopy */
    );

/* Check if cuArray2x3Copy owns the host data. */
    auto cuArray2x3CopyOwnsHostData = cuArray2x3Copy->owner();

/* Check if cuArray2x3DeepCopy owns the host data.
 * Sorry for the verbosity :), I'm sure this is painful for
 * Python devs to read (though Java devs are probably loving it).*/
    auto cuArray2x3DeepCopyOwnsHostData = cuArray2x3DeepCopy->owner();

/* Print data in both arrays. */
    for (int i = 0; i < cuArray2x3Copy->m(); i++) {
        for (int j = 0; j < cuArray2x3Copy->n(); j++) {
            std::cout
                    << cuArray2x3Copy->get(i,
                                           j)
                    << " "
                    << cuArray2x3DeepCopy->get(i,
                                               j)
                    << std::endl;
        }
    }

/* Print ownership info. */
    std::cout
            << "cuArray2x3Copy owns host data: "
            << cuArray2x3CopyOwnsHostData
            << " cuArray2x3DeepCopy owns host data: "
            << cuArray2x3DeepCopyOwnsHostData
            << std::endl;

    delete cuArray2x3Copy;
    delete cuArray2x3DeepCopy;
    delete cuArray;
    return 0;
}