#include <cuarray.h>
#include <iostream>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Creates a new float CuArray instance */
    CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
    auto rows = 300;
    auto cols = 300;
    cuArray->init(rows,
                  cols);

/* Fill the CuArray with random values */
    for (int i = 0; i < cuArray->m(); i++) {
        for (int j = 0; j < cuArray->n(); j++) {
            cuArray->host()[i * cuArray->n() + j] =
                    static_cast<float>(rand() / (float) RAND_MAX);
        }
    }

/* Create a new CuArray with indices that sort the 8th row
 * of the original CuArray.*/
    auto cuArrayRowIndex = 7;
    auto sortedIndicesCuArray = cuArray->argsort(cuArrayRowIndex);

/* Create a new CuArray containing sorted data from the 8th row
 * of the original CuArray.*/
    auto sortedCuArray = cuArray->sort(cuArrayRowIndex);

/* Print the sorted CuArray and the corresponding values from the
 * original CuArray using the sortedIndicesCuArray.*/
    for (int j = 0; j < sortedCuArray->n(); j++) {
        auto sortedIndex = sortedIndicesCuArray->get(0,
                                                     j);
        auto sortedValue = sortedCuArray->get(0,
                                              j);
        auto sortedValueFromOriginalCuArray =
                cuArray->get(sortedIndex,
                             cuArrayRowIndex);
        std::cout
                << sortedIndex
                << " "
                << sortedValue
                << " "
                << sortedValueFromOriginalCuArray
                << std::endl;
    }

/* Cleanup time. */
    delete cuArray;
    delete sortedCuArray;
    delete sortedIndicesCuArray;
    return 0;
}