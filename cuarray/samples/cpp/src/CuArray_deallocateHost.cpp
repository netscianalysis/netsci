#include <cuarray.h>
#include <random>

int main() {

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
/* Allocate device memory. */
    cuArray->allocateDevice();

/* Copy data from host to device. */
    cuArray->toDevice();

/* Deallocate the host array to reduce memory usage if it's not needed again. */

    cuArray->deallocateHost();

/* Set the number of threads per block to 1024 */
    auto threadsPerBlock = 1024;

/* Set the number of blocks to the ceiling of the number of elements
 * divided by the number of threads per block. */
    auto blocksPerGrid =
            (cuArray->size() + threadsPerBlock - 1) / threadsPerBlock;

/* Launch a CUDA kernel that does something cool and only takes
 * a single float array as an argument
 *<<<blocksPerGrid, threadsPerBlock>>>kernel(cuArray->device()); */

/* Free device memory. */
    delete cuArray;
    return 0;
}