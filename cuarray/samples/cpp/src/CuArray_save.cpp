#include "cuarray.h"
#include <iostream>

#define NETSCI_ROOT_DIR "."

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Create a new double CuArray instance that will have 10 rows and 10
* columns*/
    CuArray<float> *cuArray = new CuArray<float>();
    cuArray->init(10,
                  10
    );

/* Fill the CuArray with random values. */
    for (int i = 0; i < cuArray->m(); i++) {
        for (int j = 0; j < cuArray->n(); j++) {
            float val = static_cast <float> (rand()) /
                        static_cast <float> (RAND_MAX);
            cuArray->set(val,
                         i,
                         j);
        }
    }

/* Save the CuArray to a .npy file. */
    auto npyFname = NETSCI_ROOT_DIR "/tmp.npy";
    cuArray->save(npyFname);

/* Create a new CuArray instance from the .npy file. */
    auto cuArrayFromNpy = new CuArray<float>();
    cuArrayFromNpy->load(npyFname);

/*Print (i, j) elements of the CuArray's next to each other.
 * and check for equality*/
    for (int i = 0; i < cuArray->m(); i++) {
        for (int j = 0; j < cuArray->n(); j++) {
            auto val1 = cuArray->get(i,
                                     j);
            auto val2 = cuArrayFromNpy->get(i,
                                            j);
            bool equal = val1 == val2;
            std::cout
                    << val1
                    << " "
                    << val2
                    << " "
                    << equal
                    << std::endl;
            if (!equal) {
                std::cout
                        << "Values at ("
                        << i
                        << ", "
                        << j
                        << ") are not equal."
                        << std::endl;
                return 1;
            }


        }
    }
    delete cuArray;
    return 0;
}