#include "cuarray.h"
#include <iostream>
#include <random>

int main() {
    std::cout
            << "Running "
            << __FILE__
            << std::endl;

/* Create a new double CuArray instance. We're using a double vs. float
 * here because the numpy array is a float64 array. If you tried
 * to load this file into a CuArray<float> it would cause a
 * segmentation fault.*/
    CuArray<double> *cuArray = new CuArray<double>();

/*
 * Load a serialized numpy array with 2000 elements from the C++ test data directory.
 * NETSCI_ROOT_DIR, used here, is defined in CMakeLists. Ignore warnings in IDEs
 * about it being undefined; it's a known issue and does not affect functionality.
 */
    auto npyFname = NETSCI_ROOT_DIR
            "/tests/netcalc/cpp/data/2X_1D_1000_4.npy";
    cuArray->load(npyFname);

/* Print the CuArray. */
    for (int i = 0; i < cuArray->m(); i++) {
        for (int j = 0; j < cuArray->n(); j++) {
            std::cout
                    << cuArray->get(i,
                                    j)
                    << std::endl;
        }
    }

/* Free the CuArray. */
    delete cuArray;
    return 0;
}