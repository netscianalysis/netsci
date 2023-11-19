#include "cuarray.h"

int main() {
/* Creates a new float CuArray instance */
    CuArray<float> *cuArray = new CuArray<float>();

    delete cuArray;

    return 0;
}