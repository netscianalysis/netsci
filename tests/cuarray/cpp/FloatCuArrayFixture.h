//
// Created by andy on 4/4/23.
//

#ifndef NETSCI_FLOATCUARRAYFIXTURE_H
#define NETSCI_FLOATCUARRAYFIXTURE_H

#include "gtest/gtest.h"
#include "cuarray.h"

class FloatCuArrayFixture : public ::testing::Test {
protected:
    void SetUp() override {
        data = new float[2000];
        for (int i = 0; i < 2000; i++) {
            data[i] = static_cast<float>(rand()) /
                      static_cast<float>(RAND_MAX);
        }
        cuArrayNoData = new CuArray<float>;
        cuArrayNoData->init(
                20,
                100
        );
        cuArrayWithData = new CuArray<float>;
        cuArrayWithData->init(
                data,
                20,
                100
        );
    }

    void TearDown() override {
        delete cuArrayNoData;
        delete cuArrayWithData;
        delete[] data;
    }

    CuArray<float> *cuArrayNoData;
    CuArray<float> *cuArrayWithData;
    float *data;
};

#endif //NETSCI_FLOATCUARRAYFIXTURE_H
