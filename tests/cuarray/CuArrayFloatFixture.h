//
// Created by andy on 4/4/23.
//

#ifndef NETSCI_CUARRAYFLOATFIXTURE_H
#define NETSCI_CUARRAYFLOATFIXTURE_H

#include "gtest/gtest.h"
#include "cuarray.h"

class CuArrayFloatTest : public ::testing::Test {
protected:
    void SetUp() override {
        data = new float[2000];
        for (int i = 0; i < 2000; i++) {
            data[i] = static_cast<float>(rand()) /
                      static_cast<float>(RAND_MAX);
        }
        cuArrayNoData = new CuArray<float>(
                20,
                100
        );
        cuArrayWithData = new CuArray<float>(
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

#endif //NETSCI_CUARRAYFLOATFIXTURE_H
