
#include <gtest/gtest.h>

#include "CuArrayFloatFixture.h"


TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoData_m
) {
    EXPECT_EQ(cuArrayNoData->m(), 20);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoData_n
) {
    EXPECT_EQ(cuArrayNoData->n(), 100);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoData_bytes
) {
    EXPECT_EQ(cuArrayNoData->bytes(), 2000 * sizeof(float));
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoData_size
) {
    EXPECT_EQ(cuArrayNoData->size(), 2000);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoData_allocatedHost
) {
    EXPECT_EQ(cuArrayNoData->allocatedHost(), 1);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoData_allocatedDevice
) {
    EXPECT_EQ(cuArrayNoData->allocatedDevice(), 0);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoData_host
) {
    for (int i = 0; i < 2000; i++) {
        EXPECT_EQ(cuArrayNoData->host()[i], 0);
    }
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoData_allocateDevice
) {
    cuArrayNoData->allocateDevice();
    EXPECT_EQ(cuArrayNoData->allocatedDevice(), 1);
    cuArrayNoData->deallocateDevice();
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoDataTest_toDeviceNegative
) {
    cuArrayNoData->toDevice();
    EXPECT_EQ(cuArrayNoData->allocatedDevice(), 0);
    cuArrayNoData->deallocateDevice();
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoDataTest_toDevicePositive
) {
    cuArrayNoData->allocateDevice();
    cuArrayNoData->toDevice();
    EXPECT_EQ(cuArrayNoData->allocatedDevice(), 1);
    cuArrayNoData->deallocateDevice();
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoDataTest_toHostNegative
) {
    cuArrayNoData->toHost();
    EXPECT_EQ(cuArrayNoData->allocatedHost(), 0);
    cuArrayNoData->deallocateHost();
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatNoDataTest_toHostPositive
) {
    cuArrayNoData->allocateHost();
    cuArrayNoData->toHost();
    EXPECT_EQ(cuArrayNoData->allocatedHost(), 1);
    cuArrayNoData->deallocateHost();
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithData_m
) {
    EXPECT_EQ(cuArrayWithData->m(), 20);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithData_n
) {
    EXPECT_EQ(cuArrayWithData->n(), 100);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithData_bytes
) {
    EXPECT_EQ(cuArrayWithData->bytes(), 2000 * sizeof(float));
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithData_size
) {
    EXPECT_EQ(cuArrayWithData->size(), 2000);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithData_allocatedHost
) {
    EXPECT_EQ(cuArrayWithData->allocatedHost(), 1);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithData_allocatedDevice
) {
    EXPECT_EQ(cuArrayWithData->allocatedDevice(), 0);
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithData_host
) {
    for (int i = 0; i < 2000; i++) {
        EXPECT_EQ(cuArrayWithData->host()[i], data[i]);
    }
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithData_at
) {
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 100; j++) {
            EXPECT_EQ(cuArrayWithData->at(i, j), data[i * 100 + j]);
        }
    }
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithData_allocateDevice
) {
    cuArrayWithData->allocateDevice();
    EXPECT_EQ(cuArrayWithData->allocatedDevice(), 1);
    cuArrayWithData->deallocateDevice();
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithDataTest_toDeviceNegative
) {
    cuArrayWithData->toDevice();
    EXPECT_EQ(cuArrayWithData->allocatedDevice(), 0);
    cuArrayWithData->deallocateDevice();
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithDataTest_toDevicePositive
) {
    cuArrayWithData->allocateDevice();
    cuArrayWithData->toDevice();
    EXPECT_EQ(cuArrayWithData->allocatedDevice(), 1);
    cuArrayWithData->deallocateDevice();
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithDataTest_toHostNegative
) {
    cuArrayWithData->toHost();
    EXPECT_EQ(cuArrayWithData->allocatedHost(), 0);
    cuArrayWithData->deallocateHost();
}

TEST_F(
        CuArrayFloatTest,
        CuArrayFloatWithDataTest_toHostPositive
) {
    cuArrayWithData->allocateHost();
    cuArrayWithData->toHost();
    EXPECT_EQ(cuArrayWithData->allocatedHost(), 1);
    cuArrayWithData->deallocateHost();
}

int main(
        int argc,
        char **argv
) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
