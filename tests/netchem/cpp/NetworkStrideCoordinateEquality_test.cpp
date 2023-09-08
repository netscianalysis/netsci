//
// Created by astokely on 9/7/23.
//
#include <gtest/gtest.h>
#include "network.h"

class NetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        networkStride2.init(
                "data/test.dcd",
                "data/test.pdb",
                0,
                9,
                2
        );
        networkStride1.init(
                "data/test.dcd",
                "data/test.pdb",
                0,
                9,
                1
        );
        networkStride3.init(
                "data/test.dcd",
                "data/test.pdb",
                0,
                9,
                3
        );
    }


    Network networkStride3;
    Network networkStride2;
    Network networkStride1;

};

TEST_F(
        NetworkTest,
        NetworkStrideCoordinateEqualityStride1Stride2
) {
    for (int i = 0; i < 290; i++) {
        for (int j = 0; j < 15; j++) {
            EXPECT_EQ(
                    networkStride1.nodeCoordinates()->get(i, 2*j),
                    networkStride2.nodeCoordinates()->get(i, j)
            );
        }
    }
}

TEST_F(
        NetworkTest,
        NetworkStrideCoordinateEqualityStride1Stride3
) {
    for (int i = 0; i < 290; i++) {
        for (int j = 0; j < 9; j++) {
            EXPECT_EQ(
                    networkStride1.nodeCoordinates()->get(i, 3*j),
                    networkStride3.nodeCoordinates()->get(i, j)
            );
        }
    }
}


int main(
        int argc,
        char **argv
) {
    ::testing::InitGoogleTest(&argc,
                              argv);
    return RUN_ALL_TESTS();
}