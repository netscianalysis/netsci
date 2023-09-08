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
                    networkStride1.nodeCoordinates()->get(i,
                                                          2 * j),
                    networkStride2.nodeCoordinates()->get(i,
                                                          j)
            );
        }
    }
}

TEST_F(
        NetworkTest,
        NetworkStrideCoordinateEqualityStride1Stride3
) {
    for (int i = 0; i < 290; i++) {
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      0),
                networkStride3.nodeCoordinates()->get(i,
                                                      0)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      10),
                networkStride3.nodeCoordinates()->get(i,
                                                      4)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      20),
                networkStride3.nodeCoordinates()->get(i,
                                                      8)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      3),
                networkStride3.nodeCoordinates()->get(i,
                                                      1)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      13),
                networkStride3.nodeCoordinates()->get(i,
                                                      5)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      23),
                networkStride3.nodeCoordinates()->get(i,
                                                      9)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      6),
                networkStride3.nodeCoordinates()->get(i,
                                                      2)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      16),
                networkStride3.nodeCoordinates()->get(i,
                                                      6)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      26),
                networkStride3.nodeCoordinates()->get(i,
                                                      10)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      9),
                networkStride3.nodeCoordinates()->get(i,
                                                      3)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      19),
                networkStride3.nodeCoordinates()->get(i,
                                                      7)
        );
        EXPECT_EQ(
                networkStride1.nodeCoordinates()->get(i,
                                                      29),
                networkStride3.nodeCoordinates()->get(i,
                                                      11)
        );
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