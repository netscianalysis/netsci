//
// Created by andy on 4/5/23.
//
#include <gtest/gtest.h>
#include <cmath>
#include "network.h"

class NetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        networkStride1.init(
                "data/test.dcd",
                "data/test.pdb",
                0,
                9,
                1
        );
    }

    Network networkStride1;
};

TEST_F(
        NetworkTest,
        NetworkStride1_numFrames
) {
    EXPECT_EQ(
            10,
            networkStride1.numFrames()
    );
}

TEST_F(
        NetworkTest,
        NetworkStride1_nodeCoordinates_m
) {
    EXPECT_EQ(
            290,
            networkStride1.nodeCoordinates()->m()
    );
}

TEST_F(
        NetworkTest,
        NetworkStride1_nodeCoordinates_n
) {
    EXPECT_EQ(
            30,
            networkStride1.nodeCoordinates()->n()
    );
}



int main(
        int argc,
        char **argv
) {
    ::testing::InitGoogleTest(&argc,
                              argv);
    return RUN_ALL_TESTS();
}







