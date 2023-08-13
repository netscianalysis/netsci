//
// Created by andy on 4/5/23.
//
#include <gtest/gtest.h>
#include <cmath>
#include "network.h"

class NetworkTest : public ::testing::Test {
protected:
    void SetUp() override {

        networkStride3.init(
                "data/test.dcd",
                "data/test.pdb",
                0,
                9,
                3
        );
    }

    Network networkStride1;
    Network networkStride2;
    Network networkStride3;

};

TEST_F(
        NetworkTest,
        NetworkStride3_numFrames
) {
    EXPECT_EQ(
            3,
            networkStride3.numFrames()
    );
}

TEST_F(
        NetworkTest,
        NetworkStride3_nodeCoordinates_m
) {
    EXPECT_EQ(
            290,
            networkStride3.nodeCoordinates()->m()
    );
}

TEST_F(
        NetworkTest,
        NetworkStride3_nodeCoordinates_n
) {
    EXPECT_EQ(
            9,
            networkStride3.nodeCoordinates()->n()
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







