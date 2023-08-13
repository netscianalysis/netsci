//
// Created by andy on 4/5/23.
//
#include <gtest/gtest.h>
#include <cmath>
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
    }

    Network networkStride2;

};

TEST_F(
        NetworkTest,
        NetworkStride2_numFrames
) {
    EXPECT_EQ(
            5,
            networkStride2.numFrames()
    );
}

TEST_F(
        NetworkTest,
        NetworkStride2_nodeCoordinates_m
) {
EXPECT_EQ(
        290,
        networkStride2.nodeCoordinates()->m()
);
}

TEST_F(
        NetworkTest,
        NetworkStride2_nodeCoordinates_n
) {
    EXPECT_EQ(
            15,
            networkStride2.nodeCoordinates()->n()
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







