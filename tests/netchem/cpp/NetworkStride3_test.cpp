#include <ut.hpp>
#include <cmath>

#include "network.h"

using namespace boost::ut;

// Replacement for the Google Test fixture
struct NetworkStride3Context {
    NetworkStride3Context() {
        networkStride3.init(
            "data/test.dcd",
            "data/test.pdb",
            0,
            9,
            3
        );
    }

    Network networkStride3;
};

int main() {

    test("NetworkStride3_numFrames") = [] {
        NetworkStride3Context ctx;
        expect(ctx.networkStride3.numFrames() == 4);
    };

    test("NetworkStride3_nodeCoordinates_m") = [] {
        NetworkStride3Context ctx;
        expect(ctx.networkStride3.nodeCoordinates()->m() == 290);
    };

    test("NetworkStride3_nodeCoordinates_n") = [] {
        NetworkStride3Context ctx;
        expect(ctx.networkStride3.nodeCoordinates()->n() == 12);
    };

    return 0;
}

