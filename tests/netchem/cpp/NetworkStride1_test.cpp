#include <ut.hpp>
#include <cmath>

#include "network.h"

using namespace boost::ut;

// Replacement for the Google Test fixture
struct NetworkContext {
    NetworkContext() {
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

int main() {

    test("NetworkStride1_numFrames") = [] {
        NetworkContext ctx;
        expect(ctx.networkStride1.numFrames() == 10);
    };

    test("NetworkStride1_nodeCoordinates_m") = [] {
        NetworkContext ctx;
        expect(ctx.networkStride1.nodeCoordinates()->m() == 290);
    };

    test("NetworkStride1_nodeCoordinates_n") = [] {
        NetworkContext ctx;
        expect(ctx.networkStride1.nodeCoordinates()->n() == 30);
    };

    return 0;
}
