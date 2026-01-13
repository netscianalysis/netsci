#include <ut.hpp>

#include "network.h"

using namespace boost::ut;

// Replacement for the Google Test fixture
struct NetworkStride2Context {
    NetworkStride2Context() {
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

int main() {

    test("NetworkStride2_numFrames") = [] {
        NetworkStride2Context ctx;
        expect(ctx.networkStride2.numFrames() == 5);
    };

    test("NetworkStride2_nodeCoordinates_m") = [] {
        NetworkStride2Context ctx;
        expect(ctx.networkStride2.nodeCoordinates()->m() == 290);
    };

    test("NetworkStride2_nodeCoordinates_n") = [] {
        NetworkStride2Context ctx;
        expect(ctx.networkStride2.nodeCoordinates()->n() == 15);
    };

    return 0;
}
