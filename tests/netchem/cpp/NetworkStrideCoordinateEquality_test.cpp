#include <ut.hpp>

#include "network.h"

using namespace boost::ut;

// Replacement for the Google Test fixture
struct NetworkStrideContext {
    NetworkStrideContext() {
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

int main() {

    test("NetworkStrideCoordinateEqualityStride1Stride2") = [] {
        NetworkStrideContext ctx;

        for (int i = 0; i < 290; i++) {
            for (int j = 0; j < 15; j++) {
                expect(
                    ctx.networkStride1.nodeCoordinates()->get(i, 2 * j) ==
                    ctx.networkStride2.nodeCoordinates()->get(i, j)
                );
            }
        }
    };

    test("NetworkStrideCoordinateEqualityStride1Stride3") = [] {
        NetworkStrideContext ctx;

        for (int i = 0; i < 290; i++) {
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 0) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 0)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 10) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 4)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 20) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 8)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 3) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 1)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 13) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 5)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 23) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 9)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 6) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 2)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 16) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 6)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 26) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 10)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 9) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 3)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 19) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 7)
            );
            expect(
                ctx.networkStride1.nodeCoordinates()->get(i, 29) ==
                ctx.networkStride3.nodeCoordinates()->get(i, 11)
            );
        }
    };

    return 0;
}
