#include <cstdlib>
#include <ut.hpp>

#include "cuarray.h"

using namespace boost::ut;

struct FloatCuArrayContext {
    FloatCuArrayContext() {
        data = new float[2000];
        for (int i = 0; i < 2000; i++) {
            data[i] = static_cast<float>(rand()) /
                      static_cast<float>(RAND_MAX);
        }

        cuArrayNoData = new CuArray<float>;
        cuArrayNoData->init(20, 100);

        cuArrayWithData = new CuArray<float>;
        cuArrayWithData->init(data, 20, 100);
    }

    ~FloatCuArrayContext() {
        delete cuArrayNoData;
        delete cuArrayWithData;
        delete[] data;
    }

    CuArray<float> *cuArrayNoData{};
    CuArray<float> *cuArrayWithData{};
    float *data{};
};

int main() {

    test("FloatCuArrayNoData_m") = [] {
        FloatCuArrayContext ctx;
        expect(ctx.cuArrayNoData->m() == 20);
    };

    test("FloatCuArrayNoData_n") = [] {
        FloatCuArrayContext ctx;
        expect(ctx.cuArrayNoData->n() == 100);
    };

    test("FloatCuArrayNoData_bytes") = [] {
        FloatCuArrayContext ctx;
        expect(ctx.cuArrayNoData->bytes() == 2000 * sizeof(float));
    };

    test("FloatCuArrayNoData_size") = [] {
        FloatCuArrayContext ctx;
        expect(ctx.cuArrayNoData->size() == 2000);
    };

    test("FloatCuArrayNoData_allocatedHost") = [] {
        FloatCuArrayContext ctx;
        expect(ctx.cuArrayNoData->allocatedHost() == 1);
    };

    test("FloatCuArrayNoData_allocatedDevice") = [] {
        FloatCuArrayContext ctx;
        expect(ctx.cuArrayNoData->allocatedDevice() == 0);
    };

    test("FloatCuArrayNoData_host") = [] {
        FloatCuArrayContext ctx;
        for (int i = 0; i < 2000; i++) {
            expect(ctx.cuArrayNoData->host()[i] == 0.0f);
        }
    };

    test("FloatCuArrayNoData_allocateDevice") = [] {
        FloatCuArrayContext ctx;
        ctx.cuArrayNoData->allocateDevice();
        expect(ctx.cuArrayNoData->allocatedDevice() == 1);
        ctx.cuArrayNoData->deallocateDevice();
    };

    test("FloatCuArrayNoData_toDeviceNegative") = [] {
        FloatCuArrayContext ctx;
        ctx.cuArrayNoData->toDevice();
        expect(ctx.cuArrayNoData->allocatedDevice() == 0);
    };

    test("FloatCuArrayNoData_toDevicePositive") = [] {
        FloatCuArrayContext ctx;
        ctx.cuArrayNoData->allocateDevice();
        ctx.cuArrayNoData->toDevice();
        expect(ctx.cuArrayNoData->allocatedDevice() == 1);
        ctx.cuArrayNoData->deallocateDevice();
    };

    test("FloatCuArrayNoData_toHostNegative") = [] {
        FloatCuArrayContext ctx;

        expect(ctx.cuArrayNoData->allocatedDevice() == 0);
        expect(ctx.cuArrayNoData->allocatedHost() == 1);

        expect(ctx.cuArrayNoData->toHost() != 0);   // failure is the contract
        expect(ctx.cuArrayNoData->allocatedHost() == 1); // unchanged
    };


    test("FloatCuArrayNoData_toHostPositive") = [] {
        FloatCuArrayContext ctx;
        ctx.cuArrayNoData->allocateHost();
        ctx.cuArrayNoData->toHost();
        expect(ctx.cuArrayNoData->allocatedHost() == 1);
        ctx.cuArrayNoData->deallocateHost();
    };

    test("FloatCuArrayWithData_host") = [] {
        FloatCuArrayContext ctx;
        for (int i = 0; i < 2000; i++) {
            expect(ctx.cuArrayWithData->host()[i] == ctx.data[i]);
        }
    };

    test("FloatCuArrayWithData_get") = [] {
        FloatCuArrayContext ctx;
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 100; j++) {
                expect(ctx.cuArrayWithData->get(i, j) == ctx.data[i * 100 + j]);
            }
        }
    };

    test("FloatCuArray_fromCuArrayShallowCopy") = [] {
        FloatCuArrayContext ctx;

        auto *shallow = new CuArray<float>;
        shallow->fromCuArrayShallowCopy(
                ctx.cuArrayWithData, 19, 19, 1, 100);

        expect(shallow->m() == 1);
        expect(shallow->n() == 100);
        expect(shallow->bytes() == 100 * sizeof(float));
        expect(shallow->size() == 100);
        expect(shallow->allocatedHost() == 1);
        expect(shallow->allocatedDevice() == 0);
        expect(shallow->owner() == 0);

        for (int i = 0; i < 100; i++) {
            expect(shallow->get(0, i) == ctx.cuArrayWithData->get(19, i));
        }

        delete shallow;

        for (int i = 0; i < 100; i++) {
            ctx.cuArrayWithData->get(19, i);
        }
    };

    test("FloatCuArray_fromCuArrayDeepCopy") = [] {
        FloatCuArrayContext ctx;

        auto *deep = new CuArray<float>;
        deep->fromCuArrayDeepCopy(
                ctx.cuArrayWithData, 19, 19, 1, 100);

        expect(deep->m() == 1);
        expect(deep->n() == 100);
        expect(deep->bytes() == 100 * sizeof(float));
        expect(deep->size() == 100);
        expect(deep->allocatedHost() == 1);
        expect(deep->allocatedDevice() == 0);
        expect(deep->owner() == 1);

        for (int i = 0; i < 100; i++) {
            expect(deep->get(0, i) == ctx.cuArrayWithData->get(19, i));
        }

        delete deep;

        for (int i = 0; i < 100; i++) {
            ctx.cuArrayWithData->get(19, i);
        }
    };
    return 0;
};
