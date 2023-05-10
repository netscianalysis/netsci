import cuarray


def test_FloatCuArrayNoData_m(FloatCuArrayNoDataFixture):
    assert FloatCuArrayNoDataFixture.m() == 20


def test_FloatCuArrayNoData_n(FloatCuArrayNoDataFixture):
    assert FloatCuArrayNoDataFixture.n() == 100


def test_FloatCuArrayNoData_bytes(FloatCuArrayNoDataFixture):
    assert FloatCuArrayNoDataFixture.bytes() == 8000


def test_FloatCuArrayNoData_size(FloatCuArrayNoDataFixture):
    assert FloatCuArrayNoDataFixture.size() == 2000


def test_FloatCuArrayNoData_allocatedHost(FloatCuArrayNoDataFixture):
    assert FloatCuArrayNoDataFixture.allocatedHost() == 1


def test_FloatCuArrayNoData_allocatedDevice(FloatCuArrayNoDataFixture):
    assert FloatCuArrayNoDataFixture.allocatedDevice() == 0


def test_FloatCuArrayNoData_host(FloatCuArrayNoDataFixture):
    for i in range(FloatCuArrayNoDataFixture.m()):
        for j in range(FloatCuArrayNoDataFixture.n()):
            assert FloatCuArrayNoDataFixture.get(i, j) == 0.0


def test_FloatCuArrayNoData_toDeviceNegative(FloatCuArrayNoDataFixture):
    FloatCuArrayNoDataFixture.toDevice()
    assert FloatCuArrayNoDataFixture.allocatedDevice() == 0
    FloatCuArrayNoDataFixture.deallocateDevice()


def test_FloatCuArrayNoData_toDevicePositive(FloatCuArrayNoDataFixture):
    FloatCuArrayNoDataFixture.allocateDevice()
    FloatCuArrayNoDataFixture.toDevice()
    assert FloatCuArrayNoDataFixture.allocatedDevice() == 1
    FloatCuArrayNoDataFixture.deallocateDevice()


def test_FloatCuArray_fromCuArray(FloatCuArrayWithDataFixture):
    floatCuArray = cuarray.FloatCuArray()
    floatCuArray.fromCuArray(
        cuArray=FloatCuArrayWithDataFixture,
        start=18,
        end=19,
        m=2,
        n=100,
    )
    assert floatCuArray.m() == 2
    assert floatCuArray.n() == 100
    assert floatCuArray.bytes() == 800
    assert floatCuArray.size() == 200
    assert floatCuArray.allocatedHost() == 1
    assert floatCuArray.allocatedDevice() == 0
    assert floatCuArray.owner() == 1
    for i in range(floatCuArray.m()):
        for j in range(floatCuArray.n()):
            assert floatCuArray.get(i, j) == FloatCuArrayWithDataFixture.get(
                18 + i, j
            )
