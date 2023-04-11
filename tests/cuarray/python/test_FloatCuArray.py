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
            assert FloatCuArrayNoDataFixture.at(i, j) == 0.0


def test_FloatCuArrayNoData_toDeviceNegative(FloatCuArrayNoDataFixture):
    FloatCuArrayNoDataFixture.toDevice()
    assert FloatCuArrayNoDataFixture.allocatedDevice() == 0
    FloatCuArrayNoDataFixture.deallocateDevice()


def test_FloatCuArrayNoData_toDevocePositive(FloatCuArrayNoDataFixture):
    FloatCuArrayNoDataFixture.allocateDevice()
    FloatCuArrayNoDataFixture.toDevice()
    assert FloatCuArrayNoDataFixture.allocatedDevice() == 1
    FloatCuArrayNoDataFixture.deallocateDevice()
