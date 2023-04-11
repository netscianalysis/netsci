import pytest
import cuarray
import numpy as np


@pytest.fixture(scope='session')
def FloatCuArrayNoDataFixture():
    m, n = 20, 100
    floatCuArrayNoData = cuarray.FloatCuArray()
    floatCuArrayNoData.init(m, n)
    return floatCuArrayNoData


@pytest.fixture(scope='session')
def FloatCuArrayWithDataFixture():
    m, n = 20, 100
    floatCuArrayWithData = cuarray.FloatCuArray()
    floatCuArrayWithData.fromNumpy2D(
        np.random.random((m, n)).astype(np.float32)
    )
    return floatCuArrayWithData
