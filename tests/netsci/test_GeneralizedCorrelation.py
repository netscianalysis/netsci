import pytest
import numpy as np
import cuarray
import netsci


def test_GeneralizedCorrelation_2X1D_1000n4k_GpuCpu():
    n = 1000
    k = 4
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.init(1, n)
    Xb.init(1, n)
    x = 0.001
    for i in range(n):
        Xa.at(float(np.sin(x)), 0, i)
        Xb.at(
            float(np.cos(x)),
            0, i
        )
        x += float(np.pi / n)
    gpuGeneralizedCorrelation = netsci.gpuGeneralizedCorrelation2X1D(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
    )
    cpuGeneralizedCorrelation = netsci.cpuGeneralizedCorrelation2X1D(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
    )
    assert gpuGeneralizedCorrelation == cpuGeneralizedCorrelation


def test_GeneralizedCorrelation_2X1D_2000n4k_GpuCpu():
    n = 2000
    k = 4
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.init(1, n)
    Xb.init(1, n)
    x = 0.001
    for i in range(n):
        Xa.at(float(np.sin(x)), 0, i)
        Xb.at(
            float(np.cos(x)),
            0, i
        )
        x += float(np.pi / n)
    gpuGeneralizedCorrelation = netsci.gpuGeneralizedCorrelation2X1D(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
    )
    cpuGeneralizedCorrelation = netsci.cpuGeneralizedCorrelation2X1D(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
    )
    assert gpuGeneralizedCorrelation == cpuGeneralizedCorrelation


def test_GeneralizedCorrelation_2X2D_1000n4k_GpuCpu():
    n = 1000
    k = 4
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.init(2, n)
    Xb.init(2, n)
    x = 0.001
    for i in range(n):
        Xa.at(float(np.sin(x)), 0, i)
        Xa.at(float(x), 1, i)
        Xb.at(
            float(np.cos(x)),
            0, i
        )
        Xb.at(float(2 * x), 1, i)
        x += float(np.pi / n)
    gpuGeneralizedCorrelation = netsci.gpuGeneralizedCorrelation2X2D(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
    )
    cpuGeneralizedCorrelation = netsci.cpuGeneralizedCorrelation2X2D(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
    )
    assert gpuGeneralizedCorrelation == cpuGeneralizedCorrelation


def test_GeneralizedCorrelation_2X2D_2000n4k_GpuCpu():
    n = 2000
    k = 4
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.init(2, n)
    Xb.init(2, n)
    x = 0.001
    for i in range(n):
        Xa.at(float(np.sin(x)), 0, i)
        Xa.at(float(x), 1, i)
        Xb.at(
            float(np.cos(x)),
            0, i
        )
        Xb.at(float(2 * x), 1, i)
        x += float(np.pi / n)
    gpuGeneralizedCorrelation = netsci.gpuGeneralizedCorrelation2X2D(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
    )
    cpuGeneralizedCorrelation = netsci.cpuGeneralizedCorrelation2X2D(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
    )
    assert gpuGeneralizedCorrelation == cpuGeneralizedCorrelation
