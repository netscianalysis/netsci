import pytest
import numpy as np
import cuarray
import netcalc 


def test_MutualInformation2X1D_1000n4k09covGaussian_GpuCpu():
    n = 1000
    k = 4
    X = np.load("../../data/2X_1D_1000_4.npy").astype(np.float32)
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.fromNumpy1D(X[:n])
    Xb.fromNumpy1D(X[n:])
    mutualInformationGpu = netcalc.mutualInformationGpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=1,
    )
    mutualInformationCpu = netcalc.mutualInformationCpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=1,
    )
    assert mutualInformationGpu == mutualInformationCpu == 0.8449974060058594


def test_MutualInformation2X1D_2000n4k09covGaussian_GpuCpu():
    n = 2000
    k = 4
    X = np.load("../../data/2X_1D_2000_4.npy").astype(np.float32)
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.fromNumpy1D(X[:n])
    Xb.fromNumpy1D(X[n:])
    mutualInformationGpu = netcalc.mutualInformationGpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=1,
    )
    mutualInformationCpu = netcalc.mutualInformationCpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=1,
    )
    assert mutualInformationGpu == mutualInformationCpu == 0.8485465049743652


def test_MutualInformation2X2D_1000n4k_GpuCpu():
    n = 1000
    k = 4
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.init(2, n)
    Xb.init(2, n)
    x = 0.001
    for i in range(n):
        Xa.set(
            float(np.sin(x)),
            0, i,
        )
        Xa.set(
            float(np.cos(x)),
            1, i,
        )
        Xb.set(
            float(x),
            0, i,
        )
        Xb.set(
            float(2 * x),
            1, i,
        )
        x += float(np.pi / n)
    mutualInformationGpu = netcalc.mutualInformationGpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=2,
    )
    mutualInformationCpu = netcalc.mutualInformationCpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=2,
    )
    assert mutualInformationGpu == mutualInformationCpu


def test_MutualInformation2X2D_2000n4k_GpuCpu():
    n = 2000
    k = 4
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.init(2, n)
    Xb.init(2, n)
    x = 0.001
    for i in range(n):
        Xa.set(
            float(np.sin(x)),
            0, i,
        )
        Xa.set(
            float(np.cos(x)),
            1, i,
        )
        Xb.set(
            float(x),
            0, i,
        )
        Xb.set(
            float(2 * x),
            1, i,
        )
        x += float(np.pi / n)
    mutualInformationGpu = netcalc.mutualInformationGpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=2,
    )
    mutualInformationCpu = netcalc.mutualInformationCpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=2,
    )
    assert mutualInformationGpu == mutualInformationCpu
