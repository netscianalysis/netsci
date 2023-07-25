import numpy as np
import cuarray
import netcalc 


def test_GeneralizedCorrelation_2X1D_1000n4k_GpuCpu():
    n = 1000
    k = 4
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.init(1, n)
    Xb.init(1, n)
    x = 0.001
    for i in range(n):
        Xa.set(
            float(np.sin(x)),
            0, i,
        )
        Xb.set(
            float(np.cos(x)),
            0, i,
        )
        x += float(np.pi / n)
    generalizedCorrelationGpu = netcalc.generalizedCorrelationGpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=1,
    )
    generalizedCorrelationCpu = netcalc.generalizedCorrelationCpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=1,
    )
    assert generalizedCorrelationGpu == generalizedCorrelationCpu


def test_GeneralizedCorrelation_2X1D_2000n4k_GpuCpu():
    n = 2000
    k = 4
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.init(1, n)
    Xb.init(1, n)
    x = 0.001
    for i in range(n):
        Xa.set(
            float(np.sin(x)),
            0, i,
        )
        Xb.set(
            float(np.cos(x)),
            0, i,
        )
        x += float(np.pi / n)
    generalizedCorrelationGpu = netcalc.generalizedCorrelationGpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=1,
    )
    generalizedCorrelationCpu = netcalc.generalizedCorrelationCpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=1,
    )
    assert generalizedCorrelationGpu == generalizedCorrelationCpu


def test_GeneralizedCorrelation_2X2D_1000n4k_GpuCpu():
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
            float(x),
            1, i,
        )
        Xb.set(
            float(np.cos(x)),
            0, i,
        )
        Xb.set(
            float(2 * x),
            1, i,
        )
        x += float(np.pi / n)
    generalizedCorrelationGpu = netcalc.generalizedCorrelationGpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=2,
    )
    generalizedCorrelationCpu = netcalc.generalizedCorrelationCpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=2,
    )
    assert generalizedCorrelationGpu == generalizedCorrelationCpu


def test_GeneralizedCorrelation_2X2D_2000n4k_GpuCpu():
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
            float(x),
            1, i,
        )
        Xb.set(
            float(np.cos(x)),
            0, i,
        )
        Xb.set(
            float(2 * x),
            1, i,
        )
        x += float(np.pi / n)
    generalizedCorrelationGpu = netcalc.generalizedCorrelationGpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=2,
    )
    generalizedCorrelationCpu = netcalc.generalizedCorrelationCpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=2,
    )
    assert generalizedCorrelationGpu == generalizedCorrelationCpu


def test_GeneralizedCorrelation_2X3D_2000n4k_GpuCpu():
    n = 2000
    k = 4
    Xa = cuarray.FloatCuArray()
    Xb = cuarray.FloatCuArray()
    Xa.init(3, n)
    Xb.init(3, n)
    x = 0.001
    for i in range(n):
        Xa.set(
            float(np.sin(x)),
            0, i,
        )
        Xa.set(
            float(x),
            1, i,
        )
        Xa.set(
            float(np.log(x)),
            2, i,
        )
        Xb.set(
            float(np.cos(x)),
            0, i,
        )
        Xb.set(
            float(2 * x),
            1, i,
        )
        Xb.set(
            float(np.log(2 * x)),
            2, i,
        )
        x += float(np.pi / n)
    generalizedCorrelationGpu = netcalc.generalizedCorrelationGpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=3,
    )
    generalizedCorrelationCpu = netcalc.generalizedCorrelationCpu(
        Xa=Xa,
        Xb=Xb,
        k=k,
        n=n,
        xd=2,
        d=3,
    )
    assert generalizedCorrelationGpu == generalizedCorrelationCpu


