import pytest
import numpy as np
import cuarray
import netcalc
import pytest


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_MutualInformation2X1D_1000n4k09covGaussian_GpuCpu():
    n = 1000
    k = 4
    Xref = np.load("../../data/2X_1D_1000_4.npy").astype(np.float32).reshape(2, n)
    X = cuarray.FloatCuArray()
    # Define all pair correlations that will be computed
    num_nodes = 2
    num_node_pairs = num_nodes ** 2
    ab = cuarray.IntCuArray()
    ab.init(num_node_pairs, 2)
    for i in range(num_nodes):
        for j in range(num_nodes):
            node_pair_index = i * num_nodes + j
            ab[node_pair_index][0] = i
            ab[node_pair_index][1] = j

    Igpu = cuarray.FloatCuArray()
    Icpu = cuarray.FloatCuArray()
    X.fromNumpy2D(Xref)
    netcalc.mutualInformation(
        X=X,
        I=Igpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=1,
        platform=netcalc.GPU_PLATFORM,
    )
    netcalc.mutualInformation(
        X=X,
        I=Icpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=1,
        platform=netcalc.CPU_PLATFORM,
    )
    ref = [5.4011, 0.8449, 0.8449, 5.4011]
    for _ in range(4):
        assert Igpu[0][_] == Icpu[0][_] == pytest.approx(ref[_], rel=1e-3)


def test_MutualInformation2X1D_2000n4k09covGaussian_GpuCpu():
    n = 2000
    k = 4
    Xref = np.load("../../data/2X_1D_2000_4.npy").astype(np.float32).reshape(2, n)
    X = cuarray.FloatCuArray()
    # Define all pair correlations that will be computed
    num_nodes = 2
    num_node_pairs = num_nodes ** 2
    ab = cuarray.IntCuArray()
    ab.init(num_node_pairs, 2)
    for i in range(num_nodes):
        for j in range(num_nodes):
            node_pair_index = i * num_nodes + j
            ab[node_pair_index][0] = i
            ab[node_pair_index][1] = j

    Igpu = cuarray.FloatCuArray()
    Icpu = cuarray.FloatCuArray()
    X.fromNumpy2D(Xref)
    netcalc.mutualInformation(
        X=X,
        I=Igpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=1,
        platform=netcalc.GPU_PLATFORM,
    )
    netcalc.mutualInformation(
        X=X,
        I=Icpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=1,
        platform=netcalc.CPU_PLATFORM,
    )
    ref = [6.0945634841918945, 0.8485465049743652, 0.8485465049743652,6.0945634841918945]
    for _ in range(2):
        assert Igpu[0][_+1] == Icpu[0][_+1] == pytest.approx(ref[_+1], rel=1e-3)




def test_MutualInformation2X2D_1000n4k_GpuCpu():
    n = 1000
    k = 4
    X = cuarray.FloatCuArray()
    X.init(2, 2*n)
    x = 0.001
    for i in range(n):
        X.set(
            float(np.sin(x)),
            0, i,
        )
        X.set(
            float(np.cos(x)),
            1, i,
        )
        X.set(
            float(x),
            0, i+n,
        )
        X.set(
            float(2 * x),
            1, i+n,
        )
        x += float(np.pi / n)
    Igpu = cuarray.FloatCuArray()
    Icpu = cuarray.FloatCuArray()
    num_nodes = 2
    num_node_pairs = num_nodes ** 2
    ab = cuarray.IntCuArray()
    ab.init(num_node_pairs, 2)
    for i in range(num_nodes):
        for j in range(num_nodes):
            node_pair_index = i * num_nodes + j
            ab[node_pair_index][0] = i
            ab[node_pair_index][1] = j
    netcalc.mutualInformation(
        X=X,
        I=Igpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=2,
        platform=netcalc.GPU_PLATFORM,
    )
    netcalc.mutualInformation(
        X=X,
        I=Icpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=2,
        platform=netcalc.CPU_PLATFORM,
    )
    for _ in range(4):
        assert Igpu[0][_] == Icpu[0][_]


def test_MutualInformation2X2D_2000n4k_GpuCpu():
    n = 2000
    k = 4
    X = cuarray.FloatCuArray()
    Igpu = cuarray.FloatCuArray()
    Icpu = cuarray.FloatCuArray()
    X.init(2, 2*n)
    x = 0.001
    for i in range(n):
        X.set(
            float(np.sin(x)),
            0, i,
        )
        X.set(
            float(np.cos(x)),
            1, i,
        )
        X.set(
            float(x),
            0, i + n,
        )
        X.set(
            float(2 * x),
            1, i + n,
        )
        x += float(np.pi / n)
    num_nodes = 2
    num_node_pairs = num_nodes ** 2
    ab = cuarray.IntCuArray()
    ab.init(num_node_pairs, 2)
    for i in range(num_nodes):
        for j in range(num_nodes):
            node_pair_index = i * num_nodes + j
            ab[node_pair_index][0] = i
            ab[node_pair_index][1] = j
    netcalc.mutualInformation(
        X=X,
        I=Igpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=2,
        platform=netcalc.GPU_PLATFORM,
    )
    netcalc.mutualInformation(
        X=X,
        I=Icpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=2,
        platform=netcalc.CPU_PLATFORM,
    )
    for _ in range(4):
        assert Igpu[0][_] == Icpu[0][_]



def test_MutualInformation_UsedCpuPlatform():
    n = 1000
    k = 4
    X = cuarray.FloatCuArray()
    X.init(2, n)
    I = cuarray.FloatCuArray()
    ab = cuarray.IntCuArray()
    ab.init(1, 2)
    ab.set(0, 0, 0)
    ab.set(1, 0, 1)
    x = 0.001
    for i in range(n):
        X.set(
            float(np.sin(x)),
            0, i,
        )
        X.set(
            float(np.cos(x)),
            1, i,
        )
        x += float(np.pi / n)
    assert netcalc.mutualInformation(
        X=X,
        I=I,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=1,
        platform=netcalc.CPU_PLATFORM,
    ) == 1


def test_MutualInformation_UsedGpuPlatform():
    n = 1000
    k = 4
    X = cuarray.FloatCuArray()
    X.init(2, n)
    I = cuarray.FloatCuArray()
    ab = cuarray.IntCuArray()
    ab.init(1, 2)
    ab.set(0, 0, 0)
    ab.set(1, 0, 1)
    x = 0.001
    for i in range(n):
        X.set(
            float(np.sin(x)),
            0, i,
        )
        X.set(
            float(np.cos(x)),
            1, i,
        )
        x += float(np.pi / n)
    assert netcalc.mutualInformation(
        X=X,
        I=I,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=1,
        platform=netcalc.GPU_PLATFORM,
    ) == 0
