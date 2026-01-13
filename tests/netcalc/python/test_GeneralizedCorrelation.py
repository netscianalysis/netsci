import numpy as np
import cuarray
import netcalc
import pytest


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def _build_ab(num_nodes):
    num_node_pairs = num_nodes ** 2
    ab = cuarray.IntCuArray()
    ab.init(num_node_pairs, 2)
    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = i * num_nodes + j
            ab[idx][0] = i
            ab[idx][1] = j
    return ab


def test_GeneralizedCorrelation_2X1D_1000n4k_GpuCpu():
    n = 1000
    k = 4

    X = cuarray.FloatCuArray()
    Xref = np.zeros((2, n), dtype=np.float32)

    Rcpu = cuarray.FloatCuArray()
    Rgpu = cuarray.FloatCuArray()
    x = 0.001
    for i in range(n):
        sin_val = np.sin(x).astype(np.float32)
        cos_val = np.cos(x).astype(np.float32)
        Xref[0][i] = sin_val
        Xref[1][i] = cos_val
    X.fromNumpy2D(Xref)

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
        I=Rgpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=1,
        platform=netcalc.GPU_PLATFORM,
    )
    netcalc.mutualInformation(
        X=X,
        I=Rcpu,
        ab=ab,
        k=k,
        n=n,
        xd=2,
        d=1,
        platform=netcalc.CPU_PLATFORM,
    )
    for i in range(4):
        assert Rgpu[0][i] == Rcpu[0][i]


def test_GeneralizedCorrelation_2X1D_2000n4k_GpuCpu():
    n = 2000
    k = 4

    X = cuarray.FloatCuArray()
    Xref = np.zeros((2, n), dtype=np.float32)

    Rcpu = cuarray.FloatCuArray()
    Rgpu = cuarray.FloatCuArray()
    ab = _build_ab(num_nodes=2)

    x = 0.001
    for i in range(n):
        Xref[0][i] = np.sin(x).astype(np.float32)
        Xref[1][i] = np.cos(x).astype(np.float32)
        x += float(np.pi / n)
    X.fromNumpy2D(Xref)

    netcalc.generalizedCorrelation(
        X=X, R=Rgpu, ab=ab,
        k=k, n=n, xd=2, d=1,
        platform=netcalc.GPU_PLATFORM,
    )
    netcalc.generalizedCorrelation(
        X=X, R=Rcpu, ab=ab,
        k=k, n=n, xd=2, d=1,
        platform=netcalc.CPU_PLATFORM,
    )

    for i in range(4):
        assert Rgpu[0][i] == Rcpu[0][i]


def test_GeneralizedCorrelation_2X2D_1000n4k_GpuCpu():
    n = 1000
    k = 4

    X = cuarray.FloatCuArray()
    Xref = np.zeros((2, 2 * n), dtype=np.float32)

    Rcpu = cuarray.FloatCuArray()
    Rgpu = cuarray.FloatCuArray()
    ab = _build_ab(num_nodes=2)

    x = 0.001
    for i in range(n):
        Xref[0][i] = np.sin(x).astype(np.float32)
        Xref[1][i] = x
        Xref[0][i + n] = np.cos(x).astype(np.float32)
        Xref[1][i + n] = 2 * x
        x += float(np.pi / n)
    X.fromNumpy2D(Xref)

    netcalc.generalizedCorrelation(
        X=X, R=Rgpu, ab=ab,
        k=k, n=n, xd=2, d=2,
        platform=netcalc.GPU_PLATFORM,
    )
    netcalc.generalizedCorrelation(
        X=X, R=Rcpu, ab=ab,
        k=k, n=n, xd=2, d=2,
        platform=netcalc.CPU_PLATFORM,
    )

    for i in range(4):
        assert Rgpu[0][i] == Rcpu[0][i]


def test_GeneralizedCorrelation_2X2D_2000n4k_GpuCpu():
    n = 2000
    k = 4

    X = cuarray.FloatCuArray()
    Xref = np.zeros((2, 2 * n), dtype=np.float32)

    Rcpu = cuarray.FloatCuArray()
    Rgpu = cuarray.FloatCuArray()
    ab = _build_ab(num_nodes=2)

    x = 0.001
    for i in range(n):
        Xref[0][i] = np.sin(x).astype(np.float32)
        Xref[1][i] = x
        Xref[0][i + n] = np.cos(x).astype(np.float32)
        Xref[1][i + n] = (2 * x)
        x += float(np.pi / n)
    X.fromNumpy2D(Xref)

    netcalc.generalizedCorrelation(
        X=X, R=Rgpu, ab=ab,
        k=k, n=n, xd=2, d=2,
        platform=netcalc.GPU_PLATFORM,
    )
    netcalc.generalizedCorrelation(
        X=X, R=Rcpu, ab=ab,
        k=k, n=n, xd=2, d=2,
        platform=netcalc.CPU_PLATFORM,
    )

    for i in range(4):
        assert Rgpu[0][i] == Rcpu[0][i]


def test_GeneralizedCorrelation_2X3D_2000n4k_GpuCpu():
    n = 2000
    k = 4

    X = cuarray.FloatCuArray()
    Xref = np.zeros((3, 2 * n), dtype=np.float32)

    Rcpu = cuarray.FloatCuArray()
    Rgpu = cuarray.FloatCuArray()
    ab = _build_ab(num_nodes=2)

    x = 0.001
    for i in range(n):
        Xref[0][i] = np.sin(x).astype(np.float32)
        Xref[1][i] = x
        Xref[2][i] = np.log(x).astype(np.float32)
        Xref[0][i + n] = np.cos(x).astype(np.float32)
        Xref[1][i + n] = 2 * x
        Xref[2][i + n] = np.log(2 * x).astype(np.float32)
        x += float(np.pi / n)
    X.fromNumpy2D(Xref)

    netcalc.generalizedCorrelation(
        X=X, R=Rgpu, ab=ab,
        k=k, n=n, xd=2, d=3,
        platform=netcalc.GPU_PLATFORM,
    )
    netcalc.generalizedCorrelation(
        X=X, R=Rcpu, ab=ab,
        k=k, n=n, xd=2, d=3,
        platform=netcalc.CPU_PLATFORM,
    )

    for i in range(4):
        assert np.abs(Rgpu[0][i] - Rcpu[0][i]) < 1e-5


def test_GeneralizedCorrelation_UsedCpuPlatform():
    n = 1000
    k = 4

    X = cuarray.FloatCuArray()
    X.init(2, n)
    R = cuarray.FloatCuArray()
    ab = _build_ab(num_nodes=1)

    x = 0.001
    for i in range(n):
        X.set(float(np.sin(x)), 0, i)
        X.set(float(np.cos(x)), 1, i)
        x += float(np.pi / n)

    assert netcalc.generalizedCorrelation(
        X=X, R=R, ab=ab,
        k=k, n=n, xd=2, d=1,
        platform=netcalc.CPU_PLATFORM,
    ) == 1


def test_GeneralizedCorrelation_UsedGpuPlatform():
    n = 1000
    k = 4

    X = cuarray.FloatCuArray()
    X.init(2, n)
    R = cuarray.FloatCuArray()
    ab = _build_ab(num_nodes=1)

    x = 0.001
    for i in range(n):
        X.set(float(np.sin(x)), 0, i)
        X.set(float(np.cos(x)), 1, i)
        x += float(np.pi / n)

    assert netcalc.generalizedCorrelation(
        X=X, R=R, ab=ab,
        k=k, n=n, xd=2, d=1,
        platform=netcalc.GPU_PLATFORM,
    ) == 0
