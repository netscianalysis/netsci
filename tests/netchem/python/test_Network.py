import pytest
from pathlib import Path

import netchem

DCD_FILE = str(Path("../cpp/data/test.dcd").absolute())
PDB_FILE = str(Path("../cpp/data/test.pdb").absolute())
FIRST_FRAME = 0
LAST_FRAME = 9


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture(scope="module")
def Stride1Network():
    network = netchem.Network()
    network.init(
        trajectoryFile=DCD_FILE,
        topologyFile=PDB_FILE,
        firstFrame=FIRST_FRAME,
        lastFrame=LAST_FRAME,
        stride=1,
    )
    return network

@pytest.fixture(scope="module")
def Stride2Network():
    network = netchem.Network()
    network.init(
        trajectoryFile=DCD_FILE,
        topologyFile=PDB_FILE,
        firstFrame=FIRST_FRAME,
        lastFrame=LAST_FRAME,
        stride=2,
    )
    return network

@pytest.fixture(scope="module")
def Stride3Network():
    network = netchem.Network()
    network.init(
        trajectoryFile=DCD_FILE,
        topologyFile=PDB_FILE,
        firstFrame=FIRST_FRAME,
        lastFrame=LAST_FRAME,
        stride=3,
    )
    return network


@pytest.fixture(scope="module")
def Network():
    network = netchem.Network()
    network.init(
        trajectoryFile=DCD_FILE,
        topologyFile=PDB_FILE,
        firstFrame=FIRST_FRAME,
        lastFrame=LAST_FRAME,
    )
    return network


def test_Stride1CoordinateEquality(Stride1Network, Network):
    for i in range(289):
        for j in range(30):
            assert (
                    Stride1Network.nodeCoordinates()[i][j]
                    == Network.nodeCoordinates()[i][j]
            )


def test_Stride2CoordinateEquality(Stride2Network, Network):
    for i in range(289):
        for j in range(15):
            assert (
                    Stride2Network.nodeCoordinates()[i][j]
                    == Network.nodeCoordinates()[i][2*j]
            )


def test_Stride3CoordinateEquality(Stride3Network, Network):
    for i in range(289):
        assert (
                Stride3Network.nodeCoordinates()[i][0]
                == Network.nodeCoordinates()[i][0]
        )
        assert (
                Stride3Network.nodeCoordinates()[i][4]
                == Network.nodeCoordinates()[i][10]
        )
        assert (
                Stride3Network.nodeCoordinates()[i][8]
                == Network.nodeCoordinates()[i][20]
        )
        for j in range(1, 4):
            assert (
                    Stride3Network.nodeCoordinates()[i][j]
                    == Network.nodeCoordinates()[i][3*j]
            )
            assert (
                    Stride3Network.nodeCoordinates()[i][j+4]
                    == Network.nodeCoordinates()[i][3*j+10]
            )
            assert (
                    Stride3Network.nodeCoordinates()[i][j+8]
                    == Network.nodeCoordinates()[i][3*j+20]
            )




