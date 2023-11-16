import os

import pytest
from pathlib import Path

import netchem
import cuarray


@pytest.fixture(scope="module")
def global_network_parameters():
    trajectoryFile, topologyFile = tuple(map(
        lambda f : str(f), netchem.data_files(key="test")
    ))
    return dict(
        trajectoryFile=trajectoryFile,
        topologyFile=topologyFile,
        firstFrame=0,
        lastFrame=9,
    )


@pytest.fixture(scope="module")
def Stride1Network(global_network_parameters):
    network = netchem.Network()
    network.init(
        **global_network_parameters,
        stride=1,
    )
    return network


@pytest.fixture(scope="module")
def Stride2Network(global_network_parameters):
    network = netchem.Network()
    network.init(
        **global_network_parameters,
        stride=2,
    )
    return network


@pytest.fixture(scope="module")
def Stride3Network(global_network_parameters):
    network = netchem.Network()
    network.init(
        **global_network_parameters,
        stride=3,
    )
    return network


@pytest.fixture(scope="module")
def Network(global_network_parameters):
    network = netchem.Network()
    network.init(
        **global_network_parameters,
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
                    == Network.nodeCoordinates()[i][2 * j]
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
                    == Network.nodeCoordinates()[i][3 * j]
            )
            assert (
                    Stride3Network.nodeCoordinates()[i][j + 4]
                    == Network.nodeCoordinates()[i][3 * j + 10]
            )
            assert (
                    Stride3Network.nodeCoordinates()[i][j + 8]
                    == Network.nodeCoordinates()[i][3 * j + 20]
            )

