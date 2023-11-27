import os
import tempfile
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import tarfile
from importlib import resources
from typing import ItemsView, KeysView, ValuesView, Mapping, Union

from .netchem import Atom
from .netchem import Atoms
from .netchem import Node
from .netchem import Network
from .netchem import Network as Graph


@dataclass
class NetworkSettings(Mapping):
    trajectoryFile: Union[str, Path]
    topologyFile: Union[str, Path]
    firstFrame: int
    lastFrame: int
    stride: int = 1

    def __post_init__(self):
        self.trajectoryFile = str(self.trajectoryFile)
        self.topologyFile = str(self.topologyFile)

    def __iter__(self):
        return iter(asdict(self))

    def __len__(self):
        return len(self.__slots__)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def items(self) -> ItemsView:
        return ItemsView(self)

    def keys(self) -> KeysView:
        return KeysView(self)

    def values(self) -> ValuesView:
        return ValuesView(self)


def data_files(key: str) -> tuple:
    temp_dir = tempfile.mkdtemp()
    data_files_dict = dict(
        test="test.tar.gz",
        pyro="pyro.tar.gz",
    )
    cwd = Path.cwd()
    with resources.path("netchem.data", data_files_dict[key]) as tarball_file:
        os.chdir(tarball_file.parent)
        tarball = tarfile.open(tarball_file)
        tarball_members = [tarball_file.parent / f.name for f in tarball.getmembers()]
        temp_files = [Path(temp_dir) / f.name for f in tarball_members]
        tarball.extractall()
        tarball.close()
        for tarball_member, temp_file in zip(tarball_members, temp_files):
            shutil.copy(tarball_member, temp_file)
            tarball_member.unlink()
    os.chdir(cwd)
    return (
        Path(temp_dir) / f"{key}.dcd",
        Path(temp_dir) / f"{key}.pdb"
    )
