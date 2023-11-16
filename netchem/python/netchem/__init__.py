import os
import tempfile
import shutil
from pathlib import Path
import tarfile
from importlib import resources

from .netchem import Atom
from .netchem import Atoms
from .netchem import Node
from .netchem import Network
from .netchem import Network as Graph


def data_files(key: str) -> tuple:
    temp_dir = tempfile.mkdtemp()
    data_files_dict = dict(
        test="test.tar.gz",
        pyro="pyro.tar.gz",
    )
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
    return (
        Path(temp_dir) / f"{key}.dcd",
        Path(temp_dir) / f"{key}.pdb"
    )
