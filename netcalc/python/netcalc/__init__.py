from pathlib import Path
from typing import List, Mapping, ItemsView, KeysView, ValuesView, Dict, Union
from dataclasses import dataclass, asdict

from cuarray import FloatCuArray, IntCuArray

from .netcalc import mutualInformation
from .netcalc import mutualInformationWithCheckpointing
from .netcalc import mutualInformationGpu
from .netcalc import mutualInformationCpu
from .netcalc import generalizedCorrelation
from .netcalc import generalizedCorrelationWithCheckpointing
from .netcalc import generalizedCorrelationGpu
from .netcalc import generalizedCorrelationCpu
from .netcalc import hedetniemiShortestPaths
from .netcalc import correlationToAdjacency
from .netcalc import mean
from .netcalc import meanGpu
from .netcalc import standardDeviation
from .netcalc import standardDeviationGpu
from .netcalc import longestShortestPathNodeCount
from .netcalc import pathFromPathsCuArray
from .netcalc import GPU_PLATFORM
from .netcalc import CPU_PLATFORM
from .netcalc import generateRestartAbFromCheckpointFile


@dataclass
class MutualInformationSettings(Mapping):
    X: FloatCuArray
    I: FloatCuArray
    ab: IntCuArray
    k: int
    n: int
    xd: int
    d: int
    platform: int = GPU_PLATFORM
    checkpointFrequency: int = -1
    checkpointFileName: Union[str, Path] = None

    def __post_init__(self):
        if self.checkpointFileName is not None:
            self.checkpointFileName = str(self.checkpointFileName)

    def __iter__(self):
        if self.checkpointFrequency == -1:
            for key in asdict(self):
                if key not in ['checkpointFrequency', 'checkpointFileName']:
                    yield key
        else:
            yield from asdict(self)

    def items(self) -> ItemsView:
        return ItemsView({k: self[k] for k in self})

    def keys(self) -> KeysView:
        return KeysView(self)

    def values(self) -> ValuesView:
        return ValuesView({k: self[k] for k in self})

    def __len__(self):
        return len(self.__slots__)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __repr__(self):
        return f"MutualInformationSettings(k={self.k}, n={self.n}, xd={self.xd}, d={self.d}, platform={self.platform}, checkpointFrequency={self.checkpointFrequency}, checkpointFileName={self.checkpointFileName})"


@dataclass
class GeneralizedCorrelationSettings(Mapping):
    X: FloatCuArray
    R: FloatCuArray
    ab: IntCuArray
    k: int
    n: int
    xd: int
    d: int
    platform: int = GPU_PLATFORM
    checkpointFrequency: int = -1
    checkpointFileName: Union[str, Path] = None

    def __post_init__(self):
        if self.checkpointFileName is not None:
            self.checkpointFileName = str(self.checkpointFileName)
        if not self.ab.size():
            print("Initializing ab")
            num_random_variables = int(self.X.n() / self.d)
            ab_m = num_random_variables ** 2
            ab_n = 2
            self.ab.init(
                ab_m,
                ab_n,
            )
            for i in range(num_random_variables):
                for j in range(num_random_variables):
                    self.ab[i * num_random_variables + j][0] = i
                    self.ab[i * num_random_variables + j][1] = j

    def __iter__(self):
        if self.checkpointFrequency == -1:
            for key in asdict(self):
                if key not in ['checkpointFrequency', 'checkpointFileName']:
                    yield key
        else:
            yield from asdict(self)

    def items(self) -> ItemsView:
        return ItemsView({k: self[k] for k in self})

    def keys(self) -> KeysView:
        return KeysView(self)

    def values(self) -> ValuesView:
        return ValuesView({k: self[k] for k in self})

    def __len__(self):
        return len(self.__slots__)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __repr__(self):
        return f"GeneralizedCorrelationSettings(k={self.k}, n={self.n}, xd={self.xd}, d={self.d}, platform={self.platform}, checkpointFrequency={self.checkpointFrequency}, checkpointFileName={self.checkpointFileName})"
