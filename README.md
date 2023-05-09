# NetSci
## A Library for High Performance Scientific Network Analysis Computation
## Installation
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
```
chmod +x Miniconda3-latest-Linux-x86_64.sh
```
```
./Miniconda3-latest-Linux-x86_64.sh
```
```
source ~/.bashrc
```
```
conda install -c conda-forge git
```
```
git clone https://github.com/amstokely/netsci.git
```
```
cd netsci
```
```
conda env create -f netsci.yml
```
```
conda activate netsci
```
```
mkdir build
```
```
cd build
```
```
cmake .. -DCONDA_DIR=${CONDA_PREFIX}
```
```
cmake --build . -j
```
```
make python
```
```
ctest
```
```
cd ../tests/cuarray/python
```
```
pytest
```
```
cd ../../netsci/python
```
```
pytest
```

## Computing generalized correlation for a pyrophosphatase and calcium phosphate hydrolysis simulation

```
cd tutorial
```
``` python
import os
import tarfile

import plotly.graph_objects as go
import numpy as np

import cuarray
import netchem
import netsci
```

``` python
tutorial_files= tarfile.open(f'{os.getcwd()}/pyro.tar.gz')
tutorial_files.extractall(os.getcwd())
tutorial_files.close()
```

``` python
trajectory_file = f'{os.getcwd()}/pyro.dcd'
topology_file = f'{os.getcwd()}/pyro.pdb'
first_frame = 0
last_frame = 999

graph = netchem.Graph()
graph.init(
    trajectoryFile=trajectory_file,
    topologyFile=topology_file,
    firstFrame=first_frame,
    lastFrame=last_frame,
)
```

``` python
n = graph.numFrames()
k = 4
xd = 2
d = 3
platform = 0 #gpu
```

``` python
R = cuarray.FloatCuArray()
```

``` python
num_nodes = graph.numNodes()
num_generalized_correlation_pairs = num_nodes**2
ab = cuarray.IntCuArray()
ab.init(num_generalized_correlation_pairs, 2)
for a in range(num_nodes):
    for b in range(num_nodes):
        pair_index = a * num_nodes + b
        ab.set(a, pair_index, 0)
        ab.set(b, pair_index, 1)
```

``` python
netsci.generalizedCorrelation(
    X=graph.nodeCoordinates(),
    R=R,
    ab=ab,
    n=n,
    k=k,
    xd=xd,
    d=d,
    platform=platform,
)
```

``` python
x = [node_index for node_index in range(num_nodes)]
y = [node_index for node_index in range(num_nodes)]
z = R.toNumpy1D().reshape(
    num_nodes,
    num_nodes,
)
```

``` python
fig = go.Figure(
    data=go.Heatmap(
        x=x, 
        y=y,
        z=z,
        colorscale='jet',
        zsmooth='best',
    )
)
fig.show()
```
