# NetSci
## A Library for High Performance Scientific Network Analysis Computation
## Installation
``` Shell
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
source activate netsci
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

## A Brief Overview of CuArray: NetSci's CUDA-Compatible Array Library

``` python
import numpy as np

from cuarray import FloatCuArray, IntCuArray
```

```python
def print_CuArray(
        cuArray,
):
    for i in range(cuArray.m()):
        row_repr = ''
        for j in range(cuArray.n()):
            val = cuArray.get(i, j)
            row_repr += f'{val:.2f} '
        print(row_repr)
```

```python
def print_numpy_array(
        numpy_array,
):
    m, n = numpy_array.shape
    for i in range(m):
        row_repr = ''
        for j in range(n):
            val = numpy_array[i, j]
            row_repr += f'{val:.2f} '
        print(row_repr)
```

``` python
a = FloatCuArray()
m, n = 10, 10
a.init(m, n)
```
``` python
for i in range(m):
    for j in range(n):
        val = np.random.rand(1)[0]
        a.set(val, i, j)
```

``` python
print_CuArray(a)
```

``` python
a_copy = FloatCuArray()
a_copy.fromCuArray(
    cuArray=a,
    start=0,
    end = a.m() - 1,
    m=a.m(),
    n=a.n(),
)
```

``` python
print_CuArray(a_copy)
```

``` python
a_row0 = FloatCuArray()
a_row0.fromCuArray(
    cuArray=a,
    start=0,
    end=0,
    m=1,
    n=a.n(),
)
```

``` python
print_CuArray(a_row0)
```

``` python
a_row0_reshape_5x2 = FloatCuArray()
a_row0_reshape_5x2.fromCuArray(
    cuArray=a,
    start=0,
    end=0,
    m=5,
    n=2,
)
```

``` python
print_CuArray(a_row0_reshape_5x2)
```

``` python
a.save('a.npy')
```

``` python
b = FloatCuArray()
b.load('a.npy')
```

``` python
print_CuArray(b)
```

``` python
np_a_2D = a.toNumpy2D()
```

``` python
print_numpy_array(np_a_2D)
```

``` python
a_np_2D = FloatCuArray()
a_np_2D.fromNumpy2D(np_a_2D)
```

``` python
print_CuArray(a_np_2D)
```

``` python
np_a_row0_1D = a_row0.toNumpy1D()
```

``` python
print_numpy_array(np_a_row0_1D)
```

``` python
```


## Analyzing a Pyrophosphatase Molecular Dynamics Simulation with NetSci

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
'''
Number of nodes in the graph netchem creates from 
the trajectory.
'''
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
