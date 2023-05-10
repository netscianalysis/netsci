# **NetSci**: A Library for High Performance Scientific Network Analysis Computation
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

<h1 align="center">
<b>Usage</b>
</h1>

<h2 align="center">
<b>CuArray: NetSci's CUDA-Compatible Array Library</b>
</h2>

``` python
import numpy as np

#Import the float and integer CuArray classes.
from cuarray import FloatCuArray, IntCuArray
```

``` python
def print_CuArray(
        cuArray: FloatCuArray,
)->None:
    """
    Prints a CuArray to the console.
    @param cuArray: The CuArray to print.
    """
    # Iterate over all rows.
    for i in range(cuArray.m()):
        # Initialize the row representation string.
        row_repr = ''
        # Iterate over all columns.
        for j in range(cuArray.n()):
            # Get the value at the current row and column.  
            val = cuArray.get(i, j)
            # Append the value to the row representation string.
            row_repr += f'{val:.5f} '
        # Print the row representation string.
        print(row_repr)
```

``` python
def print_numpy_array(
        numpy_array: np.ndarray,
)->None:
    """
    Prints a numpy array to the console.
    @param numpy_array: The numpy array to print.
    """
    # Get the number of rows and columns in the numpy array.
    m, n = numpy_array.shape
    # Iterate over all rows.
    for i in range(m):
        # Initialize the row representation string.
        row_repr = ''
        # Iterate over all columns.
        for j in range(n):
            # Get the value at the current row and column.
            val = numpy_array[i, j]
            # Append the value to the row representation string.
            row_repr += f'{val:.5f} '
        # Print the row representation string.
        print(row_repr)
```

``` python
# Create a new CuArray named "a".
a = FloatCuArray()
# Initialize "a" with 10 rows and 10 columns.
m, n = 10, 10
a.init(m, n) 
```
``` python
# Set the values of "a" to random values.
for i in range(m):
    for j in range(n):
        val = np.random.rand(1)[0]
        # Set the value at the current row and column.
        a.set(val, i, j)
```

``` python
# Print the "a".
print_CuArray(a)
```

``` python
# Use the CuArray m and n methods to get the number
# of rows and columns in "a".
m, n = a.m(), a.n()
print(f'm: {m}, n: {n}')
```

``` python
# Use the CuArray bytes method to get the number 
# of bytes in "a".
a_bytes = a.bytes()
print(f'bytes: {a_bytes}')
```

``` python
# Use the CuArray size method to get the number
# of elements in "a".
a_size = a.size()
print(f'size: {a_size}')
```

``` python
# Initialize a new CuArray named "a_copy".
a_copy = FloatCuArray()
# Use the CuArray fromCuArray method to perform a 
# deep copy of "a" into "a_copy".
a_copy.fromCuArray(
    cuArray=a, # The CuArray to copy.
    start=0, # Index of the first row in "a" to copy.
    end=a.m() - 1, # Index of the last row in "a" to copy.
    m=a.m(), # The number of rows in "a_copy".
    n=a.n(), # The number of columns in "a_copy".
)
```

``` python
# Print "a_copy". 
print_CuArray(a_copy)
```

``` python
# Create a new CuArray named "a_row0".
a_row0 = FloatCuArray()
# Use the CuArray fromCuArray method to perform a
# deep copy of the first row of "a" into
# "a_row0".
a_row0.fromCuArray(
    cuArray=a,
    start=0,
    end=0,
    m=1,
    n=a.n(),
)
```

``` python
# Print "a_row0".
print_CuArray(a_row0)
```

``` python
# Create a new CuArray named "a_row0_reshape_5x2".
a_row0_reshape_5x2 = FloatCuArray()
# Use the CuArray fromCuArray method to perform a
# deep copy of the first row of "a" into
# "a_row0_reshape_5x2" and reshape it to
# have 5 rows and 2 columns.
a_row0_reshape_5x2.fromCuArray(
    cuArray=a,
    start=0,
    end=0,
    m=5,
    n=2,
)
```

``` python
# Print "a_row0_reshape_5x2".
print_CuArray(a_row0_reshape_5x2)
```

``` python
# Use the CuArray save method to save "a" to a .npy numpy
# binary file.
a.save('a.npy')
```

``` python
# Create a new CuArray named "b".
b = FloatCuArray()
# Use the CuArray load method to load "a" from the .npy
# numpy binary file into "b".
b.load('a.npy')
```

``` python
# Print "b".
print_CuArray(b)
```

``` python
# Use the CuArray toNumpy2D method to convert "a" into a 
# 2D numpy array named "np_a_2D".
np_a_2D = a.toNumpy2D()
```

``` python
# Print "np_a_2D.
print_numpy_array(np_a_2D)
```

``` python
# Create a new CuArray named "a_np_2D".
a_np_2D = FloatCuArray()
# Use the CuArray fromNumpy2D method to 
# copy "np_a_2D" into "a_np_2D".
a_np_2D.fromNumpy2D(np_a_2D)
```

``` python
# Print "a_np_2D".
print_CuArray(a_np_2D)
```

``` python
# Use the CuArray toNumpy1D method 
# copy "a_row0" into a 1D
# numpy array named "np_a_row0_1D". 
np_a_row0_1D = a_row0.toNumpy1D()
```

``` python
# Print "np_a_row0_1D".
print_numpy_array(np_a_row0_1D)
```

``` python
# Perform a out-of-place descending sort of the first
# of "a" store the result in "sorted_a_row0".
sorted_a_row0 = a.sort(0)
```

``` python
# Print "a_row0".
print_CuArray(a_row0)
# Print "sorted_a_row0".
print_CuArray(sorted_a_row0)
```

``` python
# Perform an out-of-place descending argsort of the first
# row of "a" and store the result in "argsort_a_row0".
argsort_a_row0 = a.argsort(0)
```

``` python
# Iterate over all values in argsort_a_row0 
for i in range(n):
    # Get the index of the i-th largest value in the first
    # row of "a".
    argsort_idx = argsort_a_row0.get(0, i)
    # Get the i-th largest value in the first row of "a".
    val = a_row0.get(0, argsort_idx)
    # Print the i'th largest value in the first row of "a" 
    # and its index.
    print(f'{argsort_idx}: {val:.5f}')
```

<h1 align="center">
NetChem: NetSci's Molecular Dynamics Trajectory Graph Library
</h1>

``` python
import os
import tarfile

from netchem import Graph
```

``` python
tutorial_files= tarfile.open(f'{os.getcwd()}/pyro.tar.gz')
tutorial_files.extractall(os.getcwd())
tutorial_files.close()
```

``` python
trajectory_file = f'{os.getcwd()}/pyro.dcd'
topology_file = f'{os.getcwd()}/pyro.pdb'
```

``` python
first_frame = 1
last_frame = 1000
```

``` python
graph = Graph()
graph.init(
    trajectoryFile=trajectory_file,
    topologyFile=topology_file,
    firstFrame=first_frame,
    lastFrame=last_frame,
)
```

``` python
num_nodes = graph.numNodes()
print(num_nodes)
```

``` python
num_frames = graph.numFrames()
print(num_frames)
```

``` python
for node in graph:
    print(node)
```

``` python
```


<h1 align="center">
Analyzing a Pyrophosphatase Molecular Dynamics Simulation with NetSci
</h1>

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
first_frame = 1
last_frame = 1000

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
