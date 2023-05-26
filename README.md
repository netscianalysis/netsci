# **NetSci**: A Library for High Performance Scientific Network Analysis Computation

## Installation

Follow these steps to install and set up the NetSci project:

1. Download Miniconda installer:
   ```shell
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```


2. Make the installer executable:
   ```shell
   chmod +x Miniconda3-latest-Linux-x86_64.sh
   ```

3. Run the installer:
   ```shell
   ./Miniconda3-latest-Linux-x86_64.sh
   ```

4. Update the shell environment:
   ```shell
   source ~/.bashrc
    ```

5. Install Git:
   ```shell
   conda install -c conda-forge git
    ```
6. Clone the NetSci repository:
   ```shell
   git clone https://github.com/amstokely/netsci.git
   ```
7. Navigate to the NetSci root directory:
   ```shell
    cd netsci
    ```  
8. Create a new conda environment named "netsci":
   ```shell
   conda env create -f netsci.yml
   ```
9. Activate the "netsci" environment:
   ```shell
    source activate netsci
    ```
10. Create the build directory:
   ```shell
  mkdir build
   ```
11. Move into the build directory:
    ```shell
    cd build
    ```
12. Generate the CMake build files:
    ```shell
    cmake .. -DCONDA_DIR=$CONDA_PREFIX
    ```
    where $CONDA_PREFIX is the path to the conda environment.
    By default, NetSci is built for CUDA architecture 5.2. This may need to be adjusted depending on the GPU being used.
    To build for a different CUDA architecture, add the following flag to the above command:
   ```shell
   -DCUDA_ARCHITECTURE=<cuda architecture * 10> #To build for CUDA architecture 6.1, use -DCUDA_ARCHITECTURE=61
   ```
13. Build NetSci:
   ```shell
   cmake --build . -j
   ```
14. Build the NetSci Python interface:
    ```shell
    make python
    ```
15. Run the C++ and CUDA unit tests:
   ```shell
   ctest
   ``` 
16. Navigate to the CuArray python test directory:
   ```shell
   cd ../tests/cuarray/python
   ``` 
17. Run the CuArray python tests:
   ```shell
   pytest
   ```
18. Navigate to the NetCalc python test directory:
   ```shell
   cd ../tests/netcalc/python
   ``` 
17. Run the NetCalc python tests:
   ```shell
   pytest
   ```

<h1 align="center">
<b>Tutorial</b>
</h1>

<h2 align="center">
<b>Untar Tutorial Files</b>
</h2>
If you plan to run the tutorial, you will need to untar the tutorial files tarball. 
Navigate to the netsci root directory and enter the tutorial directory.

``` Shell
cd tutorial
```

and untar **tutorial.tar.gz**.

``` Shell
tar -xvzf tutorial.tar.gz
```

<h2 align="center">
<b>CuArray: NetSci's CUDA-Compatible Array Library</b>
</h2>
CuArray is the core mathematical data structure used by all NetSci libraries. 
The CuArray library includes two main classes: FloatCuArray for floating-point arrays and IntCuArray for integer arrays. 
These classes provide GPU memory management and efficient data transfer between the CPU and GPU. Examples of how to use
all important CuArray functions, that are callable from the python interface, are provide below. To get started, import 
numpy, FloatCuArray, IntCuArray, and create two utility functions named print_CuArray and print_numpy_array.
The print_CuArray function facilitates easy printing of the contents of a FloatCuArray to the console.
It iterates over all rows and columns, retrieves each element's value using the get method, 
and prints the formatted string representation.

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
The print_numpy_array function allows for printing the contents of a NumPy array to the console. It iterates over all 
rows and columns, retrieves each element's value, and prints the formatted string representation.
``` python
def print_numpy_array(
        numpy_array: np.ndarray,
)->None:
    """
    Prints a numpy array to the console.
    @param numpy_array: The numpy array to print.
    """
    # Get the number of rows and columns in the numpy array.
    # Iterate over all rows.
    if len(numpy_array.shape) == 2:
        m, n = numpy_array.shape
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
    elif len(numpy_array.shape) == 1:
        n = numpy_array.shape[0]
        # Initialize the value representation string.
        val_repr = ''
        for i in range(n):
            # Get the value at the current index. 
            val = numpy_array[i]
            # Append the value to the val representation string.
            val_repr += f'{val:.5f} '
        # Print the val representation string.
        print(val_repr)
```

To create a new FloatCuArray, one can simply call its constructor and then initialize it with the desired 
number of rows and columns using the init method. The values of a FloatCuArray can be set by iterating over its rows 
and columns and using the set method to assign a value at a specific row and column index.

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

The m and n methods of a FloatCuArray can be used to retrieve the number of rows and columns, respectively.
``` python
# Use the CuArray m and n methods to get the number
# of rows and columns in "a".
m, n = a.m(), a.n()
print(f'm: {m}, n: {n}')
```

The bytes method of a FloatCuArray returns the total number of bytes occupied by the array's data.
``` python
# Use the CuArray bytes method to get the number 
# of bytes in "a".
a_bytes = a.bytes()
print(f'bytes: {a_bytes}')
```

The size method of a FloatCuArray returns the total number of elements in the array.
``` python
# Use the CuArray size method to get the number
# of elements in "a".
a_size = a.size()
print(f'size: {a_size}')
```

A new FloatCuArray named a_copy is initialized, and a deep copy of a subset of rows from a is performed using the 
fromCuArray method.
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

A new FloatCuArray named a_row0 is initialized, and a deep copy of the first row of a is 
performed using the fromCuArray method.
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

A new FloatCuArray named a_row0_reshape_5x2 is initialized, and a deep copy of the first row of a is performed using 
the fromCuArray method. The copied data is reshaped to have 5 rows and 2 columns.


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

The save method of a FloatCuArray is used to save its contents to a .npy numpy binary file.
``` python
# Use the CuArray save method to save "a" to a .npy numpy
# binary file.
a.save('a.npy')
```

A new FloatCuArray named b is initialized, and the contents of the previously saved .npy file are loaded into it 
using the load method.
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

The toNumpy2D method of a FloatCuArray is used to copy its data into a 2D NumPy array named np_a_2D.
``` python
# Use the CuArray toNumpy2D method to copy "a" into a 
# 2D numpy array named "np_a_2D".
np_a_2D = a.toNumpy2D()
```

``` python
# Print "np_a_2D.
print_numpy_array(np_a_2D)
```

A new FloatCuArray named a_np_2D is initialized, and the data from np_a_2D is copied into it using the 
fromNumpy2D method.
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

The toNumpy1D method of a FloatCuArray is used to copy its data into a 1D NumPy array named np_a_row0_1D.
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

An out-of-place descending sort of the first row of a is performed using the sort method, and the sorted result is
stored in a new FloatCuArray named sorted_a_row0.
``` python
# Perform a out-of-place descending sort of the first
# of "a" store the result in "sorted_a_row0".
sorted_a_row0 = a.sort(0)
```

``` python
# Print "a_row0".
print_CuArray(a_row0)
# Print "sorted_a_row0".
print_CuArray(sorted_a_row0)`
```

An out-of-place descending argsort of the first row of a is performed using the argsort method, and the resulting 
array, containing the indices that would sort the row in descending order, is stored in argsort_a_row0.
``` python
# Perform an out-of-place descending argsort of the first
# row of "a" and store the result in "argsort_a_row0".
argsort_a_row0 = a.argsort(0)
```

The loop iterates over the values in argsort_a_row0, retrieves the corresponding indices and values from the 
first row of a, and prints the i-th largest elements in the first row of a along with their indices in descending order.
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

The above code snippets provide examples of how to use all the CuArray functions that the vast majority of users will
ever use. However, for a complete list of CuArray functions, refer to the NetSci documentation.

<h1 align="center">
NetChem: NetSci's Molecular Dynamics Trajectory Graph Library
</h1>

``` python
import os

from netchem import Graph
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
node_coordinates = graph.nodeCoordinates()
print(node_coordinates)
```

``` python
nodes = graph.nodes()
print(nodes)
```

``` python
atoms = graph.atoms()
print(atoms)
```

``` python
for node in graph:
    print(node.__class__)
```

``` python
for atom in graph.atoms():
    print(atom.__class__)
```

``` python
```

<h1 align="center">
Analyzing a Pyrophosphatase Molecular Dynamics Simulation with NetSci
</h1>

``` python
import os

import plotly.graph_objects as go
import numpy as np

import cuarray
import netchem
import netcalc 
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
netcalc.generalizedCorrelation(
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
