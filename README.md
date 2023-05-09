# NetSci
## A Library for High Performance Scientific Network Analysis Computation
## Installation
```
conda env create -f netsci.yml
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
