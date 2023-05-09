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
source activate ~/.bashrc
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
import tarfile

tutorial_files = tarfile.
```
