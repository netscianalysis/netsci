<center><h1>NetSci<br><small>A Toolkit for High Performance Scientific Network Analysis Computation</small></h1></center>

---

# Overview
NetSci is a specialized toolkit designed for advanced network analysis in computational sciences. Utilizing the
capabilities of modern GPUs, it offers a powerful and efficient solution for processing computationally demanding
network analysis metrics while delivering state-of-the-art performance.
For detailed **installation** instructions and **tutorials**, please visit [NetSci User Documentation](https://netsci.readthedocs.io/)
For **API documentation** and a general **overview** of C++/CUDA portions of the project, please visit the [NetSci Developer Documentation](https://netscianalysis.github.io).

# Installation

Download Miniconda Installation Script:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Execute the Installation Script:

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Update Environment Settings:

```
source ~/.bashrc
```

Install Git with Conda:

```
conda install -c conda-forge git
```

Clone the NetSci Repository:

```
git clone https://github.com/netscianalysis/netsci.git
```

Navigate to the NetSci Root Directory:

```
cd netsci
```

Create NetSci Conda Environment:

```
conda env create -f netsci.yml
```

Activate NetSci Conda Environment:

```
conda activate netsci
```

Create CMake Build Directory:

```
mkdir build
```

Set NetSci Root Directory Variable:

```
NETSCI_ROOT=$(pwd)
```

Navigate to the CMake Build Directory:

```
cd ${NETSCI_ROOT}/build
```

Compile CUDA Architecture Script:

```
nvcc ${NETSCI_ROOT}/build_scripts/cuda_architecture.cu -o cuda_architecture
```

Set CUDA Architecture Variable:

```
CUDA_ARCHITECTURE=$(./cuda_architecture)
```

Configure the Build with CMake:

```
cmake .. -DCONDA_DIR=$CONDA_PREFIX -DCUDA_ARCHITECTURE=${CUDA_ARCHITECTURE}
```

Build NetSci:

```
cmake --build . -j
```

Build NetSci Python Interface:

```
make python
```

Test C++ and CUDA Backend:

```
ctest
```

Run Python Interface Tests::

```
cd ${NETSCI_ROOT}
pytest
```

# Examples and Tutorials

Examples may be found in the examples/ subdirectory.

Detailed tutorials can be found at [NetSci User Documentation](https://netsci.readthedocs.io/).

Jupyter notebooks of tutorials can be found in the tutorials/ subdirectory.

# Citing NetSci

If you use NetSci, please cite the following paper:

* NetSci: A Library for High Performance Biomolecular Simulation Network Analysis Computation
Andrew M. Stokely, Lane W. Votapka, Marcus T. Hock, Abigail E. Teitgen, J. Andrew McCammon, Andrew D. McCulloch, and Rommie E. Amaro
Journal of Chemical Information and Modeling 2024 64 (20), 7966-7976
DOI: 10.1021/acs.jcim.4c00899

# Copyright

Copyright (c) 2024, Andy Stokely and Lane Votapka


