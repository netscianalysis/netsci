<center><h1>NetSci<br><small>A Toolkit for High Performance Scientific Network Analysis Computation</small></h1></center>

NetSci is a specialized toolkit designed for advanced network analysis in computational sciences. Utilizing the
capabilities of modern GPUs, it offers a powerful and efficient solution for processing computationally demanding
network analysis metrics, delivering exceptional performance.
<!-- TOC -->

* [Installation](#installation)
* [API Documentation](#api-documentation)
    * [CuArray](#cuarray)
        * [C++](#c)
        * [Python](#python)
        * [Tcl](#tcl)
    * [NetChem](#netchem)
        * [C++](#c-1)
        * [Python](#python-1)
    * [NetCalc](#netcalc)
        * [C++](#c-2)
        * [Python](#python-2)
* [Usage](#usage)
    * [NetCalc](#netcalc-1)
        * [Python](#python-3)
* [Tutorials](#tutorials)
    * [NetCalc](#netcalc-2)
        * [Python](#python-4)

<!-- TOC -->

# Installation

NetSci is designed with a focus on ease of installation and long-term stability, ensuring compatibility with Linux
systems featuring CUDA-capable GPUs (compute capability 3.5 and above). It leverages well-supported core C++ and Python
libraries to maintain simplicity and reliability.

1. **Download Miniconda Installation Script:**
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```
1. **Execute the Installation Script:**
    ```bash
    bash https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```
1. **Update Environment Settings:**
    ```bash
    source ~/.bashrc
    ```
1. **Install Git with Conda:**
    ```bash
    conda install -c conda-forge git
    ```
1. **Clone the NetSci Repository:**
    ```bash
    git clone https://github.com/netscianalysis/netsci.git
    ```

1. **Navigate to the NetSci Root Directory:**
    ```bash
    cd netsci
    ```
1. **Set NetSci Root Directory Variable:**
    ```bash
    NETSCI_ROOT=$(pwd)
    ```

1. **Create NetSci Conda Environment:**
    ```bash
    conda env create -f netsci.yml
    ```
1. **Activate NetSci Conda Environment:**
    ```bash
    source activate netsci
    ```
1. **Prepare the Build Directory:**
    ```bash
    mkdir ${NETSCI_ROOT}/build
    cd ${NETSCI_ROOT}/build
    ```

1. **Compile CUDA Script for GPU Capability:**
    ```bash
    nvcc ${NETSCI_ROOT}/build_scripts/cuda_architecture.cu -o cuda_architecture
    ```
1. **Set CUDA Architecture Variable:**
    ```bash
    CUDA_ARCHITECTURE=$(./cuda_architecture)
    ```
1. **Configure the Build with CMake:**
    ```bash
    cmake .. -DCONDA_DIR=$CONDA_PREFIX -DCUDA_ARCHITECTURE=${CUDA_ARCHITECTURE}
    ```
1. **Build NetSci:**
    ```bash
    cmake --build . -j
    ```
1. **Build NetSci Python Interface:**
    ```bash
    make python
    ```
1. **Test C++ and CUDA Backend:**
    ```bash
    ctest
    ```
1. **Run Python Interface Tests:**
    ```bash
    cd ${NETSCI_ROOT}
    pytest
    ```

# API Documentation

## CuArray

### C++

| <pre><code><b>CuArray()</b></code></pre>                                          |
|-----------------------------------------------------------------------------------|
| <pre><code><b>CuArray::init(<br/>    int m, <br/>    int n<br/>)</b></code></pre> |
|                                                                                   |

### Python

### Tcl

## NetChem

### C++

### Python

## NetCalc

### C++

### Python

# Usage

## NetCalc

### Python

# Tutorials

## NetCalc

### Python

