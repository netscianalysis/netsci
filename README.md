<center><h1>NetSci<br><small>A Toolkit for High Performance Scientific Network Analysis Computation</small></h1></center>

---

# Overview
NetSci is a specialized toolkit designed for advanced network analysis in computational sciences. Utilizing the
capabilities of modern GPUs, it offers a powerful and efficient solution for processing computationally demanding
network analysis metrics while delivering state-of-the-art performance.

---

# Installation

NetSci is designed with a focus on ease of installation and long-term stability, ensuring compatibility with Linux
systems featuring CUDA-capable GPUs (compute capability 3.5 and above). It leverages well-supported core C++ and Python
libraries to maintain simplicity and reliability.
<details>

<summary><b>Steps</b></summary>

* [Download Miniconda Installation Script](#download-miniconda-installation-script)
* [Execute the Installation Script](#execute-the-installation-script)
* [Update Environment Settings](#update-environment-settings)
* [Install Git with Conda](#install-git-with-conda)
* [Clone the NetSci Repository](#clone-the-netsci-repository)
* [Navigate to the NetSci Root Directory](#navigate-to-the-netsci-root-directory)
* [Create NetSci Conda Environment](#create-netsci-conda-environment)
* [Activate NetSci Conda Environment](#activate-netsci-conda-environment)
* [Create CMake Build Directory](#create-cmake-build-directory)
* [Set NetSci Root Directory Variable](#set-netsci-root-directory-variable)
* [Navigate to the CMake Build Directory](#navigate-to-the-cmake-build-directory)
* [Compile CUDA Script for GPU Capability](#compile-cuda-script-for-gpu-capability)
* [Set CUDA Architecture Variable](#set-cuda-architecture-variable)
* [Configure the Build with CMake](#configure-the-build-with-cmake)
* [Build NetSci](#build-netsci)
* [Build NetSci Python Interface](#build-netsci-python-interface)
* [Test C++ and CUDA Backend](#test-c-and-cuda-backend)
* [Run Python Interface Tests](#run-python-interface-tests)


1. #### Download Miniconda Installation Script:
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```
1. #### Execute the Installation Script:
    ```bash
    bash https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```
1. #### Update Environment Settings:
    ```bash
    source ~/.bashrc
    ```
1. #### Install Git with Conda:
    ```bash
    conda install -c conda-forge git
    ```
1. #### Clone the NetSci Repository:
    ```bash
    git clone https://github.com/netscianalysis/netsci.git
    ```

1. #### Navigate to the NetSci Root Directory:
    ```bash
    cd netsci
    ```

1. #### Create NetSci Conda Environment:
    ```bash
    conda env create -f netsci.yml
    ```
1. #### Activate NetSci Conda Environment:
    ```bash
    source activate netsci
    ```
   
1. #### Create CMake Build Directory:
    ```bash
    mkdir build
    ```
   
1. #### Set NetSci Root Directory Variable:
    ```bash
    NETSCI_ROOT=$(pwd)
    ```
   
1. #### Navigate to the CMake Build Directory:
    ```bash
    cd ${NETSCI_ROOT}/build
    ```

1. #### Compile CUDA Architecture Script:
    ```bash
    nvcc ${NETSCI_ROOT}/build_scripts/cuda_architecture.cu -o cuda_architecture
    ```
1. #### Set CUDA Architecture Variable:
    ```bash
    CUDA_ARCHITECTURE=$(./cuda_architecture)
    ```
1. #### Configure the Build with CMake:
    ```bash
    cmake .. -DCONDA_DIR=$CONDA_PREFIX -DCUDA_ARCHITECTURE=${CUDA_ARCHITECTURE}
    ```
1. #### Build NetSci:
    ```bash
    cmake --build . -j
    ```
1. #### Build NetSci Python Interface:
    ```bash
    make python
    ```
1. #### Test C++ and CUDA Backend:
    ```bash
    ctest
    ```
1. #### Run Python Interface Tests:
    ```bash
    cd ${NETSCI_ROOT}
    pytest
    ```

 </details>

---

# Libraries

- [CuArray](cuarray/README.md)
- [NetChem](netchem/README.md)

