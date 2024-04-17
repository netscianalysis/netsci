Installation
============

At this time, NetSci has only been tested on Linux systems. Therefore, all
installation instructions are for Linux only.

NetSci is designed with a focus on ease of installation and long-term stability, ensuring compatibility with Linux systems featuring CUDA-capable GPUs (compute capability 3.5 and above). It leverages well-supported core C++ and Python libraries to maintain simplicity and reliability.

Download Miniconda Installation Script:

``wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``

Execute the Installation Script:

``bash Miniconda3-latest-Linux-x86_64.sh``

Update Environment Settings:

``source ~/.bashrc``

Install Git with Conda:

``conda install -c conda-forge git``

Clone the NetSci Repository:

``git clone https://github.com/netscianalysis/netsci.git``

Navigate to the NetSci Root Directory:

``cd netsci``

Create NetSci Conda Environment:

``conda env create -f netsci.yml``

Activate NetSci Conda Environment:

``conda activate netsci``

Create CMake Build Directory:

``mkdir build``

Set NetSci Root Directory Variable:

``NETSCI_ROOT=$(pwd)``

Navigate to the CMake Build Directory:

``cd ${NETSCI_ROOT}/build``

Compile CUDA Architecture Script:

``nvcc ${NETSCI_ROOT}/build_scripts/cuda_architecture.cu -o cuda_architecture``

Set CUDA Architecture Variable:

``CUDA_ARCHITECTURE=$(./cuda_architecture)``

Configure the Build with CMake:

``cmake .. -DCONDA_DIR=$CONDA_PREFIX -DCUDA_ARCHITECTURE=${CUDA_ARCHITECTURE}``

Build NetSci:

``cmake --build . -j``

Build NetSci Python Interface:

``make python``

Test C++ and CUDA Backend:

``ctest``

Run Python Interface Tests::

  cd ${NETSCI_ROOT}
  pytest

