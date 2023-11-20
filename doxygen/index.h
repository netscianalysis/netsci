/*!
\mainpage NetSci: A Toolkit for High Performance Scientific Network Analysis Computation

\section overview_sec Overview

NetSci is a specialized toolkit designed for advanced network analysis in computational sciences. Utilizing the
capabilities of modern GPUs, it offers a powerful and efficient solution for processing computationally demanding
network analysis metrics while delivering state-of-the-art performance.

\section install_sec Installation

NetSci is designed with a focus on ease of installation and long-term stability, ensuring compatibility with Linux
systems featuring CUDA-capable GPUs (compute capability 3.5 and above). It leverages well-supported core C++ and Python
libraries to maintain simplicity and reliability.

1. **Download Miniconda Installation Script**:
   \code
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   \endcode

2. **Execute the Installation Script**:
   \code
   bash Miniconda3-latest-Linux-x86_64.sh
   \endcode

3. **Update Environment Settings**:
   \code
   source ~/.bashrc
   \endcode

4. **Install Git with Conda**:
   \code
   conda install -c conda-forge git
   \endcode

5. **Clone the NetSci Repository**:
   \code
   git clone https://github.com/netscianalysis/netsci.git
   \endcode

6. **Navigate to the NetSci Root Directory**:
   \code
   cd netsci
   \endcode

7. **Create NetSci Conda Environment**:
   \code
   conda env create -f netsci.yml
   \endcode

8. **Activate NetSci Conda Environment**:
   \code
   conda activate netsci
   \endcode

9. **Create CMake Build Directory**:
   \code
   mkdir build
   \endcode

10. **Set NetSci Root Directory Variable**:
    \code
    NETSCI_ROOT=$(pwd)
    \endcode

11. **Navigate to the CMake Build Directory**:
    \code
    cd ${NETSCI_ROOT}/build
    \endcode

12. **Compile CUDA Architecture Script**:
    \code
    nvcc ${NETSCI_ROOT}/build_scripts/cuda_architecture.cu -o cuda_architecture
    \endcode

13. **Set CUDA Architecture Variable**:
    \code
    CUDA_ARCHITECTURE=$(./cuda_architecture)
    \endcode

14. **Configure the Build with CMake**:
    \code
    cmake .. -DCONDA_DIR=$CONDA_PREFIX -DCUDA_ARCHITECTURE=${CUDA_ARCHITECTURE}
    \endcode

15. **Build NetSci**:
    \code
    cmake --build . -j
    \endcode

16. **Build NetSci Python Interface**:
    \code
    make python
    \endcode

17. **Test C++ and CUDA Backend**:
    \code
    ctest
    \endcode

18. **Run Python Interface Tests**:
    \code
    cd ${NETSCI_ROOT}
    pytest
    \endcode
*/
