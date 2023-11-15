# NetSci: A Toolkit for High Performance Scientific Network Analysis Computation

Netsci is a toolkit tailored for advanced network analysis in
the computational sciences' domain. Leveraging GPU acceleration, it
provides a robust solution for handling large datasets in dynamic
network analysis.

## Accelerating Network Analysis in Computational Sciences

---

In the fast-paced world of computational sciences, efficient data
analysis is crucial. Every day, vast amounts of data are generated,
demanding robust methods for analysis. Netsci, our GPU-accelerated
computational tool, steps in to meet this challenge, especially in the
realm of network analysis.

### The Importance of Network Analysis

---

Network analysis is an essential tool across various fields such as computational chemistry, bioinformatics, genomics, and machine learning. It
provides insights into the relationships and interactions between nodes in network representations of complex systems. A
key aspect of network analysis is the assessment of internode correlations, which help in understanding the degree of
connection or influence between nodes.

Traditional methods, like Pearson correlation, are widely used in this context. However, they have their limitations.
Primarily, Pearson correlation may not effectively detect nonlinear relationships, a common occurrence in complex
networks. As a result, there's a growing need to explore and develop more robust
techniques in network analysis to accommodate these challenges.

### A New Approach with Mutual Information (MI)

---

Netsci focuses on an alternative approach: Mutual Information (MI).
Originating from information theory, MI is a powerful metric for
quantifying relationships between variables, surpassing the limits of
Pearson correlation. It measures the average divergence between the
joint information content of two variables and their individual
information contents. Unlike Pearson correlation, MI has no upper bound
and is more versatile for different types of data distributions.

### Advancements in Estimating MI

---

Estimating MI accurately is challenging without knowing the data's
underlying distribution. Netsci addresses this through a k-nearest
neighbor approach, proven effective even in data-rich scenarios like
molecular dynamics (MD) simulations. This method is data-efficient,
adaptive, and minimally biased, making it ideal for various
applications.

### Netsci in Action

---

Netsci shines in its application to real-world computational challenges.
We demonstrate its capabilities in molecular dynamics network analysis,
specifically in correlation analyses where traditional methods like
Pearson correlation are insufficient. By comparing its performance with
CPU implementations, Netsci proves to be significantly faster and more
scalable, limited only by system memory constraints.

<hr style="border:2px solid rgb(128,128,128)">

## Installation

---

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

<hr style="border:2px solid rgb(128,128,128)">

## Python API Documentation

---

### CuArray

---
 ```python
   CuArray.CuArray()->None
  ```
  - **Description**: Constructs an empty CuArray object.

   - **Examples**:
       ```python
      from cuarray import FloatCuArray
      from cuarray import IntCuArray 
     
     """
     Initializes a float32 CuArray object.
     """
     float_cuarray = FloatCuArray()
     """
      Initializes an int32 CuArray object.
     """
       int_cuarray = IntCuArray()
       ```
     
   ---

```python
CuArray.init(
    m: int,
    n: int,
)->int
```
- **Description**: Initializes the CuArray object with the specified dimensions, 
where m is the number of rows and n is the number of columns.
- **Parameters**:
    - **m**: The number of rows.
    - **n**: The number of columns.
- **Returns**: 0 if successful, otherwise an error code.
- **Examples**:
    ```python
    from cuarray import FloatCuArray
    
    """
    Initializes a float32 CuArray object.
    """
    float_cuarray = FloatCuArray()
    """
    Initializes a float32 CuArray object with 10 rows and 10 columns.
    """
    float_cuarray.init(10, 10)
    ```



