<center><h1>NetSci<br><small>A Toolkit for High Performance Scientific Network Analysis Computation</small></h1></center>

NetSci is a specialized toolkit designed for advanced network analysis in computational sciences. Utilizing the
capabilities of modern GPUs, it offers a powerful and efficient solution for processing computationally demanding
network analysis metrics while delivering state-of-the-art performance.

---

* [Installation](#installation)
* [API Documentation](#api-documentation)

# Installation
NetSci is designed with a focus on ease of installation and long-term stability, ensuring compatibility with Linux
systems featuring CUDA-capable GPUs (compute capability 3.5 and above). It leverages well-supported core C++ and Python
libraries to maintain simplicity and reliability.
<details>

<summary>Steps</summary>

  * [Download Miniconda Installation Script](#download-miniconda-installation-script)
  * [Execute the Installation Script](#execute-the-installation-script)
  * [Update Environment Settings](#update-environment-settings)
  * [Install Git with Conda](#install-git-with-conda)
  * [Clone the NetSci Repository](#clone-the-netsci-repository)
  * [Navigate to the NetSci Root Directory](#navigate-to-the-netsci-root-directory)
  * [Set NetSci Root Directory Variable](#set-netsci-root-directory-variable)
  * [Create NetSci Conda Environment](#create-netsci-conda-environment)
  * [Activate NetSci Conda Environment](#activate-netsci-conda-environment)
  * [Prepare the Build Directory](#prepare-the-build-directory)
  * [Compile CUDA Script for GPU Capability](#compile-cuda-script-for-gpu-capability)
  * [Set CUDA Architecture Variable](#set-cuda-architecture-variable)
  * [Configure the Build with CMake](#configure-the-build-with-cmake)
  * [Build NetSci](#build-netsci)
  * [Build NetSci Python Interface](#build-netsci--interface)
  * [Test C++ and CUDA Backend](#test-c-and-cuda-backend)
  * [Run Python Interface Tests](#run--interface-tests)



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
1. #### Set NetSci Root Directory Variable:
    ```bash
    NETSCI_ROOT=$(pwd)
    ```

1. #### Create NetSci Conda Environment:
    ```bash
    conda env create -f netsci.yml
    ```
1. #### Activate NetSci Conda Environment:
    ```bash
    source activate netsci
    ```
1. #### Prepare the Build Directory:
    ```bash
    mkdir ${NETSCI_ROOT}/build
    cd ${NETSCI_ROOT}/build
    ```

1. #### Compile CUDA Script for GPU Capability:
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

# API Documentation
<details>

<summary>Libraries</summary>

- [CuArray](#cuarray)
- [NetChem](#netchem)
- [NetCalc](#netcalc)


---

# CuArray
  <details><summary>Classes</summary>

- [CuArray](#cuarray-class)

</details>




---

## CuArray ___class___

- **Languages**: C++, Python, Tcl 
- **Library**: [CuArray](#cuarray)

- <details><summary>Methods</summary>

  <details><summary>C++</summary>

  * [`CuArray()` ___constructor___](#cuarray-constructor)
  * [`~CuArray()` ___destructor___](#cuarray-destructor)
  * [`CuArrayError init(int m, int n)`](#cuarrayerror-initint-m-int-n)
  * [`CuArrayError init(T *host, int m, int n)`](#cuarrayerror-initt-host-int-m-int-n)
  * [`CuArrayError fromCuArrayShallowCopy(CuArray<T> *cuArray, int start, int end, int m, int n)`](#cuarrayerror-fromcuarrayshallowcopycuarrayt-cuarray-int-start-int-end-int-m-int-n)
  * [`CuArrayError fromCuArrayDeepCopy(CuArray<T> *cuArray, int start, int end, int m, int n)`](#cuarrayerror-fromcuarraydeepcopycuarrayt-cuarray-int-start-int-end-int-m-int-n)
  * [`int n() const`](#int-n-const)
  * [`int m() const`](#int-m-const)
  * [`int size() const`](#int-size-const)
  * [`size_t bytes() const`](#sizet-bytes-const)
  * [`T *&host()`](#t-host)
  * [`T *&device()`](#t-device)
  * [`CuArrayError allocateHost()`](#cuarrayerror-allocatehost)
  * [`CuArrayError allocateDevice()`](#cuarrayerror-allocatedevice)
  * [`CuArrayError allocatedHost() const`](#cuarrayerror-allocatedhost-const)
  * [`CuArrayError allocatedDevice() const`](#cuarrayerror-allocateddevice-const)
  * [`CuArrayError toDevice()`](#cuarrayerror-todevice)
  * [`CuArrayError toHost()`](#cuarrayerror-tohost)
  * [`CuArrayError deallocateHost()`](#cuarrayerror-deallocatehost)
  * [`CuArrayError deallocateDevice()`](#cuarrayerror-deallocatedevice)
  * [`CuArrayError fromNumpy(T *NUMPY_ARRAY, int NUMPY_ARRAY_DIM1, int NUMPY_ARRAY_DIM2)`](#cuarrayerror-fromnumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2)
  * [`void toNumpy(T **NUMPY_ARRAY, int **NUMPY_ARRAY_DIM1, int **NUMPY_ARRAY_DIM2)`](#void-tonumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2)
  * [`T get(int i, int j) const`](#t-getint-i-int-j-const)
  * [`CuArrayError set(T value, int i, int j)`](#cuarrayerror-sett-value-int-i-int-j)
  * [`CuArrayError load(const std::string &fname)`](#cuarrayerror-loadconst-stdstring-fname)
  * [`void save(const std::string &fname)`](#void-saveconst-stdstring-fname)
  * [`CuArray<T> *sort(int i)`](#cuarrayt-sortint-i)
  * [`T &operator[](int i) const`](#t-operatorint-i-const)
  * [`int owner() const`](#int-owner-const)
  * [`CuArray<int> *argsort(int i)`](#cuarrayint-argsortint-i)

  </details>

  <details><summary>Python</summary>

  * [`__init__()`](#init)
  * [`__del__()`](#__del__)
  * [`init(self, m: int, n: int) -> int`](#initself-m-int-n-int---int)
  * [`init(self, host, m: int, n: int) -> int`](#initself-host-m-int-n-int---int)
  * [`fromCuArrayShallowCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int`](#fromcuarrayshallowcopyself-cuarray-start-int-end-int-m-int-n-int---int)
  * [`fromCuArrayDeepCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int`](#fromcuarraydeepcopyself-cuarray-start-int-end-int-m-int-n-int---int)
  * [`m(self) -> int`](#mself---int)
  * [`n(self) -> int`](#nself---int)
  * [`size(self) -> int`](#sizeself---int)
  * [`bytes(self) -> int`](#bytesself---int)
  * [`host(self)`](#hostself)
  * [`device(self)`](#deviceself)
  * [`allocateHost(self) -> int`](#allocatehostself---int)
  * [`allocateDevice(self) -> int`](#allocatedeviceself---int)
  * [`allocatedHost(self) -> int`](#allocatedhostself---int)
  * [`allocatedDevice(self) -> int`](#allocateddeviceself---int)
  * [`toDevice(self) -> int`](#todeviceself---int)
  * [`toHost(self) -> int`](#tohostself---int)
  * [`deallocateHost(self) -> int`](#deallocatehostself---int)
  * [`deallocateDevice(self) -> int`](#deallocatedeviceself---int)
  * [`fromNumpy(self, numpy_array, dim1: int, dim2: int) -> int`](#fromnumpyself-numpyarray-dim1-int-dim2-int---int)
  * [`toNumpy(self) -> (numpy_array, dim1: int, dim2: int)`](#tonumpyself---numpyarray-dim1-int-dim2-int)
  * [`get(self, i: int, j: int) -> ElementType`](#getself-i-int-j-int---elementtype)
  * [`set(self, value: ElementType, i: int, j: int) -> int`](#setself-value-elementtype-i-int-j-int---int)
  * [`load(self, filename: str) -> int`](#loadself-filename-str---int)
  * [`save(self, filename: str)`](#saveself-filename-str)
  * [`sort(self, column_index: int) -> CuArray`](#sortself-columnindex-int---cuarray)
  * [`__getitem__(self, index: int) -> ElementType`](#getitemself-index-int---elementtype)
  * [`owner(self) -> int`](#ownerself---int)
  * [`argsort(self, column_index: int) -> CuArray`](#argsortself-columnindex-int---cuarray)

  </details>
  </details>

---

### Overview

The `CuArray` class is designed for managing arrays with CUDA support, providing methods for initialization, memory
management, data manipulation, and utility operations.

---

### C++ Methods

---

#### `CuArray()` ___constructor___

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Default constructor. Constructs an empty `CuArray` object.
- **Related**: [`__init__()` ](#__init__)

---

#### `~CuArray()` ___destructor___

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Destructor. Deallocates the memory on both the host and the device.
- **Related**: [`__del__()` ](#__del__)

---

#### `CuArrayError init(int m, int n)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Initialize the `CuArray` with specified dimensions, allocating memory on both the host and the
  device.
- **Parameters**:
    - `int m`: Number of rows.
    - `int n`: Number of columns.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`init(self, m: int, n: int) -> int` ](#initself-m-int-n-int---int)

---

#### `CuArrayError init(T *host, int m, int n)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Initialize with host data and dimensions, performing a shallow copy.
- **Parameters**:
    - `T *host`: Pointer to input host data.
    - `int m`: Number of rows.
    - `int n`: Number of columns.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`init(self, host, m: int, n: int) -> int` ](#initself-host-m-int-n-int---int)

---

#### `CuArrayError fromCuArrayShallowCopy(CuArray<T> *cuArray, int start, int end, int m, int n)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Shallow copy data from another `CuArray`.
- **Parameters**:
    - `CuArray<T> *cuArray`: Source `CuArray`.
    - `int start`: Index of the first row to copy.
    - `int end`: Index of the last row to copy.
    - `int m`: Number of rows in this `CuArray`.
    - `int n`: Number of columns in this `CuArray`.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related
  **: [`fromCuArrayShallowCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int` ](#fromcuarrayshallowcopyself-cuarray-start-int-end-int-m-int-n-int---int)

---

#### `CuArrayError fromCuArrayDeepCopy(CuArray<T> *cuArray, int start, int end, int m, int n)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Deep copy data from another `CuArray`.
- **Parameters**:
    - `CuArray<T> *cuArray`: Source `CuArray`.
    - `int start`: Index of the first row to copy.
    - `int end`: Index of the last row to copy.
    - `int m`: Number of rows in this `CuArray`.
    - `int n`: Number of columns in this `CuArray`.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related
  **: [`fromCuArrayDeepCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int` ](#fromcuarraydeepcopyself-cuarray-start-int-end-int-m-int-n-int---int)

---

#### `int n() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the number of columns in the `CuArray`.
- **Returns**: Number of columns as `int`.
- **Related**: [`n(self) -> int` ](#nself---int)

---

#### `int m() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the number of rows in the `CuArray`.
- **Returns**: Number of rows as `int`.
- **Related**: [`m(self) -> int` ](#mself---int)

---

#### `int size() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the total number of elements in the `CuArray`.
- **Returns**: Total number of elements as `int`.
- **Related**: [`size(self) -> int` ](#sizeself---int)

---

#### `size_t bytes() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the total size in bytes of the `CuArray` data.
- **Returns**: Size in bytes as `size_t`.
- **Related**: [`bytes(self) -> int` ](#bytesself---int)

---

#### `T *&host()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get a reference to the host data.
- **Returns**: Reference to the host data as `T*&`.
- **Related**: [`host(self)` ](#hostself)

---

#### `T *&device()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get a reference to the device data.
- **Returns**: Reference to the device data as `T*&`.
- **Related**: [`device(self)` ](#deviceself)

---

#### `CuArrayError allocateHost()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Allocate memory for the host data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`allocateHost(self) -> int` ](#allocatehostself---int)

---

#### `CuArrayError allocateDevice()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Allocate memory for the device data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`allocateDevice(self) -> int` ](#allocatedeviceself---int)

---

#### `CuArrayError allocatedHost() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Check if memory is allocated for the host data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`allocatedHost(self) -> int` ](#allocatedhostself---int)

---

#### `CuArrayError allocatedDevice() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Check if memory is allocated for the device data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`allocatedDevice(self) -> int` ](#allocateddeviceself---int)

---

#### `CuArrayError toDevice()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the host to the device.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`toDevice(self) -> int` ](#todeviceself---int)

---

#### `CuArrayError toHost()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the device to the host.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`toHost(self) -> int` ](#tohostself---int)

---

#### `CuArrayError deallocateHost()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Deallocate memory for the host data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`deallocateHost(self) -> int` ](#deallocatehostself---int)

---

#### `CuArrayError deallocateDevice()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Deallocate memory for the device data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`deallocateDevice(self) -> int` ](#deallocatedeviceself---int)

---

#### `CuArrayError fromNumpy(T *NUMPY_ARRAY, int NUMPY_ARRAY_DIM1, int NUMPY_ARRAY_DIM2)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from a NumPy array to the `CuArray`.
- **Parameters**:
    - `T *NUMPY_ARRAY`: Pointer to the input NumPy array.
    - `int NUMPY_ARRAY_DIM1`: Dimension 1 of the NumPy array.
    - `int NUMPY_ARRAY_DIM2`: Dimension 2 of the NumPy array.
- **Returns**: `CuArrayError` indicating success (`0’) or specific error code.
- **Related
  **: [`fromNumpy(self, numpy_array, dim1: int, dim2: int) -> int` ](#fromnumpyself-numpyarray-dim1-int-dim2-int---int)

---

#### `void toNumpy(T **NUMPY_ARRAY, int **NUMPY_ARRAY_DIM1, int **NUMPY_ARRAY_DIM2)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the `CuArray` to a NumPy array.
- **Parameters**:
    - `T **NUMPY_ARRAY`: Pointer to the output NumPy array.
    - `int **NUMPY_ARRAY_DIM1`: Dimension 1 of the NumPy array.
    - `int **NUMPY_ARRAY_DIM2`: Dimension 2 of the NumPy array.
- **Related**: [`toNumpy(self) -> (numpy_array, dim1: int, dim2: int)` ](#tonumpyself---numpyarray-dim1-int-dim2-int)

---

#### `T get(int i, int j) const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the value at the specified position (i, j) in the `CuArray`.
- **Parameters**:
    - `int i`: Row index.
    - `int j`: Column index.
- **Returns**: Value at the specified position.
- **Related**: [`get(self, i: int, j: int) -> ElementType` ](#getself-i-int-j-int---elementtype)

---

#### `CuArrayError set(T value, int i, int j)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Set the value at the specified position (i, j) in the `CuArray`.
- **Parameters**:
    - `T value`: The value to set.
    - `int i`: Row index.
    - `int j`: Column index.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`set(self, value: ElementType, i: int, j: int) -> int` ](#setself-value-elementtype-i-int-j-int---int)

---

#### `CuArrayError load(const std::string &fname)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Load the `CuArray` data from the specified file.
- **Parameters**:
    - `const std::string &fname`: Name of the file to load.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`load(self, filename: str) -> int` ](#loadself-filename-str---int)

---

#### `void save(const std::string &fname)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Save the `CuArray` data to the specified file.
- **Parameters**:
    - `const std::string &fname`: Name of the file to save.
- **Related**: [`save(self, filename: str)` ](#saveself-filename-str)

---

#### `CuArray<T> *sort(int i)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Sort the `CuArray` in ascending order based on the values in the specified column.
- **Parameters**:
    - `int i`: Column index to sort.
- **Returns**: Pointer to a new `CuArray` containing the sorted data.
- **Related**: [`sort(self, column_index: int) -> CuArray` ](#sortself-columnindex-int---cuarray)

---

#### `T &operator[](int i) const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get a reference to the element at the specified index in the `CuArray`.
- **Parameters**:
    - `int i`: Index of the element.
- **Returns**: Reference to the element at the specified index.
- **Related**: [`__getitem__(self, index: int) -> ElementType` ](#getitemself-index-int---elementtype)

---

#### `int owner() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the owner of the `CuArray`, which indicates whether the `CuArray` is responsible for memory
  deallocation.
- **Returns**: Owner of the `CuArray`.
- **Related**: [`owner(self) -> int` ](#ownerself---int)

---

#### `CuArray<int> *argsort(int i)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Perform an argsort on the specified column of the `CuArray` and return a new `CuArray` containing the
  sorted indices.
- **Parameters**:
    - `int i`: Column index to argsort.
- **Returns**: Pointer to a new `CuArray` containing the sorted indices.
- **Related**: [`argsort(self, column_index: int) -> CuArray` ](#argsortself-columnindex-int---cuarray)

---

### Python Methods

#### `__init__()`

- **Language**: Python 
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Default constructor. Constructs an empty `CuArray` object.
- **Class**: [CuArray](#cuarray-class)
- **Related**: [`CuArray()` ](#cuarray-constructor)

---

#### `__del__()`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Destructor. Deallocates the memory on both the host and the device.
- **Class**: [CuArray](#cuarray-class)
- **Related**: [`~CuArray()` ](#cuarray-destructor)

---

#### `init(self, m: int, n: int) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Initialize the `CuArray` with specified dimensions, allocating memory on both the host and the
  device.
- **Parameters**:
    - `m` (`int`): Number of rows.
    - `n` (`int`): Number of columns.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError init(int m, int n)` ](#cuarrayerror-initint-m-int-n)

---

#### `init(self, host, m: int, n: int) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Initialize with host data and dimensions, performing a shallow copy.
- **Parameters**:
    - `host`: Pointer to input host data.
    - `m` (`int`): Number of rows.
    - `n` (`int`): Number of columns.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError init(T *host, int m, int n)` ](#cuarrayerror-initt-host-int-m-int-n)

---

#### `fromCuArrayShallowCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Shallow copy data from another `CuArray`.
- **Parameters**:
    - `cuArray`: Source `CuArray` object.
    - `start` (`int`): Index of the first row to copy.
    - `end` (`int`): Index of the last row to copy.
    - `m` (`int`): Number of rows in this `CuArray`.
    - `n` (`int`): Number of columns in this `CuArray`.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related
  **: [`CuArrayError fromCuArrayShallowCopy(CuArray<T> *cuArray, int start, int end, int m, int n)` ](#cuarrayerror-fromcuarrayshallowcopycuarrayt-cuarray-int-start-int-end-int-m-int-n)

---

#### `fromCuArrayDeepCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Deep copy data from another `CuArray`.
- **Parameters**:
    - `cuArray`: Source `CuArray` object.
    - `start` (`int`): Index of the first row to copy.
    - `end` (`int`): Index of the last row to copy.
    - `m` (`int`): Number of rows in this `CuArray`.
    - `n` (`int`): Number of columns in this `CuArray`.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related
  **: [`CuArrayError fromCuArrayDeepCopy(CuArray<T> *cuArray, int start, int end, int m, int n)` ](#cuarrayerror-fromcuarraydeepcopycuarrayt-cuarray-int-start-int-end-int-m-int-n)

---

#### `m(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the number of rows in the `CuArray`.
- **Returns**: Number of rows as `int`.
- **Related**: [`int m() const` ](#int-n-const)

---

#### `n(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the number of columns in the `CuArray`.
- **Returns**: Number of columns as `int`.
- **Related**: [`int n() const` ](#int-n-const)

#### `size(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the total number of elements in the `CuArray`.
- **Returns**: Total number of elements as `int`.
- **Related**: [`int size() const` ](#int-size-const)

---

#### `bytes(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the total size in bytes of the `CuArray` data.
- **Returns**: Size in bytes as `int`.
- **Related**: [`size_t bytes() const` ](#sizet-bytes-const)

---

#### `host(self)`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get a reference to the host data.
- **Returns**: Reference to the host data.
- **Related**: [`T *&host()` ](#t-host)

---

#### `device(self)`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get a reference to the device data.
- **Returns**: Reference to the device data.
- **Related**: [`T *&device()` ](#t-device)

---

#### `allocateHost(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Allocate memory for the host data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError allocateHost()` ](#cuarrayerror-allocatehost)

---

#### `allocateDevice(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Allocate memory for the device data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError allocateDevice()` ](#cuarrayerror-allocatedevice)

---

#### `allocatedHost(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Check if memory is allocated for the host data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError allocatedHost() const` ](#cuarrayerror-allocatedhost-const)

---

#### `allocatedDevice(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Check if memory is allocated for the device data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError allocatedDevice() const` ](#cuarrayerror-allocateddevice-const)

---

#### `toDevice(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the host to the device.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError toDevice()` ](#cuarrayerror-todevice)

---

#### `toHost(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the device to the host.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError toHost()` ](#cuarrayerror-tohost)

---

#### `deallocateHost(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Deallocate memory for the host data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError deallocateHost()` ](#cuarrayerror-deallocatehost)

---

#### `deallocateDevice(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Deallocate memory for the device data.
- **Returns**: `int`: `CuArrayError` indicating success (`0’) or specific error code.
- **Related**: [`CuArrayError deallocateDevice()` ](#cuarrayerror-deallocatedevice)

---

#### `fromNumpy(self, numpy_array, dim1: int, dim2: int) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from a NumPy array to the `CuArray`.
- **Parameters**:
    - `numpy_array`: NumPy array to copy from.
    - `dim1` (`int`): Dimension 1 of the NumPy array.
    - `dim2` (`int`): Dimension 2 of the NumPy array.
- **Returns**: `int`: `CuArrayError` indicating success (`0’) or specific error code.
- **Related
  **: [`CuArrayError fromNumpy(T *NUMPY_ARRAY, int NUMPY_ARRAY_DIM1, int NUMPY_ARRAY_DIM2)` ](#cuarrayerror-fromnumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2)

---

#### `toNumpy(self) -> (numpy_array, dim1: int, dim2: int)`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the `CuArray` to a NumPy array.
- **Returns**: Tuple containing the NumPy array and its dimensions.
- **Related
  **: [`void toNumpy(T **NUMPY_ARRAY, int **NUMPY_ARRAY_DIM1, int **NUMPY_ARRAY_DIM2)` ](#void-tonumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2)

---

#### `get(self, i: int, j: int) -> ElementType`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the value at the specified position (i, j) in the `CuArray`.
- **Parameters**:
    - `i` (`int`): Row index.
    - `j` (`int`): Column index.
- **Returns**: Value at the specified position.
- **Related**: [`T get(int i, int j) const` ](#t-getint-i-int-j-const)

---

#### `set(self, value: ElementType, i: int, j: int) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Set the value at the specified position (i, j) in the `CuArray`.
- **Parameters**:
    - `value`: The value to set.
    - `i` (`int`): Row index.
    - `j` (`int`): Column index.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError set(T value, int i, int j)` ](#cuarrayerror-sett-value-int-i-int-j)

---

#### `load(self, filename: str) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Load the `CuArray` data from the specified file.
- **Parameters**:
    - `filename` (`str`): Name of the file to load.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`CuArrayError load(const std::string &fname)` ](#cuarrayerror-loadconst-stdstring-fname)

---

#### `save(self, filename: str)`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Save the `CuArray` data to the specified file.
- **Parameters**:
    - `filename` (`str`): Name of the file to save.
- **Related**: [`void save(const std::string &fname)` ](#void-saveconst-stdstring-fname)

---

#### `sort(self, column_index: int) -> CuArray`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Sort the `CuArray` in ascending order based on the values in the specified column.
- **Parameters**:
    - `column_index` (`int`): Column index to sort.
- **Returns**: New `CuArray` object containing sorted data.
- **Related**: [`CuArray<T> *sort(int i)` ](#cuarrayt-sortint-i)

---

#### `__getitem__(self, index: int) -> ElementType`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the element at the specified index in the `CuArray`.
- **Parameters**:
    - `index` (`int`): Index of the element.
- **Returns**: Element at the specified index.
- **Related**: [`T &operator[](int i) const` ](#t-operatorint-i-const)

---

#### `owner(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the owner of the `CuArray`, which indicates whether the `CuArray` is responsible for memory
  deallocation.
- **Returns**: Owner of the `CuArray`.
- **Related**: [`int owner() const` ](#int-owner-const)

---

#### `argsort(self, column_index: int) -> CuArray`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Perform an argsort on the specified column of the `CuArray` and return a new `CuArray` containing the
  sorted indices.
- **Parameters**:
    - `column_index` (`int`): Column index to argsort.
- **Returns**: New `CuArray` object containing sorted indices.
- **Related**: [`CuArray<int> *argsort(int i)` ](#cuarrayint-argsortint-i)

---

## NetChem

 ---

## NetCalc

---

</details>
