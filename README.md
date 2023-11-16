<center><h1>NetSci<br><small>A Toolkit for High Performance Scientific Network Analysis Computation</small></h1></center>

NetSci is a specialized toolkit designed for advanced network analysis in computational sciences. Utilizing the
capabilities of modern GPUs, it offers a powerful and efficient solution for processing computationally demanding
network analysis metrics, delivering exceptional performance.
<!-- TOC -->
* [Installation](#installation)
* [API Documentation](#api-documentation)
* [CuArray Class Documentation](#cuarray-class-documentation)
  * [Overview](#overview)
    * [`CuArray()` *C++*](#cuarray-c)
    * [`__init__()` *Python*](#init-python)
    * [`~CuArray()` *C++*](#cuarray-c-1)
    * [`__del__()` *Python*](#del-python)
    * [`CuArrayError init(int m, int n)` *C++*](#cuarrayerror-initint-m-int-n-c)
    * [`init(self, m: int, n: int) -> int` *Python*](#initself-m-int-n-int---int-python)
    * [`CuArrayError init(T *host, int m, int n)` *C++*](#cuarrayerror-initt-host-int-m-int-n-c)
    * [`init(self, host, m: int, n: int) -> int` *Python*](#initself-host-m-int-n-int---int-python)
    * [`CuArrayError fromCuArrayShallowCopy(CuArray<T> *cuArray, int start, int end, int m, int n)` *C++*](#cuarrayerror-fromcuarrayshallowcopycuarrayt-cuarray-int-start-int-end-int-m-int-n-c)
    * [`fromCuArrayShallowCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int` *Python*](#fromcuarrayshallowcopyself-cuarray-start-int-end-int-m-int-n-int---int-python)
    * [`CuArrayError fromCuArrayDeepCopy(CuArray<T> *cuArray, int start, int end, int m, int n)` *C++*](#cuarrayerror-fromcuarraydeepcopycuarrayt-cuarray-int-start-int-end-int-m-int-n-c)
    * [`fromCuArrayDeepCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int` *Python*](#fromcuarraydeepcopyself-cuarray-start-int-end-int-m-int-n-int---int-python)
    * [`int n() const` *C++*](#int-n-const-c)
    * [`n(self) -> int` *Python*](#nself---int-python)
    * [`int m() const` *C++*](#int-m-const-c)
    * [`m(self) -> int` *Python*](#mself---int-python)
    * [`int size() const` *C++*](#int-size-const-c)
    * [`size(self) -> int` *Python*](#sizeself---int-python)
    * [`size_t bytes() const` *C++*](#sizet-bytes-const-c)
    * [`bytes(self) -> int` *Python*](#bytesself---int-python)
    * [`T *&host()` *C++*](#t-host-c)
    * [`host(self)` *Python*](#hostself-python)
    * [`T *&device()` *C++*](#t-device-c)
    * [`device(self)` *Python*](#deviceself-python)
    * [`CuArrayError allocateHost()` *C++*](#cuarrayerror-allocatehost-c)
    * [`allocateHost(self) -> int` *Python*](#allocatehostself---int-python)
    * [`CuArrayError allocateDevice()` *C++*](#cuarrayerror-allocatedevice-c)
    * [`allocateDevice(self) -> int` *Python*](#allocatedeviceself---int-python)
    * [`CuArrayError allocatedHost() const` *C++*](#cuarrayerror-allocatedhost-const-c)
    * [`allocatedHost(self) -> int` *Python*](#allocatedhostself---int-python)
    * [`CuArrayError allocatedDevice() const` *C++*](#cuarrayerror-allocateddevice-const-c)
    * [`allocatedDevice(self) -> int` *Python*](#allocateddeviceself---int-python)
    * [`CuArrayError toDevice()` *C++*](#cuarrayerror-todevice-c)
    * [`toDevice(self) -> int` *Python*](#todeviceself---int-python)
    * [`CuArrayError toHost()` *C++*](#cuarrayerror-tohost-c)
    * [`toHost(self) -> int` *Python*](#tohostself---int-python)
    * [`CuArrayError deallocateHost()` *C++*](#cuarrayerror-deallocatehost-c)
    * [`deallocateHost(self) -> int` *Python*](#deallocatehostself---int-python)
    * [`CuArrayError deallocateDevice()` *C++*](#cuarrayerror-deallocatedevice-c)
    * [`deallocateDevice(self) -> int` *Python*](#deallocatedeviceself---int-python)
    * [`CuArrayError fromNumpy(T *NUMPY_ARRAY, int NUMPY_ARRAY_DIM1, int NUMPY_ARRAY_DIM2)` *C++*](#cuarrayerror-fromnumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2-c)
    * [`fromNumpy(self, numpy_array, dim1: int, dim2: int) -> int` *Python*](#fromnumpyself-numpyarray-dim1-int-dim2-int---int-python)
    * [`void toNumpy(T **NUMPY_ARRAY, int **NUMPY_ARRAY_DIM1, int **NUMPY_ARRAY_DIM2)` *C++*](#void-tonumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2-c)
    * [`toNumpy(self) -> (numpy_array, dim1: int, dim2: int)` *Python*](#tonumpyself---numpyarray-dim1-int-dim2-int-python)
    * [`T get(int i, int j) const` *C++*](#t-getint-i-int-j-const-c)
    * [`get(self, i: int, j: int) -> ElementType` *Python*](#getself-i-int-j-int---elementtype-python)
    * [`CuArrayError set(T value, int i, int j)` *C++*](#cuarrayerror-sett-value-int-i-int-j-c)
    * [`set(self, value: ElementType, i: int, j: int) -> int` *Python*](#setself-value-elementtype-i-int-j-int---int-python)
    * [`CuArrayError load(const std::string &fname)` *C++*](#cuarrayerror-loadconst-stdstring-fname-c)
    * [`load(self, filename: str) -> int` *Python*](#loadself-filename-str---int-python)
    * [`void save(const std::string &fname)` *C++*](#void-saveconst-stdstring-fname-c)
    * [`save(self, filename: str)` *Python*](#saveself-filename-str-python)
    * [`CuArray<T> *sort(int i)` *C++*](#cuarrayt-sortint-i-c)
    * [`sort(self, column_index: int) -> CuArray` *Python*](#sortself-columnindex-int---cuarray-python)
    * [`T &operator[](int i) const` *C++*](#t-operatorint-i-const-c)
    * [`__getitem__(self, index: int) -> ElementType` *Python*](#getitemself-index-int---elementtype-python)
    * [`int owner() const` *C++*](#int-owner-const-c)
    * [`owner(self) -> int` *Python*](#ownerself---int-python)
    * [`CuArray<int> *argsort(int i)` *C++*](#cuarrayint-argsortint-i-c)
    * [`argsort(self, column_index: int) -> CuArray` *Python*](#argsortself-columnindex-int---cuarray-python)
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

---

# CuArray Class Documentation

## Overview

The `CuArray` class is designed for managing arrays with CUDA support, providing methods for initialization, memory management, data manipulation, and utility operations.

---

### `CuArray()` *C++*
- **Description**: Default constructor. Constructs an empty `CuArray` object.

### `__init__()` *Python*
- **Description**: Default constructor. Constructs an empty `CuArray` object.

---

### `~CuArray()` *C++*
- **Description**: Destructor. Deallocates the memory on both the host and the device.

### `__del__()` *Python*
- **Description**: Destructor. Deallocates the memory on both the host and the device.

---

### `CuArrayError init(int m, int n)` *C++*
- **Description**: Initialize the `CuArray` with specified dimensions, allocating memory on both the host and the device.
- **Parameters**:
    - `int m`: Number of rows.
    - `int n`: Number of columns.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `init(self, m: int, n: int) -> int` *Python*
- **Description**: Initialize the `CuArray` with specified dimensions, allocating memory on both the host and the device.
- **Parameters**:
    - `m` (`int`): Number of rows.
    - `n` (`int`): Number of columns.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError init(T *host, int m, int n)` *C++*
- **Description**: Initialize with host data and dimensions, performing a shallow copy.
- **Parameters**:
    - `T *host`: Pointer to input host data.
    - `int m`: Number of rows.
    - `int n`: Number of columns.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `init(self, host, m: int, n: int) -> int` *Python*
- **Description**: Initialize with host data and dimensions, performing a shallow copy.
- **Parameters**:
    - `host`: Pointer to input host data.
    - `m` (`int`): Number of rows.
    - `n` (`int`): Number of columns.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError fromCuArrayShallowCopy(CuArray<T> *cuArray, int start, int end, int m, int n)` *C++*
- **Description**: Shallow copy data from another `CuArray`.
- **Parameters**:
    - `CuArray<T> *cuArray`: Source `CuArray`.
    - `int start`: Index of the first row to copy.
    - `int end`: Index of the last row to copy.
    - `int m`: Number of rows in this `CuArray`.
    - `int n`: Number of columns in this `CuArray`.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `fromCuArrayShallowCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int` *Python*
- **Description**: Shallow copy data from another `CuArray`.
- **Parameters**:
    - `cuArray`: Source `CuArray` object.
    - `start` (`int`): Index of the first row to copy.
    - `end` (`int`): Index of the last row to copy.
    - `m` (`int`): Number of rows in this `CuArray`.
    - `n` (`int`): Number of columns in this `CuArray`.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError fromCuArrayDeepCopy(CuArray<T> *cuArray, int start, int end, int m, int n)` *C++*
- **Description**: Deep copy data from another `CuArray`.
- **Parameters**:
    - `CuArray<T> *cuArray`: Source `CuArray`.
    - `int start`: Index of the first row to copy.
    - `int end`: Index of the last row to copy.
    - `int m`: Number of rows in this `CuArray`.
    - `int n`: Number of columns in this `CuArray`.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `fromCuArrayDeepCopy(self, cuArray, start: int, end: int, m: int, n: int) -> int` *Python*
- **Description**: Deep copy data from another `CuArray`.
- **Parameters**:
    - `cuArray`: Source `CuArray` object.
    - `start` (`int`): Index of the first row to copy.
    - `end` (`int`): Index of the last row to copy.
    - `m` (`int`): Number of rows in this `CuArray`.
    - `n` (`int`): Number of columns in this `CuArray`.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `int n() const` *C++*
- **Description**: Get the number of columns in the `CuArray`.
- **Returns**: Number of columns as `int`.

### `n(self) -> int` *Python*
- **Description**: Get the number of columns in the `CuArray`.
- **Returns**: Number of columns as `int`.

---

### `int m() const` *C++*
- **Description**: Get the number of rows in the `CuArray`.
- **Returns**: Number of rows as `int`.

### `m(self) -> int` *Python*
- **Description**: Get the number of rows in the `CuArray`.
- **Returns**: Number of rows as `int`.

---

### `int size() const` *C++*
- **Description**: Get the total number of elements in the `CuArray`.
- **Returns**: Total number of elements as `int`.

### `size(self) -> int` *Python*
- **Description**: Get the total number of elements in the `CuArray`.
- **Returns**: Total number of elements as `int`.

---

### `size_t bytes() const` *C++*
- **Description**: Get the total size in bytes of the `CuArray` data.
- **Returns**: Size in bytes as `size_t`.

### `bytes(self) -> int` *Python*
- **Description**: Get the total size in bytes of the `CuArray` data.
- **Returns**: Size in bytes as `int`.

---

### `T *&host()` *C++*
- **Description**: Get a reference to the host data.
- **Returns**: Reference to the host data as `T*&`.

### `host(self)` *Python*
- **Description**: Get a reference to the host data.
- **Returns**: Reference to the host data.

---

### `T *&device()` *C++*
- **Description**: Get a reference to the device data.
- **Returns**: Reference to the device data as `T*&`.

### `device(self)` *Python*
- **Description**: Get a reference to the device data.
- **Returns**: Reference to the device data.

---

### `CuArrayError allocateHost()` *C++*
- **Description**: Allocate memory for the host data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `allocateHost(self) -> int` *Python*
- **Description**: Allocate memory for the host data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError allocateDevice()` *C++*
- **Description**: Allocate memory for the device data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `allocateDevice(self) -> int` *Python*
- **Description**: Allocate memory for the device data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError allocatedHost() const` *C++*
- **Description**: Check if memory is allocated for the host data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `allocatedHost(self) -> int` *Python*
- **Description**: Check if memory is allocated for the host data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError allocatedDevice() const` *C++*
- **Description**: Check if memory is allocated for the device data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `allocatedDevice(self) -> int` *Python*
- **Description**: Check if memory is allocated for the device data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError toDevice()` *C++*
- **Description**: Copy data from the host to the device.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `toDevice(self) -> int` *Python*
- **Description**: Copy data from the host to the device.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError toHost()` *C++*
- **Description**: Copy data from the device to the host.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `toHost(self) -> int` *Python*
- **Description**: Copy data from the device to the host.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError deallocateHost()` *C++*
- **Description**: Deallocate memory for the host data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `deallocateHost(self) -> int` *Python*
- **Description**: Deallocate memory for the host data.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError deallocateDevice()` *C++*
- **Description**: Deallocate memory for the device data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `deallocateDevice(self) -> int` *Python*
- **Description**: Deallocate memory for the device data.
- **Returns**: `int`: `CuArrayError` indicating success (`0’) or specific error code.

---

### `CuArrayError fromNumpy(T *NUMPY_ARRAY, int NUMPY_ARRAY_DIM1, int NUMPY_ARRAY_DIM2)` *C++*
- **Description**: Copy data from a NumPy array to the `CuArray`.
- **Parameters**:
    - `T *NUMPY_ARRAY`: Pointer to the input NumPy array.
    - `int NUMPY_ARRAY_DIM1`: Dimension 1 of the NumPy array.
    - `int NUMPY_ARRAY_DIM2`: Dimension 2 of the NumPy array.
- **Returns**: `CuArrayError` indicating success (`0’) or specific error code.

### `fromNumpy(self, numpy_array, dim1: int, dim2: int) -> int` *Python*
- **Description**: Copy data from a NumPy array to the `CuArray`.
- **Parameters**:
    - `numpy_array`: NumPy array to copy from.
    - `dim1` (`int`): Dimension 1 of the NumPy array.
    - `dim2` (`int`): Dimension 2 of the NumPy array.
- **Returns**: `int`: `CuArrayError` indicating success (`0’) or specific error code.

---

### `void toNumpy(T **NUMPY_ARRAY, int **NUMPY_ARRAY_DIM1, int **NUMPY_ARRAY_DIM2)` *C++*
- **Description**: Copy data from the `CuArray` to a NumPy array.
- **Parameters**:
    - `T **NUMPY_ARRAY`: Pointer to the output NumPy array.
    - `int **NUMPY_ARRAY_DIM1`: Dimension 1 of the NumPy array.
    - `int **NUMPY_ARRAY_DIM2`: Dimension 2 of the NumPy array.

### `toNumpy(self) -> (numpy_array, dim1: int, dim2: int)` *Python*
- **Description**: Copy data from the `CuArray` to a NumPy array.
- **Returns**: Tuple containing the NumPy array and its dimensions.

---

### `T get(int i, int j) const` *C++*
- **Description**: Get the value at the specified position (i, j) in the `CuArray`.
- **Parameters**:
    - `int i`: Row index.
    - `int j`: Column index.
- **Returns**: Value at the specified position.

### `get(self, i: int, j: int) -> ElementType` *Python*
- **Description**: Get the value at the specified position (i, j) in the `CuArray`.
- **Parameters**:
    - `i` (`int`): Row index.
    - `j` (`int`): Column index.
- **Returns**: Value at the specified position.

---

### `CuArrayError set(T value, int i, int j)` *C++*
- **Description**: Set the value at the specified position (i, j) in the `CuArray`.
- **Parameters**:
    - `T value`: The value to set.
    - `int i`: Row index.
    - `int j`: Column index.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `set(self, value: ElementType, i: int, j: int) -> int` *Python*
- **Description**: Set the value at the specified position (i, j) in the `CuArray`.
- **Parameters**:
    - `value`: The value to set.
    - `i` (`int`): Row index.
    - `j` (`int`): Column index.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `CuArrayError load(const std::string &fname)` *C++*
- **Description**: Load the `CuArray` data from the specified file.
- **Parameters**:
    - `const std::string &fname`: Name of the file to load.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

### `load(self, filename: str) -> int` *Python*
- **Description**: Load the `CuArray` data from the specified file.
- **Parameters**:
    - `filename` (`str`): Name of the file to load.
- **Returns**: `int`: `CuArrayError` indicating success (`0`) or specific error code.

---

### `void save(const std::string &fname)` *C++*
- **Description**: Save the `CuArray` data to the specified file.
- **Parameters**:
    - `const std::string &fname`: Name of the file to save.

### `save(self, filename: str)` *Python*
- **Description**: Save the `CuArray` data to the specified file.
- **Parameters**:
    - `filename` (`str`): Name of the file to save.

---

### `CuArray<T> *sort(int i)` *C++*
- **Description**: Sort the `CuArray` in ascending order based on the values in the specified column.
- **Parameters**:
    - `int i`: Column index to sort.
- **Returns**: Pointer to a new `CuArray` containing the sorted data.

### `sort(self, column_index: int) -> CuArray` *Python*
- **Description**: Sort the `CuArray` in ascending order based on the values in the specified column.
- **Parameters**:
    - `column_index` (`int`): Column index to sort.
- **Returns**: New `CuArray` object containing sorted data.

---

### `T &operator[](int i) const` *C++*
- **Description**: Get a reference to the element at the specified index in the `CuArray`.
- **Parameters**:
    - `int i`: Index of the element.
- **Returns**: Reference to the element at the specified index.

### `__getitem__(self, index: int) -> ElementType` *Python*
- **Description**: Get the element at the specified index in the `CuArray`.
- **Parameters**:
    - `index` (`int`): Index of the element.
- **Returns**: Element at the specified index.

---

### `int owner() const` *C++*
- **Description**: Get the owner of the `CuArray`, which indicates whether the `CuArray` is responsible for memory deallocation.
- **Returns**: Owner of the `CuArray`.

### `owner(self) -> int` *Python*
- **Description**: Get the owner of the `CuArray`, which indicates whether the `CuArray` is responsible for memory deallocation.
- **Returns**: Owner of the `CuArray`.

---

### `CuArray<int> *argsort(int i)` *C++*
- **Description**: Perform an argsort on the specified column of the `CuArray` and return a new `CuArray` containing the sorted indices.
- **Parameters**:
    - `int i`: Column index to argsort.
- **Returns**: Pointer to a new `CuArray` containing the sorted indices.

### `argsort(self, column_index: int) -> CuArray` *Python*
- **Description**: Perform an argsort on the specified column of the `CuArray` and return a new `CuArray` containing the sorted indices.
- **Parameters**:
    - `column_index` (`int`): Column index to argsort.
- **Returns**: New `CuArray` object containing sorted indices.

---