<center><h1>NetSci<br><small>A Toolkit for High Performance Scientific Network Analysis Computation</small></h1></center>

---

<details><summary><b>Table of Contents</b></summary>

* [Installation](#installation)
* [API Documentation](#api-documentation)

</details>

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

# API Documentation

<details>
<summary><b>Libraries</b></summary>

- [CuArray](#cuarray)
- [NetChem](#netchem)
- [NetCalc](#netcalc)

---

# CuArray

  <details><summary><b>Classe</b>s</summary>

- [CuArray](#cuarray-class)

</details>




---

## CuArray ___class___

- **Languages**: C++, Python, Tcl
- **Library**: [CuArray](#cuarray)
- **Description**: The `CuArray` class is designed for managing arrays with CUDA support, providing methods for
  initialization, memory management, data manipulation, and utility operations. It is implemented as a template class
  in C++, and has wrapper interfaces for Python and Tcl. However, since these languages do not support templates, the
  CuArray class must be imported as `<ElementType>CuArray`, where `<ElementType>` is the 
  CuArray templated data type, which is always capitalized. As of now, only `float` and `int` types are supported in Python and Tcl, while all numeric data 
   types are supported in C++.

- <details><summary><b>Methods</b></summary>

  <details><summary><b>C++</b></summary>

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

  <details><summary><b>Python</b></summary>

    * [`__init__()`](#__init__)
    * [`init(self, m: int, n: int) -> int`](#initself-m-int-n-int---int)
    * [`fromCuArray(self, cuArray, start: int, end: int, m: int, n: int) -> int`](#fromcuarrayself-cuarray-start-int-end-int-m-int-n-int---int)
    * [`m(self) -> int`](#mself---int)
    * [`n(self) -> int`](#nself---int)
    * [`size(self) -> int`](#sizeself---int)
    * [`fromNumpy1D(self, numpy_array: numpy.ndarray) -> int`](#fromnumpy1dself-numpyarray-numpyndarray---int)
    * [`fromNumpy2D(self, numpy_array: numpy.ndarray) -> int`](#fromnumpy2dself-numpyarray-numpyndarray---int)
    * [`toNumpy1D(self) -> numpy.ndarray`](#tonumpy1dself---numpyndarray)
    * [`toNumpy2D(self) -> numpy.ndarray`](#tonumpy2dself---numpyndarray)
    * [`get(self, i: int, j: int) -> ElementType`](#getself-i-int-j-int---elementtype)
    * [`set(self, value: ElementType, i: int, j: int) -> int`](#setself-value-elementtype-i-int-j-int---int)
    * [`load(self, filename: str) -> int`](#loadself-filename-str---int)
    * [`save(self, filename: str)`](#saveself-filename-str)
    * [`sort(self, i: int) -> CuArray`](#sortself-i-int---cuarray)
    * [`__getitem__(self, index: int) -> Union[CuArray, ElementType]`](#__getitem__self-index-int---unionelementtype-cuarray)
    * [`argsort(self, i: int) -> CuArray`](#argsortself-i-int---cuarray)

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

<details><summary><b>Example</b></summary>

  ```cpp
#include <cuarray.h>
  
  /* Creates a new float CuArray instance */
  CuArray<float> *cuArray = new CuArray<float>();
  
  delete cuArray;
  ```

</details>



---

#### `~CuArray()` ___destructor___

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Destructor. Deallocates the memory on both the host and the device.

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

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* 
 * Initializes the CuArray with 10 rows and 5 columns 
 * and allocates memory on host.
 */
cuArray->init(10, 5);

delete cuArray;
```

</details>

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

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* 
 * Initializes the CuArray with 10 rows and 5 columns 
 * and allocates memory on host.
 */
cuArray->init(10, 5);

/* Create a 50-element float vector and fill it with random values */
auto a = new float[50];
for (int i = 0; i < 50; i++) {
    a[i] = static_cast<float>(rand() / (float) RAND_MAX);
}

/* Initialize the CuArray with data from "a", preserving 
 * overall size while setting new dimensions 
 * (similar to NumPy's reshape method). */
cuArray->init(a, 10, 5);

/* Print each element in cuArray's host memory.
 * The host data is linear and stored in row major order. To
 * access element i,j you would use the linear index
 * i*n+j, where n is the number of columns.*/
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        std::cout << cuArray->get(i, j) << " ";
        std::cout << a[i * cuArray->n() + j] << std::endl;
    }
    std::cout << std::endl;
}

/* Delete "a" and cuArray */
delete[] a;
delete cuArray;

```

</details>

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

<details><summary><b>Example</b></summary>

  ```cpp
#include <cuarray.h>
#include <iostream>

/* Create a new float CuArray instance */
auto cuArray = new CuArray<float>;

/* Initialize the CuArray with 3 rows and 3 columns */
cuArray->init(3, 3);

/*Set each i, j element equal to i*3 + j */
for (int i = 0; i < 9; i++) {
    cuArray->host()[i] = i;
}

/*
 * Create a float 'CuArray' that 
 * will be a shallow copy of the last two cuArray rows
 */
auto cuArray2x3Copy = new CuArray<float>;
cuArray2x3Copy->init(2, 3);

/* First row to copy from cuArray into cuArray2x3Copy */
int startRowIndex = 1;

/* Last row to copy from cuArray into cuArray2x3Copy */
int endRowIndex = 2;

cuArray2x3Copy->fromCuArrayShallowCopy(
        cuArray, /* Source for copying data into cuArray2x3Copy. 
            * Both cuArray and cuArray2x3Copy will point to the same 
            * data, which helps with
            * performance at the expense of being extremely dangerous. As an
            * attempt to make this method somewhat safe, there is an "owner"
            * attribute that is set to 1 if the CuArray owns the data and 0
            * otherwise. Logic is implemented in the destructor to check for ownership
            * and only delete data if the CuArray owns the data. As of now, this method has 
            * passed all real life stress tests, and CUDA-MEMCHECK doesn't hate it,
            * but it still shouldn't be used in the vast majority of cases.
            * The legitimate reason this should ever be called is when you have to 
            * pass the CuArray data as a double pointer to a function that 
            * cannot itself take a CuArray object. Eg.) A CUDA kernel.*/
        startRowIndex, /* First row to copy from cuArray into cuArray2x3Copy */
        endRowIndex, /* Last row to copy from cuArray into cuArray2x3Copy */
        cuArray2x3Copy->m(), /* Number of rows in cuArray2x3Copy */
        cuArray2x3Copy->n() /* Number of columns in cuArray2x3Copy */
        );

/* Print each element in cuArray2x3Copy */
for (int i = 0; i < cuArray2x3Copy->m(); i++) {
    for (int j = 0; j < cuArray2x3Copy->n(); j++) {
        std::cout << cuArray2x3Copy->get(i, j) << " ";
    }
    std::cout << std::endl;
}
/* Output: 
 * 3 4 5
 * 6 7 8
 */
delete cuArray2x3Copy;
delete cuArray;

  ```

 </details>

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
- **Related**: [`fromCuArray(self, cuArray, start: int, end: int, m: int, n: int) -> int` ](#fromcuarrayself-cuarray-start-int-end-int-m-int-n-int---int)

<details><summary><b>Example</b></summary>

  ```cpp
#include <cuarray.h>
#include <iostream>
#include <cuarray.h>
#include <iostream>

/* Create a new float CuArray instance */
auto cuArray = new CuArray<float>;

/* Initialize the CuArray with 3 rows and 3 columns */
cuArray->init(3, 3);

/*Set each i, j element equal to i*3 + j */
for (int i = 0; i < 9; i++) {
    cuArray->host()[i] = i;
}

/*
 * Create a float 'CuArray' that 
 * will be a deep copy of the last two cuArray rows
 */
auto cuArray2x3Copy = new CuArray<float>;
cuArray2x3Copy->init(2, 3);

/* First row to copy from cuArray into cuArray2x3Copy */
int startRowIndex = 1;

/* Last row to copy from cuArray into cuArray2x3Copy */
int endRowIndex = 2;

cuArray2x3Copy->fromCuArrayDeepCopy(
        cuArray, /*Source for copying data into cuArray2x3Copy. This method is 
            * significantly safer than its shallow copy equivalent. However, it is also 
            * slower, which can impact performance if it's called a lot.*/ 
        startRowIndex, /* First row to copy from cuArray into cuArray2x3Copy */
        endRowIndex, /* Last row to copy from cuArray into cuArray2x3Copy */
        cuArray2x3Copy->m(), /* Number of rows in cuArray2x3Copy */
        cuArray2x3Copy->n() /* Number of columns in cuArray2x3Copy */
        );

/* Print each element in cuArray2x3Copy */
for (int i = 0; i < cuArray2x3Copy->m(); i++) {
    for (int j = 0; j < cuArray2x3Copy->n(); j++) {
        std::cout << cuArray2x3Copy->get(i, j) << " ";
    }
    std::cout << std::endl;
}
/* Output: 
 * 3 4 5
 * 6 7 8
 */

 /* Both cuArray and cuArray2x3Copy own their data.*/
std::cout
<< cuArray->owner() << " "
<< cuArray2x3Copy->owner()
<< std::endl;
/* Output: 
 * 1 1
 */

delete cuArray2x3Copy;
delete cuArray;

```

</details>

---

#### `int n() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the number of columns in the `CuArray`.
- **Returns**: Number of columns as `int`.
- **Related**: [`n(self) -> int` ](#nself---int)

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <iostream>
    
  /* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* 
 * Initializes the CuArray with 10 rows and 5 columns 
 * and allocates memory on host.
 */
cuArray->init(10, 5);

/* Get the number of columns in the CuArray */
int n = cuArray->n();

/* Print the number of columns */
std::cout 
<< "Number of columns: "
<< n
<< std::endl;
/* Output: 
 * Number of columns: 5
 */

delete cuArray;
```

</details>

---

#### `int m() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the number of rows in the `CuArray`.
- **Returns**: Number of rows as `int`.
- **Related**: [`m(self) -> int` ](#mself---int)

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h
#include <iostream>
    
/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* 
 * Initializes the CuArray with 10 rows and 5 columns 
 * and allocates memory on host.
 */
cuArray->init(10, 5);

/* Get the number of rows in the CuArray */
int n = cuArray->n();

/* Print the number of rows */
std::cout 
<< "Number of rows: "
<< m
<< std::endl;
/* Output: 
 * Number of rows: 10
 */

delete cuArray;
 ```

</details>

---

#### `int size() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the total number of elements in the `CuArray`.
- **Returns**: Total number of elements as `int`.
- **Related**: [`size(self) -> int` ](#sizeself---int)

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <iostream>
  
/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* 
 * Initializes the CuArray with 10 rows and 5 columns 
 * and allocates memory on host.
 */
cuArray->init(10, 5);

/* Get the total number of values in the CuArray */
int size = cuArray->size();

/* Print the total number of values in cuArray. */
std::cout 
<< "Number of values: "
<< size
<< std::endl;
/* Output: 
 * Number of values: 50
 */

delete cuArray;
 ```

</details>

---

#### `size_t bytes() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the total size in bytes of the `CuArray` data.
- **Returns**: Size in bytes as `size_t`.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <iostream>
    
/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* 
 * Initializes the CuArray with 10 rows and 5 columns 
 * and allocates memory on host.
 */
cuArray->init(10, 5);

/* Get the number of bytes the CuArray data occupies */ 
auto bytes_ = cuArray->bytes();

/* Print the total number of bytes in cuArray. */
std::cout 
<< "Number of bytes: "
<< bytes_
<< std::endl;
/* Output: 
 * Number of bytes: 200 
 */

delete cuArray;
 ```

</details>

---

#### `T *&host()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get a reference to the host data.
- **Returns**: Reference to the host data as `T*&`.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <iostream>
    
/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 3 rows and 3 columns */
cuArray->init(3, 3);

/*Set each i, j element equal to i*3 + j */
for (int i = 0; i < 9; i++) {
    cuArray->host()[i] = i;
}

/* Print each element in cuArray's host memory.
 * The host data is linear and stored in row major order. To
 * access element i,j you would use the linear index
 * i*n+j, where n is the number of columns.*/
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        std::cout << cuArray->host()[i * cuArray->n() + j] << " ";
    }
    std::cout << std::endl;
}
/* Output: 
 * 0 1 2
 * 3 4 5
 * 6 7 8
 */

delete cuArray;
 ```

</details>

---

#### `T *&device()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get a reference to the device data.
- **Returns**: Reference to the device data as `T*&`.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <iostream>
    
/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 3 rows and 3 columns */
cuArray->init(3, 3);

/*Set each i, j element equal to i*3 + j */
for (int i = 0; i < 9; i++) {
    cuArray->host()[i] = i;
}

/* Allocate device memory. */
cuArray->allocateDevice();

/* Copy data from host to device. */
cuArray->toDevice();

/* Set deviceArray equal to cuArray's device data via the 
 * device() method, */
auto deviceArray = cuArray->device();
/* which can be used in CUDA kernels. 
 * Eg.) <<<1, 1>>>kernel(deviceArray)*/


/* delete frees both host and device memory. */
delete cuArray;
 ```

</details>

---

#### `CuArrayError allocateHost()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Allocate memory for the host data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Fill the CuArray with random values */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        cuArray->host()[i * cuArray->n() + j] =
                static_cast<float>(rand() / (float) RAND_MAX);
    }
}
/* Allocate device memory. */
cuArray->allocateDevice();

/* Copy data from host to device. */
cuArray->toDevice();

/* Free host memory, since it is no longer needed.*/
cuArray->deallocateHost();

/*Do some complicated GPU calculations
* and then allocate host memory when you need it again.
* Also, this is extremely wasteful, it's just an example of
* how to use this method. Realistically, most users will never have
* to manually allocate host memory as that is handled by the
* init methods.*/
cuArray->allocateHost();

/* Copy data from device to host. */
cuArray->toHost();

delete cuArray;
 ```

</details>

---

#### `CuArrayError allocateDevice()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Allocate memory for the device data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Fill the CuArray with random values */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        cuArray->host()[i * cuArray->n() + j] =
                static_cast<float>(rand() / (float) RAND_MAX);
    }
}
/* Allocate device memory. */
cuArray->allocateDevice();

/* Frees host and device memory. */
delete cuArray;
 ```

</details>

---

#### `CuArrayError allocatedHost() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Check if memory is allocated for the host data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Check if host memory is allocated. If it is,
 * allocatedHost() will return 1, other wise it 
 * will return 0. This is convenient for boolean checks.*/
auto hostMemoryAllocated = cuArray->allocatedHost();

/* Print whether or not host memory is allocated. */
std::cout
<< "Host memory allocated: "
<< hostMemoryAllocated
<< std::endl;

delete cuArray;
 ```

</details>

---

#### `CuArrayError allocatedDevice() const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Check if memory is allocated for the device data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Allocate device memory. */
cuArray->allocateDevice();

/* Check if device memory is allocated. If it is,
 * allocatedDevice() will return 1, other wise it 
 * will return 0. This is convenient for boolean checks.*/
auto deviceMemoryAllocated = cuArray->allocatedDevice();

/* Print whether or not device memory is allocated. */
std::cout
<< "Device memory allocated: "
<< deviceMemoryAllocated
<< std::endl;

delete cuArray;
 ```

</details>

---

#### `CuArrayError toDevice()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the host to the device.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Fill the CuArray with random values */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        cuArray->host()[i * cuArray->n() + j] =
                static_cast<float>(rand() / (float) RAND_MAX);
    }
}
/* Allocate device memory. */
cuArray->allocateDevice();

/* Copy data from host to device. */
cuArray->toDevice();

/* Frees host and device memory. */
delete cuArray;
 ```

</details>

---

#### `CuArrayError toHost()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the device to the host.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Fill the CuArray with random values */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        cuArray->host()[i * cuArray->n() + j] =
                static_cast<float>(rand() / (float) RAND_MAX);
    }
}
/* Allocate device memory. */
cuArray->allocateDevice();

/* Copy data from host to device. */
cuArray->toDevice();

/* Set the number of threads per block to 1024 */
auto threadsPerBlock = 1024;

/* Set the number of blocks to the ceiling of the number of elements
 * divided by the number of threads per block. */
auto blocksPerGrid = (cuArray->size() + threadsPerBlock - 1) / threadsPerBlock;

/* Launch a CUDA kernel that does something cool and only takes
 * a single float array as an argument
 *<<<blocksPerGrid, threadsPerBlock>>>kernel(cuArray->device()); */ 

/* Copy data from device to host. */
cuArray->toHost();

/* Frees host and device memory. */
delete cuArray;
 ```

</details>

---

#### `CuArrayError deallocateHost()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Deallocate memory for the host data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Fill the CuArray with random values */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        cuArray->host()[i * cuArray->n() + j] =
                static_cast<float>(rand() / (float) RAND_MAX);
    }
}
/* Allocate device memory. */
cuArray->allocateDevice();

/* Copy data from host to device. */
cuArray->toDevice();

/* Deallocate the host array to reduce memory usage if it's not needed again. */

cuArray->deallocateHost();

/* Set the number of threads per block to 1024 */
auto threadsPerBlock = 1024;

/* Set the number of blocks to the ceiling of the number of elements
 * divided by the number of threads per block. */
auto blocksPerGrid = (cuArray->size() + threadsPerBlock - 1) / threadsPerBlock;

/* Launch a CUDA kernel that does something cool and only takes
 * a single float array as an argument
 *<<<blocksPerGrid, threadsPerBlock>>>kernel(cuArray->device()); */ 

/* Free device memory. */
delete cuArray;

 ```

</details>

---

#### `CuArrayError deallocateDevice()`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Deallocate memory for the device data.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>
#include <algorithm>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Fill the CuArray with random values */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        cuArray->host()[i * cuArray->n() + j] =
                static_cast<float>(rand() / (float) RAND_MAX);
    }
}
/* Allocate device memory. */
cuArray->allocateDevice();

/* Copy data from host to device. */
cuArray->toDevice();

/* Set the number of threads per block to 1024 */
auto threadsPerBlock = 1024;

/* Set the number of blocks to the ceiling of the number of elements
 * divided by the number of threads per block. */
auto blocksPerGrid = (cuArray->size() + threadsPerBlock - 1) / threadsPerBlock;

/* Launch a CUDA kernel that does something cool and only takes
 * a single float array as an argument
 *<<<blocksPerGrid, threadsPerBlock>>>kernel(cuArray->device()); */ 

/* Transfer data from device to host. */
cuArray->toHost();

/* Deallocate the device array to reduce memory usage if it's not needed again. */
cuArray->deallocateDevice();

/* Perform some more calculations on the host array. */
auto sum = std::accumulate(
        cuArray->host(), 
        cuArray->host() + cuArray->size(), 
        0.0f
);

/* Free device memory. */
delete cuArray;
 ```

</details>

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
- **Related**:
  - [`fromNumpy2D(self, numpy_array: numpy.ndarray) -> int`](#fromnumpy2dself-numpyarray-numpyndarray---int)
  - [`fromNumpy1D(self, numpy_array: numpy.ndarray) -> int`](#fromnumpy1dself-numpyarray-numpyndarray---int)

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <iostream>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Create a linear float array that has 10 rows and 10 columns.*/
auto *NUMPY_ARRAY = new float[100];
int rows = 10;
int cols = 10;

/* Fill the NUMPY_ARRAY with random values */
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        NUMPY_ARRAY[i * cols + j] = (float)rand() / (float)RAND_MAX;
    }
}

/* Copy the NUMPY_ARRAY data into the CuArray. The 
 * CuArray has the same dimensions as the NUMPY_ARRAY. */
cuArray->fromNumpy(
        NUMPY_ARRAY,
        dim1,
        dim2
);


/* Free the NUMPY_ARRAY and CuArray. */
delete cuArray;
delete[] NUMPY_ARRAY;

```

</details>

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
- **Related**: 
  - [`toNumpy2D(self) -> numpy.ndarray`](#tonumpy2dself---numpyndarray)
  - [`toNumpy1D(self) -> numpy.ndarray`](#tonumpy2dself---numpyndarray)

<details><summary><b>Example</b></summary>

```cpp
#include "cuarray.h"
#include <iostream>
#include <random>

/* Creates a new float CuArray instance that will have 10 rows 
 * and 10 columns*/
CuArray<float> *cuArray = new CuArray<float>();
int m = 10; /* Number of rows */
int n = 10; /* Number of columns */
cuArray->init(m, n);

/* Create a double pointer to a float array. It will
 * store the data from the CuArray. */
auto NUMPY_ARRAY = new float*[1];

/* Create two double pointer int arrays that will store
 * the number rows and columns in the CuArray. 
 * Btw this is what the NumPy C backend is doing every time  
 * you create a numpy array in Python*/
auto rows = new int*[1];
auto cols = new int*[1];

/* Fill the CuArray with random values */
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        cuArray->set((float)rand() / (float)RAND_MAX, i, j);
    }
}

/* Copy the CuArray data into the NUMPY_ARRAY. The 
 * NUMPY_ARRAY has the same dimensions as the CuArray. */
cuArray->toNumpy(
        NUMPY_ARRAY,
        rows,
        cols
);

/* Print the NUMPY_ARRAY data and the CuArray data. */
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        std::cout << cuArray->get(i, j) << " ";
        std::cout << (*(NUMPY_ARRAY))[i * m + j] << std::endl;
    }
    std::cout << std::endl;
}

/* Clean this mess up. Makes you appreciate std::vectors :).*/
delete cuArray;
delete [] NUMPY_ARRAY[0];
delete [] NUMPY_ARRAY;
delete [] rows[0];
delete [] rows;
delete [] cols[0];
delete [] cols;

```

</details>

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

<details><summary><b>Example</b></summary>

```cpp
#include "cuarray.h"
#include <iostream>
#include <random>

/* Creates a new float CuArray instance that will have 10 rows 
 * and 10 columns*/
CuArray<float> *cuArray = new CuArray<float>();
int m = 10; /* Number of rows */
int n = 10; /* Number of columns */
cuArray->init(m, n);

/* Fill the CuArray with random values */
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        cuArray->set((float)rand() / (float)RAND_MAX, i, j);
    }
}

/* As it's name implies, get(i, j) returns the value at the 
 * specified position (i, j) in the CuArray. */

/* Use the get method to print the value at each position in the CuArray. */
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        std::cout << cuArray->get(i, j) << " ";
    }
    std::cout << std::endl;
}

/* Free the CuArray. */
delete cuArray;

```

</details>

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

<details><summary><b>Example</b></summary>

```cpp
#include "cuarray.h"
#include <iostream>
#include <random>

/* Creates a new float CuArray instance that will have 10 rows 
 * and 10 columns*/
CuArray<float> *cuArray = new CuArray<float>();
int m = 10; /* Number of rows */
int n = 10; /* Number of columns */
cuArray->init(m, n);

/* As it's name implies, set(value, i, j) sets the value at the 
 * specified position (i, j) in the CuArray. */

/* Use the set method to set the value at each position in the CuArray
 * to a random number.*/
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        cuArray->set((float)rand() / (float)RAND_MAX, i, j);
    }
}

/* Print the CuArray. */
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        std::cout << cuArray->get(i, j) << " ";
    }
    std::cout << std::endl;
}

/* Free the CuArray. */
delete cuArray;

```

</details>

---

#### `CuArrayError load(const std::string &fname)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Load the `CuArray` data from the specified NumPy binary (.npy) file.
- **Parameters**:
    - `const std::string &fname`: Name of the NumPy binary (.npy) file to load the CuArray from.
- **Returns**: `CuArrayError` indicating success (`0`) or specific error code.
- **Related**: [`load(self, filename: str) -> int` ](#loadself-filename-str---int)

<details><summary><b>Example</b></summary>

```cpp
#include "cuarray.h"
#include <iostream>
#include <random>

#define NETSCI_ROOT_DIR ""

/* Create a new double CuArray instance. We're using a double vs. float
 * here because the numpy array is a float64 array. If you tried 
 * to load this file into a CuArray<float> it would cause a 
 * segmentation fault.*/
CuArray<double> *cuArray = new CuArray<double>();

/* 2000 element .npy file in the cpp test data directory.
 * Adjust the NETSCI_ROOT_DIR macro to point to the project root directory. */
auto npyFname = NETSCI_ROOT_DIR "/tests/netcalc/cpp/data/2X_1D_1000_4.npy";

/* Load the data from the .npy file into the CuArray. */
cuArray->load(npyFname);

/* Print the CuArray. */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        std::cout << cuArray->get(i, j) << std::endl;
    }
}

/* Free the CuArray. */
delete cuArray;

```

</details>

---

#### `void save(const std::string &fname)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Save the `CuArray` data to the specified NumPy binary (.npy) file.
- **Parameters**:
    - `const std::string &fname`: Name of the NumPy binary (.npy) file to save the CuArray to.
- **Related**: [`save(self, filename: str)` ](#saveself-filename-str)

<details><summary><b>Example</b></summary>

```cpp
#include "cuarray.h"
#include <iostream>

#define NETSCI_ROOT_DIR "."

/* Create a new double CuArray instance that will have 10 rows and 10
* columns*/
CuArray<float> *cuArray = new CuArray<float>();
cuArray->init(10,
              10
);

/* Fill the CuArray with random values. */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        float val = static_cast <float> (rand()) /
                    static_cast <float> (RAND_MAX);
        cuArray->set(val,
                     i,
                     j);
    }
}

/* Save the CuArray to a .npy file. */
auto npyFname = NETSCI_ROOT_DIR "/tmp.npy";
cuArray->save(npyFname);

/* Create a new CuArray instance from the .npy file. */
auto cuArrayFromNpy = new CuArray<float>();
cuArrayFromNpy->load(npyFname);

/*Print (i, j) elements of the CuArray's next to each other.
 * and check for equality*/
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        auto val1 = cuArray->get(i, j);
        auto val2 = cuArrayFromNpy->get(i, j);
        bool equal = val1 == val2;
        std::cout
        << val1 << " "
        << val2 << " "
        << equal
        << std::endl;
        if (!equal) {
            std::cout
            << "Values at ("
            << i << ", "
            << j << ") are not equal."
            << std::endl;
            return 1;
        }


    }
}
delete cuArray;
return 0;

```

</details>

---

#### `CuArray<T> *sort(int i)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Sort the `CuArray` in descending order based on the values in the specified row.
- **Parameters**:
    - `int i`: Index of row to sort.
- **Returns**: Pointer to a new `CuArray` containing the sorted data.
- **Related**: [`sort(self, i: int) -> CuArray` ](#sortself-i-int---cuarray)

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <random>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Fill the CuArray with random values */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        cuArray->host()[i * cuArray->n() + j] =
                static_cast<float>(rand() / (float) RAND_MAX);
    }
}

/* Create a new CuArray that contains the sorted data from the 
 * 8th row of the original CuArray. */
auto sortedCuArray = cuArray->sort(7);

/* Print the sorted CuArray. */
for (int j = 0; j < sortedCuArray->n(); j++) {
    std::cout << sortedCuArray->get(0, j) << std::endl;
}

/* Cleanup time. */
delete cuArray;
delete sortedCuArray;
```

</details>

---

#### `T &operator[](int i) const`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get a reference to the element at the specified **linear** index in the `CuArray`. The linear index is calculated as `i * n + j`, where n is the number of columns.
- **Parameters**:
    - `int i`: Index of the element.
- **Returns**: Reference to the element at the specified index.
- **Related**: [`__getitem__(self, index: int) -> [Union, ElementType]` ](#__getitem__self-index-int---unionelementtype-cuarray)
 
<details><summary><b>Example</b></summary>

  ```cpp
#include <cuarray.h>
#include <iostream>

/* Create a new float CuArray instance */
auto cuArray = new CuArray<float>;

/* Initialize the CuArray with 3 rows and 3 columns */
cuArray->init(3, 3);

/*Set each i, j element equal to i*3 + j */
for (int i = 0; i < 9; i++) {
    cuArray->host()[i] = i;
}

/* Calculate the linear index that  
 * retrieves the 3rd element in the 2nd row of the CuArray. */
int i = 1;
int j = 2;
int linearIndex = i * cuArray->n() + j;
auto ijLinearVal = (*(cuArray))[linearIndex];
auto ijVal = cuArray->get(i, j);

/* Print the values at the linear index and the (i, j) index. */
std::cout
<< ijLinearVal << " "
<< ijVal
<< std::endl;

/*Deallocate memory*/ 
delete cuArray;

```

 </details>

---
#### `int owner() const`
* Language: C++
* Library: [CuArray](#cuarray)
* Class: [CuArray](#cuarray-class)
* Description: Determines if the CuArray instance owns the host data.
* Returns: `1` if the CuArray owns the host data, `0` otherwise.

<details><summary><b>Example</b></summary>

  ```cpp
#include <cuarray.h>
#include <iostream>

/* Create a new float CuArray instance */
auto cuArray = new CuArray<float>;

/* Initialize the CuArray with 3 rows and 3 columns */
cuArray->init(3, 3);

/*Set each i, j element equal to i*3 + j */
for (int i = 0; i < 9; i++) {
    cuArray->host()[i] = i;
}

/*
 * Create a float 'CuArray' that 
 * will be a shallow copy of the last two cuArray rows
 */
auto cuArray2x3Copy = new CuArray<float>;
cuArray2x3Copy->init(2, 3);

/* First row to copy from cuArray into cuArray2x3Copy */
int startRowIndex = 1;

/* Last row to copy from cuArray into cuArray2x3Copy */
int endRowIndex = 2;

cuArray2x3Copy->fromCuArrayShallowCopy(
        cuArray, /* Source for copying data into cuArray2x3Copy. See
                  * CuArray::fromCuArrayShallowCopy for more info. */
        startRowIndex, /* First row to copy from cuArray into cuArray2x3Copy */
        endRowIndex, /* Last row to copy from cuArray into cuArray2x3Copy */
        cuArray2x3Copy->m(), /* Number of rows in cuArray2x3Copy */
        cuArray2x3Copy->n() /* Number of columns in cuArray2x3Copy */
        );

/* Now make another CuArray that is a deep copy of cuArray2x3Copy */
auto cuArray2x3DeepCopy = new CuArray<float>;
cuArray2x3DeepCopy->init(2, 3);
cuArray2x3DeepCopy->fromCuArrayDeepCopy(
        cuArray, /* Source for copying data into cuArray2x3DeepCopy. See
                  * CuArray::fromCuArrayDeepCopy for more info. */
        startRowIndex, /* First row to copy from cuArray into cuArray2x3DeepCopy */
        endRowIndex, /* Last row to copy from cuArray into cuArray2x3DeepCopy */
        cuArray2x3DeepCopy->m(), /* Number of rows in cuArray2x3DeepCopy */
        cuArray2x3DeepCopy->n() /* Number of columns in cuArray2x3DeepCopy */
        );

/* Check if cuArray2x3Copy owns the host data. */
auto cuArray2x3CopyOwnsHostData = cuArray2x3Copy->owner();

/* Check if cuArray2x3DeepCopy owns the host data. 
 * Sorry for the verbosity :), I'm sure this is painful for 
 * Python devs to read (though Java devs are probably loving it).*/
auto cuArray2x3DeepCopyOwnsHostData = cuArray2x3DeepCopy->owner();

/* Print data in both arrays. */
for (int i = 0; i < cuArray2x3Copy->m(); i++) {
    for (int j = 0; j < cuArray2x3Copy->n(); j++) {
        std::cout
        << cuArray2x3Copy->get(i, j) << " "
        << cuArray2x3DeepCopy->get(i, j) << std::endl;
    }
} 

/* Print ownership info. */
std::cout
<< "cuArray2x3Copy owns host data: "
<< cuArray2x3CopyOwnsHostData
<< " cuArray2x3DeepCopy owns host data: "
<< cuArray2x3DeepCopyOwnsHostData
<< std::endl;

delete cuArray2x3Copy;
delete cuArray2x3DeepCopy;
delete cuArray;

```

 </details>

---

#### `CuArray<int> *argsort(int i)`

- **Language**: C++
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Perform an argsort on the specified row of the `CuArray` and return a new `CuArray` containing the
  sorted indices.
- **Parameters**:
    - `int i`: Index of row to perform argsort on.
- **Returns**: Pointer to a new `CuArray` containing the sorted indices.
- **Related**: [`argsort(self, i: int) -> CuArray` ](#argsortself-i-int---cuarray)

<details><summary><b>Example</b></summary>

```cpp
#include <cuarray.h>
#include <iostream>

/* Creates a new float CuArray instance */
CuArray<float> *cuArray = new CuArray<float>();

/* Initialize the CuArray with 300 rows and 300 columns */
auto rows = 300;
auto cols = 300;
cuArray->init(rows,
              cols);

/* Fill the CuArray with random values */
for (int i = 0; i < cuArray->m(); i++) {
    for (int j = 0; j < cuArray->n(); j++) {
        cuArray->host()[i * cuArray->n() + j] =
                static_cast<float>(rand() / (float) RAND_MAX);
    }
}

/* Create a new CuArray with indices that sort the 8th row 
 * of the original CuArray.*/
auto cuArrayRowIndex = 7;
auto sortedIndicesCuArray = cuArray->argsort(cuArrayRowIndex);

/* Create a new CuArray containing sorted data from the 8th row 
 * of the original CuArray.*/
auto sortedCuArray = cuArray->sort(cuArrayRowIndex);

/* Print the sorted CuArray and the corresponding values from the 
 * original CuArray using the sortedIndicesCuArray.*/
for (int j = 0; j < sortedCuArray->n(); j++) {
    auto sortedIndex = sortedIndicesCuArray->get(0, j);
    auto sortedValue = sortedCuArray->get(0, j);
    auto sortedValueFromOriginalCuArray = 
            cuArray->get(sortedIndex, cuArrayRowIndex);
    std::cout
    << sortedIndex << " "
    << sortedValue << " "
    << sortedValueFromOriginalCuArray << std::endl;
}

/* Cleanup time. */
delete cuArray;
delete sortedCuArray;
delete sortedIndicesCuArray;
```

</details>

---

### Python Methods

#### `__init__()`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Default constructor. Constructs an empty `CuArray` object.
- **Related**: [`CuArray()` ](#cuarray-constructor)

<details><summary><b>Example</b></summary>

```python
"""
Always precede CuArray with the data type
Here we are importing CuArray int and float templates.
"""
from cuarray import FloatCuArray, IntCuArray

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""Create a new int CuArray instance"""
int_cuarray = IntCuArray()

```

</details>

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

<details><summary><b>Example</b></summary>

```python
"""
Always precede CuArray with the data type
Here we are importing float templates.
"""
from cuarray import FloatCuArray

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""Initialize the float CuArray with 10 rows and 10 columns"""
float_cuarray.init(10, 10)

"""
Print the CuArray, 
which has a __repr__ method implemented in the SWIG interface
"""
print(float_cuarray)

```

</details>

---

#### `fromCuArray(self, cuArray, start: int, end: int, m: int, n: int) -> int`

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
- **Related**: [`CuArrayError fromCuArrayDeepCopy(CuArray<T> *cuArray, int start, int end, int m, int n)` ](#cuarrayerror-fromcuarraydeepcopycuarrayt-cuarray-int-start-int-end-int-m-int-n)

<details><summary><b>Example</b></summary>

```python
import numpy as np
"""
Always precede CuArray with the data type
Here we are float template.
"""
from cuarray import FloatCuArray


"""Create two new float CuArray instances"""
float_cuarray1 = FloatCuArray()
float_cuarray2 = FloatCuArray()

"""Initialize float_cuarray1 with 10 rows and 10 columns"""
float_cuarray1.init(10, 10)

"""Fill float_cuarray1 with random values"""
for i in range(float_cuarray1.m()):
    for j in range(float_cuarray1.n()):
        val = np.random.random()
        float_cuarray1[i][j] = val
        
"""Copy the data from float_cuarray1 into float_cuarray2"""
float_cuarray2.fromCuArray(float_cuarray1, 0, 9, 10, 10)

"""
Print both CuArrays. Also this performs a deep copy for 
memory safety.
"""
for i in range(float_cuarray1.m()):
    for j in range(float_cuarray1.n()):
        print(float_cuarray1[i][j], float_cuarray2[i][j])

```

</details>

---

#### `m(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the number of rows in the `CuArray`.
- **Returns**: Number of rows as `int`.
- **Related**: [`int m() const` ](#int-n-const)

<details><summary><b>Example</b></summary>

```python
import numpy as np
"""
Always precede CuArray with the data type
Here we are importing float template.
"""
from cuarray import FloatCuArray


"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""Initialize the float CuArray with 10 rows and 2 columns"""
float_cuarray.init(10, 2)

"""Print the number of rows in the CuArray"""
print(float_cuarray.m())

```

</details>

---

#### `n(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the number of columns in the `CuArray`.
- **Returns**: Number of columns as `int`.
- **Related**: [`int n() const` ](#int-n-const)

<details><summary><b>Example</b></summary>

```python
"""
Always precede CuArray with the data type
Here we are importing the float template. 
"""
from cuarray import FloatCuArray


"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""Initialize the float CuArray with 10 rows and 2 columns"""
float_cuarray.init(10, 2)

"""Print the number of columns in the CuArray"""
print(float_cuarray.n())

```

</details>

---

#### `size(self) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Get the total number of elements in the `CuArray`.
- **Returns**: Total number of elements as `int`.
- **Related**: [`int size() const` ](#int-size-const)

<details><summary><b>Example</b></summary>

```python
"""
Always precede CuArray with the data type
Here we are importing the float template.
"""
from cuarray import FloatCuArray


"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""Initialize the float CuArray with 10 rows and 2 columns"""
float_cuarray.init(10, 2)

"""Print the total number of values in the CuArray"""
print(float_cuarray.size())

```

</details>

---

#### `fromNumpy1D(self, numpy_array: numpy.ndarray) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from a 1-dimensional NumPy array to the `CuArray`.
- **Parameters**:
    - `numpy_array`(`numpy.ndarray`): NumPy array to copy from.
- **Returns**: `int`: `CuArrayError` indicating success (`0’) or specific error code.
- **Related**:[`CuArrayError fromNumpy(T *NUMPY_ARRAY, int NUMPY_ARRAY_DIM1, int NUMPY_ARRAY_DIM2)` ](#cuarrayerror-fromnumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2)
 
<details><summary><b>Example</b></summary>

```python
import numpy as np
"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""
Create a random float32, 1-dimension numpy array, 
with 10 elements
"""

np_array = np.random.rand(10).astype(np.float32)

"""Copy the numpy array to the CuArray instance"""
float_cuarray.fromNumpy1D(np_array)

"""Print the CuArray and numpy array to compare."""
for _ in range(10):
    print(float_cuarray[0][_], np_array[_])

```

</details>

---
#### `fromNumpy2D(self, numpy_array: numpy.ndarray) -> int`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from a 2-dimensional NumPy array to the `CuArray`.
- **Parameters**:
  - `numpy_array` (`numpy.ndarray`): NumPy array to copy from.
- **Returns**: `int`: `CuArrayError` indicating success (`0’) or specific error code.
- **Related**:[`CuArrayError fromNumpy(T *NUMPY_ARRAY, int NUMPY_ARRAY_DIM1, int NUMPY_ARRAY_DIM2)` ](#cuarrayerror-fromnumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2)
 
<details><summary><b>Example</b></summary>

```python
import numpy as np
"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""
Create a random float32, 2-dimension numpy array
with 10 rows and 10 columns.
"""
np_array = np.random.random((10, 10)).astype(np.float32)

"""Copy the numpy array to the CuArray instance"""
float_cuarray.fromNumpy2D(np_array)

"""Print the CuArray and numpy array to compare."""
for i in range(10):
    for j in range(10):
        print(float_cuarray[i][j], np_array[i][j])

```

</details>

---

#### `toNumpy1D(self) -> numpy.ndarray`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the `CuArray` to a 1-dimension NumPy array.
- **Returns**: Tuple containing the NumPy array and its dimensions.
- **Related:**[`void toNumpy(T **NUMPY_ARRAY, int **NUMPY_ARRAY_DIM1, int **NUMPY_ARRAY_DIM2)` ](#void-tonumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2)

<details><summary><b>Example</b></summary>

```python
import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""
Create a random float32, 1-dimension numpy array
with 10 elements.
"""
np_array1 = np.random.rand(10).astype(np.float32)

"""Copy the numpy array to the CuArray instance"""
float_cuarray.fromNumpy1D(np_array1)

"""Convert the CuArray instance to a numpy array"""
np_array2 = float_cuarray.toNumpy1D()

"""Print the CuArray and both numpy arrays to compare."""
for _ in range(10):
      print(
          float_cuarray[0][_],
          np_array1[_],
          np_array2[_]
      )

```

</details>

---

#### `toNumpy2D(self) -> numpy.ndarray`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Copy data from the `CuArray` to a NumPy array.
- **Returns**: Tuple containing the NumPy array and its dimensions.
- **Related:**[`void toNumpy(T **NUMPY_ARRAY, int **NUMPY_ARRAY_DIM1, int **NUMPY_ARRAY_DIM2)` ](#void-tonumpyt-numpyarray-int-numpyarraydim1-int-numpyarraydim2)

<details><summary><b>Example</b></summary>

```python
import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""Create a new float CuArray instance"""
float_cuarray = FloatCuArray()

"""
Create a random float32, 2-dimension numpy array
with 10 rows and 10 columns.
"""
np_array1 = np.random.random((10, 10)).astype(np.float32)

"""Copy the numpy array to the CuArray instance"""
float_cuarray.fromNumpy2D(np_array1)

"""Convert the CuArray instance to a numpy array"""
np_array2 = float_cuarray.toNumpy2D()

"""Print the CuArray and both numpy arrays to compare."""
for i in range(10):
    for j in range(10):
        print(
            float_cuarray[i][j],
            np_array1[i][j],
            np_array2[i][j]
        )

```

</details>

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

<details><summary><b>Example</b></summary>

```python
import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""
Create a new float CuArray instance with 
10 rows and 10 columns
"""
float_cuarray = FloatCuArray()
float_cuarray.init(10, 10)

"""Fill the array with random values"""

for i in range(10):
    for j in range(10):
        val = np.random.random()
        float_cuarray.set(val, i, j)

"""Print the array"""
print(float_cuarray)

```

</details>

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

<details><summary><b>Example</b></summary>

```python
import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""
Create a new float CuArray instance with 
10 rows and 10 columns
"""
float_cuarray = FloatCuArray()
float_cuarray.init(10, 10)

"""Fill the array with random values"""

for i in range(10):
    for j in range(10):
        val = np.random.random()
        float_cuarray.set(val, i, j)

"""Print the array using the get method"""
for i in range(10):
    for j in range(10):
        val = float_cuarray.get(i, j)
        print('{0:.{1}f}'.format(val, 5), end=" ")
    print()

```

</details>

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

<details><summary><b>Example</b></summary>

```python
import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""
Create a new float CuArray instance with 
10 rows and 10 columns
"""
float_cuarray = FloatCuArray()

"""
Create a random float32 numpy array with 10 rows
and 10 columns
"""
numpy_array = np.random.rand(10, 10).astype(np.float32)

"""Save the numpy array to a .npy file"""
np.save("tmp.npy", numpy_array)

"""
Load the .npy file into the float CuArray instance
"""
float_cuarray.load("tmp.npy")

"""Print the CuArray and the numpy array"""
for i in range(10):
    for j in range(10):
        print(float_cuarray[i][j], numpy_array[i, j])

```

</details>

---

#### `save(self, filename: str)`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Save the `CuArray` data to the specified file.
- **Parameters**:
    - `filename` (`str`): Name of the file to save.
- **Related**: [`void save(const std::string &fname)` ](#void-saveconst-stdstring-fname)

<details><summary><b>Example</b></summary>

```python
import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""
Create a new float CuArray instance with 
10 rows and 10 columns
"""
float_cuarray = FloatCuArray()

"""
Create a random float32 numpy array with 10 rows
and 10 columns
"""
numpy_array = np.random.rand(10, 10).astype(np.float32)

"""Save the numpy array to a .npy file"""
np.save("tmp.npy", numpy_array)

"""
Load the .npy file into the float CuArray instance
"""
float_cuarray.load("tmp.npy")

"""Print the CuArray and the numpy array"""
for i in range(10):
    for j in range(10):
        print(float_cuarray[i][j], numpy_array[i, j])

```

</details>

---

#### `sort(self, i: int) -> CuArray`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Sort the `CuArray` in descending order based on the values in the specified row.
- **Parameters**:
    - `i` (`int`): Row index to sort.
- **Returns**: New `CuArray` object containing sorted data.
- **Related**: [`CuArray<T> *sort(int i)` ](#cuarrayt-sortint-i)

<details><summary><b>Example</b></summary>

```python
import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""
Create a new float CuArray instance 
"""
float_cuarray = FloatCuArray()

"""
Create a random float32 numpy array with 10 rows
and 10 columns
"""
numpy_array = np.random.rand(10, 10).astype(np.float32)

"""Load the numpy array into the CuArray"""
float_cuarray.fromNumpy2D(numpy_array)

"""
Perform an out of place descending sort on the 
8th column of float_cuarray
"""
sorted_cuarray = float_cuarray.sort(7)

"""
Print the 8th row of the original 
CuArray and sorted_cuarray
"""
print(sorted_cuarray)
print(float_cuarray[7])

```

</details>

---

#### `__getitem__(self, index: int) -> Union[ElementType, CuArray]`
- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Retrieves the element or row at the specified index in the `CuArray`. If the `CuArray` consists of a single row, returns the element at the given index.
- **Parameters**:
  - `index (int)`: The index of the element or row to retrieve.
- **Returns**: The element or row at the specified index.
- **Related**: [`T &operator[](int i) const` ](#t-operatorint-i-const)

<details><summary><b>Example</b></summary>

```python
import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray float template 
"""
from cuarray import FloatCuArray

"""
Create a new float CuArray instance 
with 10 rows and 10 columns.
"""
float_cuarray = FloatCuArray()
float_cuarray.init(10, 10)

"""Fill it with random values"""
for i in range(10):
    for j in range(10):
        val = np.random.rand()
        float_cuarray.set(val, i, j)

"""Print the 8th row"""
print(float_cuarray[7])

"""Print the 5th element of the 8th row"""
print(float_cuarray[7][4])

```

</details>

---

#### `argsort(self, i: int) -> CuArray`

- **Language**: Python
- **Library**: [CuArray](#cuarray)
- **Class**: [CuArray](#cuarray-class)
- **Description**: Perform an argsort on the specified row of the `CuArray` and return a new `CuArray` containing the
  sorted indices.
- **Parameters**:
    - `i` (`int`): Index of row to perform argsort on.
- **Returns**: New `CuArray` object containing sorted indices.
- **Related**: [`CuArray<int> *argsort(int i)` ](#cuarrayint-argsortint-i)

<details><summary><b>Example</b></summary>

```python
import numpy as np

"""
Always precede CuArray with the data type
Here we are importing the CuArray int and float templates 
"""
from cuarray import FloatCuArray

"""
Create a new float CuArray instance 
"""
float_cuarray = FloatCuArray()

"""
Create a random float32 numpy array with 10 rows
and 10 columns
"""
numpy_array = np.random.rand(10, 10).astype(np.float32)

"""Load the numpy array into the CuArray"""
float_cuarray.fromNumpy2D(numpy_array)

"""
Perform a descending sort on 
the 8th row of float_cuarray
"""
sorted_cuarray = float_cuarray.sort(7)

"""
Get the indices that sort the 8th row of float_cuarray
"""
argsort_cuarray = float_cuarray.argsort(7)

"""
Print the sorted 8th row of float_cuarray using 
sorted_cuarray and argsort_cuarray indices
"""
for _ in range(10):
    sort_idx = argsort_cuarray[0][_]
    print(
        sorted_cuarray[0][_],
        float_cuarray[7][sort_idx]
    )


```

</details>

---

## NetChem

 ---

## NetCalc

---

---


</details>
