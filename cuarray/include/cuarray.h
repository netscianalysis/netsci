//
// Created by andy on 3/23/23.
//

#ifndef MUTUAL_INFORMATION_SHARED_MEMORY_CUARRAY_H
#define MUTUAL_INFORMATION_SHARED_MEMORY_CUARRAY_H

#include <cstddef>
#include <string>

typedef int CuArrayError;

/*!
 * @class CuArray
 * @param T Data type of the array elements.
 * @brief Manages CUDA-supported arrays, offering initialization,
 * memory management, and data manipulation. Implemented as a template
 * class in C++, with Python and Tcl wrapper interfaces. In Python and
 * Tcl, use as <ElementType>CuArray (e.g., FloatCuArray, IntCuArray), as
 * they don't support templates. Supports float and int types in Python
 * and Tcl, and all numeric types in C++.
 */
template<typename T>
class CuArray {
public:
    /*!
 * @function{CuArray} @type{constructor}
 * @brief Constructs an empty CuArray object.
 *
 * @CppExample{CuArray_CuArray.cpp}
 *
 * @PythonExample{CuArray___init__.py}
 */
    CuArray();

/*!
 * @function{init} @type{CuArrayError}
 * @brief Initializes CuArray with specified dimensions and allocates
 * memory on host and device.
 *
 * @param m Number of rows.
 * @param n Number of columns.
 * @return CuArrayError indicating operation success or failure.
 *
 * @CppExample{CuArray_init1.cpp}
 * @PythonExample{CuArray_init.py}
 */

    CuArrayError init(
            int m,
            int n
    );

    /*!
 * @function{init} @type{CuArrayError}
 * @brief Initializes CuArray with specified host data and dimensions, performing a shallow copy.
 * Allocates memory on both the host and the device. The data is shallow copied, so the ownership remains unchanged.
 *
 * @param host Pointer to input host data.
 * @param m Number of rows.
 * @param n Number of columns.
 * @return CuArrayError indicating operation success or failure.
 *
 * @CppExample{CuArray_init2.cpp}
 */


    CuArrayError init(
            T *host,
            int m,
            int n
    );

    /*!
 * @function{fromCuArrayShallowCopy} @type{CuArrayError}
 * @brief Performs a shallow copy of data from another CuArray within a specified row range.
 * Copies the host data from the given CuArray, within the inclusive range
 * specified by 'start' and 'end'. This CuArray does not own the copied data,
 * and deallocation is handled by the source CuArray.
 *
 * @param cuArray Pointer to the source CuArray.
 * @param start Index of the first row to copy.
 * @param end Index of the last row to copy.
 * @param m Number of rows in this CuArray.
 * @param n Number of columns in this CuArray.
 * @return CuArrayError indicating the operation's success or failure.
 *
 * @CppExample{CuArray_fromCuArrayShallowCopy.cpp}
 */

    CuArrayError fromCuArrayShallowCopy(
            CuArray<T> *cuArray,
            int start,
            int end,
            int m,
            int n
    );

    /*!
   * @function{fromCuArrayDeepCopy} @type{CuArrayError}
   * @brief Performs a deep copy of data from another CuArray within a specified row range.
   * Copies the host data from the given CuArray, including all data within the inclusive range defined
   * by 'start' and 'end'. Memory for the copied data is allocated in this CuArray's host memory.
   *
   * @param cuArray Pointer to the source CuArray.
   * @param start Index of the first row to copy.
   * @param end Index of the last row to copy.
   * @param m Number of rows in this CuArray.
   * @param n Number of columns in this CuArray.
   * @return CuArrayError indicating the operation's success or failure.
   *
   * @CppExample{CuArray_fromCuArrayDeepCopy.cpp}
   * @PythonExample{CuArray_fromCuArray.py}
   */

    CuArrayError fromCuArrayDeepCopy(
            CuArray<T> *cuArray,
            int start,
            int end,
            int m,
            int n
    );

    /*!
   * @brief Destructor for CuArray.
   * Deallocates memory on both the host and the device.
   */

    ~CuArray();

    /*!
  * @function{n} @type{int}
  * @brief Returns the number of columns in the CuArray.
  *
  * @return Number of columns.
  *
  * @CppExample{CuArray_n.cpp}

  * @PythonExample{CuArray_n.py}
  */
    int n() const;

/*!
 * @function{m} @type{int}
 * @brief Returns the number of rows in the CuArray.
 *
 * @return Number of rows.
 *
 * @CppExample{CuArray_m.cpp}

 * @PythonExample{CuArray_m.py}
 */
    int m() const;

/*!
 * @function{size} @type{int}
 * @brief Returns the total number of elements in the CuArray.
 *
 * @return Total number of elements (rows multiplied by columns).
 *
 * @CppExample{CuArray_size.cpp}

 * @PythonExample{CuArray_size.py}
 */
    int size() const;


    /*!
   * @function{bytes} @type{size_t}
   * @brief Returns the total size in bytes of the CuArray data.
   *
   * Includes both the host and device memory.
   *
   * @return Size in bytes.
   *
   * @CppExample{CuArray_bytes.cpp}

   * @PythonExample{CuArray_bytes.py}
   */
    size_t bytes() const;

/*!
 * @function{host} @type{T *&}
 * @brief Returns a reference to the host data.
 *
 * @return Reference to the host data.
 *
 * @CppExample{CuArray_host.cpp}
 */
    T *&host();

/*!
 * @function{device} @type{T}
 * @brief Returns a reference to the device data.
 *
 * @return Reference to the device data.
 *
 * @CppExample{CuArray_device.cpp}
 */
    T *&device();


    /*!
   * @function{allocateHost} @type{CuArrayError}
   * @brief Allocates memory for the host data.
   *
   * @return CuArrayError indicating success or failure of the operation.
   *
   * @CppExample{CuArray_allocateHost.cpp}
   */
    CuArrayError allocateHost();

/*!
 * @function{allocateDevice} @type{CuArrayError}
 * @brief Allocates memory for the device data.
 *
 * @return CuArrayError indicating success or failure of the operation.
 *
 * @CppExample{CuArray_allocateDevice.cpp}
 */
    CuArrayError allocateDevice();

/*!
 * @function{allocatedHost} @type{CuArrayError}
 * @brief Checks if memory is allocated for the host data.
 *
 * @return CuArrayError indicating success or failure of the operation.
 *
 * @CppExample{CuArray_allocatedHost.cpp}
 */
    CuArrayError allocatedHost() const;

/*!
 * @function{allocatedDevice} @type{CuArrayError}
 * @brief Checks if memory is allocated for the device data.
 *
 * @return CuArrayError indicating success or failure of the operation.
 *
 * @CppExample{CuArray_allocatedDevice.cpp}
 */
    CuArrayError allocatedDevice() const;

/*!
 * @function{toDevice} @type{CuArrayError}
 * @brief Copies data from the host to the device.
 *
 * @return CuArrayError indicating success or failure of the operation.
 *
 * @CppExample{CuArray_toDevice.cpp}
 */
    CuArrayError toDevice();


    /*!
   * @function{toHost} @type{CuArrayError}
   * @brief Copies data from the device to the host.
   *
   * @return CuArrayError indicating success or failure of the operation.
   *
   * @CppExample{CuArray_toHost.cpp}
   */
    CuArrayError toHost();

/*!
 * @function{deallocateHost} @type{CuArrayError}
 * @brief Deallocates memory for the host data.
 *
 * @return CuArrayError indicating success or failure of the operation.
 *
 * @CppExample{CuArray_deallocateHost.cpp}
 */
    CuArrayError deallocateHost();

/*!
 * @function{deallocateDevice} @type{CuArrayError}
 * @brief Deallocates memory for the device data.
 *
 * @return CuArrayError indicating success or failure of the operation.
 *
 * @CppExample{CuArray_deallocateDevice.cpp}
 */
    CuArrayError deallocateDevice();

/*!
 * @function{fromNumpy} @type{CuArrayError}
 * @brief Copies data from a NumPy array to the CuArray.
 *
 * @param NUMPY_ARRAY Pointer to the input NumPy array.
 * @param NUMPY_ARRAY_DIM1 Dimension 1 of the NumPy array.
 * @param NUMPY_ARRAY_DIM2 Dimension 2 of the NumPy array.
 * @return CuArrayError indicating success or failure of the operation.
 *
 * @CppExample{CuArray_fromNumpy2.cpp}
 *
 * @PythonExample{CuArray_fromNumpy2D.py}
 */
    CuArrayError fromNumpy(
            T *NUMPY_ARRAY,
            int NUMPY_ARRAY_DIM1,
            int NUMPY_ARRAY_DIM2
    );

/*!
 * @function{fromNumpy} @type{CuArrayError}
 * @brief Copies data from a NumPy array to the CuArray.
 *
 * @param NUMPY_ARRAY Pointer to input NumPy array.
 * @param NUMPY_ARRAY_DIM1 Dimension 1 of the NumPy array.
 * @return CuArrayError indicating success or failure of the operation.
 *
 * @CppExample{CuArray_fromNumpy1.cpp}
 *
 * @PythonExample{CuArray_fromNumpy1D.py}
 */
    CuArrayError fromNumpy(
            T *NUMPY_ARRAY,
            int NUMPY_ARRAY_DIM1
    );

/*!
 * @function{toNumpy} @type{void}
 * @brief Copies data from the CuArray to a NumPy array.
 *
 * @param NUMPY_ARRAY Pointer to output NumPy array.
 * @param NUMPY_ARRAY_DIM1 Dimension 1 of the NumPy array.
 * @param NUMPY_ARRAY_DIM2 Dimension 2 of the NumPy array.
 *
 * @CppExample{CuArray_toNumpy2.cpp}
 *
 * @PythonExample{CuArray_toNumpy2D.py}
 */
    void toNumpy(
            T **NUMPY_ARRAY,
            int **NUMPY_ARRAY_DIM1,
            int **NUMPY_ARRAY_DIM2
    );

/*!
 * @function{toNumpy} @type{void}
 * @brief Copies data from the CuArray to a NumPy array.
 *
 * @param NUMPY_ARRAY Pointer to output NumPy array.
 * @param NUMPY_ARRAY_DIM1 Dimension 1 of the NumPy array.
 *
 * @CppExample{CuArray_toNumpy1.cpp}
 *
 * @PythonExample{CuArray_toNumpy1D.py}
 */
    void toNumpy(
            T **NUMPY_ARRAY,
            int **NUMPY_ARRAY_DIM1
    );

/*!
 * @function{get} @type{T}
 * @brief Returns the value at a specified position in the CuArray.
 *
 * @param i Row index.
 * @param j Column index.
 * @return Value at the specified position.
 *
 * @CppExample{CuArray_get.cpp}
 * @PythonExample{CuArray_get.py}
 */
    T get(
            int i,
            int j
    ) const;


    /*!
  * @function{set} @type{CuArrayError}
  * @brief Sets a value at a specified position in the CuArray.
  *
  * @param value Value to set.
  * @param i Row index.
  * @param j Column index.
  * @return CuArrayError indicating success or failure of the operation.
  *
  * @CppExample{CuArray_set.cpp}
  * @PythonExample{CuArray_set.py}
  */
    CuArrayError set(
            T value,
            int i,
            int j
    );

/*!
 * @function{load} @type{CuArrayError}
 * @brief Loads CuArray data from a specified file.
 *
 * @param fname File name to load from.
 * @return CuArrayError indicating success or failure of the operation.
 *
 * @CppExample{CuArray_load.cpp}
 * @PythonExample{CuArray_load.py}
 */
    CuArrayError load(const std::string &fname);

/*!
 * @function{save} @type{void}
 * @brief Saves CuArray data to a specified file.
 *
 * @param fname File name to save to.
 *
 * @CppExample{CuArray_save.cpp}
 * @PythonExample{CuArray_save.py}
 */
    void save(const std::string &fname);

/*!
 * @function{sort} @type{CuArray<T>}
 * @brief Sorts CuArray based on the values in a specified row.
 *
 * @param i Index of the row to sort by.
 * @return Pointer to a new CuArray with sorted data.
 *
 * @CppExample{CuArray_sort.cpp}
 * @PythonExample{CuArray_sort.py}
 */
    CuArray<T> *sort(int i);

/*!
 * @function{operator[]} @type{T &}
 * @brief Returns a reference to the element at a specified index in the CuArray.
 *
 * @param i Index of the element.
 * @return Reference to the element at the specified index.
 *
 * @CppExample{CuArray_subscriptOperator.cpp}
 * @PythonExample{CuArray___getitem__.py}
 */
    T &operator[](int i) const;

/*!
 * @function{owner} @type{int}
 * @brief Returns the owner status of the CuArray.
 * Indicates whether the CuArray is responsible for memory deallocation.
 *
 * @return Owner status of the CuArray.
 *
 * @CppExample{CuArray_owner.cpp}
 */
    int owner() const;

/*!
 * @function{argsort} @type{CuArray}
 * @brief Performs an argsort on a specified row of the CuArray.
 * Returns a new CuArray containing sorted indices.
 *
 * @param i Column index to argsort.
 * @return Pointer to a new CuArray with sorted indices.
 *
 * @CppExample{CuArray_argsort.cpp}
 * @PythonExample{CuArray_argsort.py}
 */
    CuArray<int> *argsort(int i);

private:
    T *host_;
    T *device_;
    int n_{};
    int m_{};
    int size_{};
    size_t bytes_{};
    int allocatedDevice_{};
    int allocatedHost_{};
    int owner_{};
};

template<typename T>
class CuArrayRow {
public:
    CuArrayRow(
            CuArray<T> *cuArray,
            int i
    );

    T &operator[](int i) const;

    int n() const;

    T *data() const;

private:
    T *data_;
    int n_{};
};

#endif // MUTUAL_INFORMATION_SHARED_MEMORY_CUARRAY_H
