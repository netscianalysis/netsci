//
// Created by andy on 3/23/23.
//

#ifndef MUTUAL_INFORMATION_SHARED_MEMORY_CUARRAY_H
#define MUTUAL_INFORMATION_SHARED_MEMORY_CUARRAY_H

#include <cstddef>
#include <string>

typedef int CuArrayError;

template<typename T>
class CuArray {
public:
    /**
     * \brief Default constructor for CuArray.
     *
     * Constructs an empty CuArray object.
     */
    CuArray();

    /**
     * \brief Initialize the CuArray with the specified dimensions.
     *
     * Initializes the CuArray with the specified dimensions, allocating memory on both the host and the device.
     *
     * \param m The number of rows.
     * \param n The number of columns.
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError init(
            int m,
            int n
    );

    /**
     * \brief Initialize the CuArray with the specified host data and dimensions.
     *
     * Initializes the CuArray with the specified host data and dimensions, allocating memory on both the host and the device.
     * The data is shallow copied, meaning the ownership is not transferred.
     *
     * \param host Pointer to the input host data.
     * \param m The number of rows.
     * \param n The number of columns.
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError init(
            T *host,
            int m,
            int n
    );

    /**
     * \brief Shallow copy data from another CuArray.
     *
     * Shallow copies the host data from the provided CuArray. All data
     * in the range of rows, specified by the 'start' and 'end'
     * parameters, is copied. The range is inclusive. This CuArray
     * does not own the data, so the data can only be deallocated by
     * deleting the source CuArray.
     *
     * \param cuArray Pointer to the source CuArray.
     * \param start The index of the first row to copy.
     * \param end The index of the last row to copy.
     * \param m The number of rows in this CuArray.
     * \param n The number of columns in this CuArray.
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError fromCuArrayShallowCopy(
            CuArray<T> *cuArray,
            int start,
            int end,
            int m,
            int n
    );

    /**
     * \brief Deep copy data from another CuArray.
     *
     * Deep copies the host data from the provided CuArray. All data in
     * the range of rows, specified by the 'start' and 'end'
     * parameters, is copied.  The range is inclusive.
     * Memory is allocated on the host CuArray.
     *
     * \param cuArray Pointer to the source CuArray.
     * \param start The index of the first row to copy.
     * \param end The index of the last row to copy.
     * \param m The number of rows in this CuArray.
     * \param n The number of columns in this CuArray.
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError fromCuArrayDeepCopy(
            CuArray<T> *cuArray,
            int start,
            int end,
            int m,
            int n
    );

    /**
     * \brief Destructor for CuArray.
     *
     * Deallocates the memory on both the host and the device.
     */
    ~CuArray();

    /**
     * \brief Get the number of columns in the CuArray.
     *
     * Returns the number of columns in the CuArray.
     *
     * \return The number of columns.
     */
    int n() const;

    /**
     * \brief Get the number of rows in the CuArray.
     *
     * Returns the number of rows in the CuArray.
     *
     * \return The number of rows.
     */
    int m() const;

    /**
     * \brief Get the total number of elements in the CuArray.
     *
     * Returns the total number of elements in the CuArray, which is equal to the number of rows multiplied by the number of columns.
     *
     * \return The total number of elements.
     */
    int size() const;

    /**
     * \brief Get the total size in bytes of the CuArray data.
     *
     * Returns the total size in bytes of the CuArray data, including both the host and device memory.
     *
     * \return The size in bytes.
     */
    size_t bytes() const;

    /**
     * \brief Get a reference to the host data.
     *
     * Returns a reference to the host data.
     *
     * \return A reference to the host data.
     */
    T *&host();

    /**
     * \brief Get a reference to the device data.
     *
     * Returns a reference to the device data.
     *
     * \return A reference to the device data.
     */
    T *&device();

    /**
     * \brief Allocate memory for the host data.
     *
     * Allocates memory for the host data.
     *
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError allocateHost();

    /**
     * \brief Allocate memory for the device data.
     *
     * Allocates memory for the device data.
     *
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError allocateDevice();

    /**
     * \brief Check if memory is allocated for the host data.
     *
     * Checks if memory is allocated for the host data.
     *
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError allocatedHost() const;

    /**
     * \brief Check if memory is allocated for the device data.
     *
     * Checks if memory is allocated for the device data.
     *
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError allocatedDevice() const;

    /**
     * \brief Copy data from the host to the device.
     *
     * Copies the data from the host to the device.
     *
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError toDevice();

    /**
     * \brief Copy data from the device to the host.
     *
     * Copies the data from the device to the host.
     *
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError toHost();

    /**
     * \brief Deallocate memory for the host data.
     *
     * Deallocates the memory for the host data.
     *
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError deallocateHost();

    /**
     * \brief Deallocate memory for the device data.
     *
     * Deallocates the memory for the device data.
     *
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError deallocateDevice();

    /**
     * \brief Copy data from a NumPy array to the CuArray.
     *
     * Copies data from the specified NumPy array to the CuArray.
     *
     * \param NUMPY_ARRAY    Pointer to the input NumPy array.
     * \param NUMPY_ARRAY_DIM1    Pointer to the dimension 1 of the NumPy array.
     * \param NUMPY_ARRAY_DIM2    Pointer to the dimension 2 of the NumPy array.
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError fromNumpy(
            T *NUMPY_ARRAY,
            int NUMPY_ARRAY_DIM1,
            int NUMPY_ARRAY_DIM2
    );

    /**
     * \brief Copy data from a NumPy array to the CuArray.
     *
     * Copies data from the specified NumPy array to the CuArray.
     *
     * \param NUMPY_ARRAY    Pointer to the input NumPy array.
     * \param NUMPY_ARRAY_DIM1    Pointer to the dimension 1 of the NumPy array.
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError fromNumpy(
            T *NUMPY_ARRAY,
            int NUMPY_ARRAY_DIM1
    );

    /**
     * \brief Copy data from the CuArray to a NumPy array.
     *
     * Copies data from the CuArray to the specified NumPy array.
     *
     * \param NUMPY_ARRAY    Pointer to the output NumPy array.
     * \param NUMPY_ARRAY_DIM1    Pointer to the dimension 1 of the NumPy array.
     * \param NUMPY_ARRAY_DIM2    Pointer to the dimension 2 of the NumPy array.
     */
    void toNumpy(
            T **NUMPY_ARRAY,
            int **NUMPY_ARRAY_DIM1,
            int **NUMPY_ARRAY_DIM2
    );

    /**
     * \brief Copy data from the CuArray to a NumPy array.
     *
     * Copies data from the CuArray to the specified NumPy array.
     *
     * \param NUMPY_ARRAY    Pointer to the output NumPy array.
     * \param NUMPY_ARRAY_DIM1    Pointer to the dimension 1 of the NumPy array.
     */
    void toNumpy(
            T **NUMPY_ARRAY,
            int **NUMPY_ARRAY_DIM1
    );

    /**
     * \brief Get the value at the specified position in the CuArray.
     *
     * Returns the value at the specified position (i, j) in the CuArray.
     *
     * \param i The row index.
     * \param j The column index.
     * \return The value at the specified position.
     */
    T get(
            int i,
            int j
    ) const;

    /**
     * \brief Set the value at the specified position in the CuArray.
     *
     * Sets the value at the specified position (i, j) in the CuArray to the given value.
     *
     * \param value The value to set.
     * \param i The row index.
     * \param j The column index.
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError set(
            T value,
            int i,
            int j
    );

    /**
     * \brief Load the CuArray from a file.
     *
     * Loads the CuArray data from the specified file.
     *
     * \param fname The name of the file to load.
     * \return The CuArrayError indicating the success or failure of the operation.
     */
    CuArrayError load(const std::string &fname);

    /**
     * \brief Save the CuArray to a file.
     *
     * Saves the CuArray data to the specified file.
     *
     * \param fname The name of the file to save.
     */
    void save(const std::string &fname);

    /**
     * \brief Sort the CuArray based on the specified column.
     *
     * Sorts the CuArray in ascending order based on the values in the specified column.
     *
     * \param i The column index to sort.
     * \return A pointer to a new CuArray containing the sorted data.
     */
    CuArray<T> *sort(int i);

    /**
     * \brief Get a reference to the element at the specified index in the CuArray.
     *
     * Returns a reference to the element at the specified index in the CuArray.
     *
     * \param i The index of the element.
     * \return A reference to the element at the specified index.
     */
    T &operator[](int i) const;

    /**
     * \brief Get the owner of the CuArray.
     *
     * Returns the owner of the CuArray, which indicates whether the CuArray is responsible for memory deallocation.
     *
     * \return The owner of the CuArray.
     */
    int owner() const;

    /**
     * \brief Perform an argsort on the specified column of the CuArray.
     *
     * Performs an argsort on the specified column of the CuArray and returns a new CuArray that contains the sorted indices.
     *
     * \param i The column index to argsort.
     * \return A pointer to a new CuArray containing the sorted indices.
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

private:
    T *data_;
    int n_{};
};

#endif // MUTUAL_INFORMATION_SHARED_MEMORY_CUARRAY_H
