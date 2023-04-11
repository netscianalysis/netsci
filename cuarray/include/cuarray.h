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
    CuArray();

    CuArrayError init(
            int m,
            int n
    );

    CuArrayError init(
            T *host,
            int m,
            int n
    );

    ~CuArray();

    int n() const;

    int m() const;

    int size() const;

    size_t bytes() const;

    T *&host();

    T *&device();

    CuArrayError allocateHost();

    CuArrayError allocateDevice();

    CuArrayError allocatedHost() const;

    CuArrayError allocatedDevice() const;

    CuArrayError toDevice();

    CuArrayError toHost();

    CuArrayError deallocateHost();

    CuArrayError deallocateDevice();

    CuArrayError fromNumpy(
            T *NUMPY_ARRAY,
            int NUMPY_ARRAY_DIM1,
            int NUMPY_ARRAY_DIM2
    );

    CuArrayError fromNumpy(
            T *NUMPY_ARRAY,
            int NUMPY_ARRAY_DIM1
    );

    void toNumpy(
            T **NUMPY_ARRAY,
            int **NUMPY_ARRAY_DIM1,
            int **NUMPY_ARRAY_DIM2
    );

    void toNumpy(
            T **NUMPY_ARRAY,
            int **NUMPY_ARRAY_DIM1
    );

    T at(
            int i,
            int j
    ) const;

    CuArrayError at(
            T value,
            int i,
            int j
    );

    CuArrayError load(
            const std::string &fname
    );

    void save(
            const std::string &fname
    );

    T &operator[](int i) const;


private:
    T *host_;
    T *device_;
    int n_{};
    int m_{};
    int size_{};
    size_t bytes_{};
    int allocatedDevice_{};
    int allocatedHost_{};
};


#endif //MUTUAL_INFORMATION_SHARED_MEMORY_CUARRAY_H
