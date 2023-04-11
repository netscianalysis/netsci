//
// Created by andy on 3/23/23.
//
#include <algorithm>
#include "cuarray.h"
#include "cnpy.h"

template<typename T>
CuArray<T>::CuArray() {
    this->allocatedDevice_ = 0;
    this->allocatedHost_ = 0;
    this->host_ = nullptr;
    this->device_ = nullptr;
}

template<typename T>
CuArrayError CuArray<T>::init(
        int m,
        int n
) {
    this->n_ = n;
    this->m_ = m;
    this->size_ = n * m;
    this->bytes_ = m * n * sizeof(T);
    this->allocatedDevice_ = 0;
    this->allocatedHost_ = 1;
    this->host_ = new T[this->size_];
    this->device_ = nullptr;
}

template<typename T>
CuArrayError CuArray<T>::init(
        T *host,
        int m,
        int n
) {
    this->n_ = n;
    this->m_ = m;
    this->size_ = n * m;
    this->bytes_ = this->size_ * sizeof(T);
    this->allocatedDevice_ = 0;
    this->allocatedHost_ = 1;
    this->host_ = new T[this->size_];
    std::copy(host, host + this->size_, this->host_);
    this->device_ = nullptr;
}

template<typename T>
CuArray<T>::~CuArray() {
    if (this->allocatedHost_ == 1) {
        this->deallocateHost();
    }
    if (this->allocatedDevice_ == 1) {
        this->deallocateDevice();
    }
}

template<typename T>
int CuArray<T>::n() const {
    return this->n_;
}

template<typename T>
size_t CuArray<T>::bytes() const {
    return this->bytes_;
}

template<typename T>
T *&CuArray<T>::host() {
    return this->host_;
}

template<typename T>
T *&CuArray<T>::device() {
    return this->device_;
}

template<typename T>
CuArrayError CuArray<T>::allocateHost() {
    if (this->allocatedHost_ == 0) {
        this->host_ = new T[this->size_];
        this->allocatedHost_ = 1;
    }
    return (this->allocatedHost_ == 1) ? 0 : 1;
}

template<typename T>
CuArrayError CuArray<T>::allocateDevice() {
    if (this->allocatedDevice_ == 0) {
        cudaMalloc(&this->device_, this->bytes_);
        this->allocatedDevice_ = 1;
    }
    return (this->allocatedDevice_ == 1) ? 0 : 1;
}

template<typename T>
CuArrayError CuArray<T>::allocatedHost() const {
    return this->allocatedHost_;
}

template<typename T>
CuArrayError CuArray<T>::allocatedDevice() const {
    return this->allocatedDevice_;
}

template<typename T>
CuArrayError CuArray<T>::toDevice() {
    if (this->allocatedDevice_ == 1 && this->allocatedHost_ == 1) {
        cudaMemcpy(this->device_, this->host_, this->bytes_,
                   cudaMemcpyHostToDevice);
        return 0;
    }
    return 1;
}

template<typename T>
CuArrayError CuArray<T>::toHost() {
    if (this->allocatedDevice_ == 1 && this->allocatedHost_ == 1) {
        cudaMemcpy(this->host_, this->device_, this->bytes_,
                   cudaMemcpyDeviceToHost);
        return 0;
    }
    return 1;
}

template<typename T>
CuArrayError CuArray<T>::deallocateHost() {
    if (this->allocatedHost_ == 1) {
        delete[] this->host_;
        this->allocatedHost_ = 0;
    }
    return (this->allocatedHost_ == 0) ? 0 : 1;
}

template<typename T>
CuArrayError CuArray<T>::deallocateDevice() {
    if (this->allocatedDevice_ == 1) {
        cudaFree(this->device_);
        this->allocatedDevice_ = 0;
    }
    return (this->allocatedDevice_ == 0) ? 0 : 1;
}

template<typename T>
T &CuArray<T>::operator[](int i) const {
    return this->host_[i];
}

template<typename T>
int CuArray<T>::m() const {
    return this->m_;
}

template<typename T>
int CuArray<T>::size() const {
    return this->size_;
}

template<typename T>
CuArrayError CuArray<T>::fromNumpy(
        T *NUMPY_ARRAY,
        int NUMPY_ARRAY_DIM1
) {
    CuArrayError err;
    this->m_ = 1;
    this->n_ = NUMPY_ARRAY_DIM1;
    this->size_ = this->m_ * this->n_;
    this->bytes_ = this->size_ * sizeof(T);
    if (this->allocatedHost_ == 1) {
        err = this->deallocateHost();
    }
    if (this->allocatedDevice_ == 1) {
        err = this->deallocateDevice();
    }
    err = this->allocateHost();
    std::copy(NUMPY_ARRAY, NUMPY_ARRAY + this->size_, this->host_);
    return err;
}

template<typename T>
CuArrayError CuArray<T>::fromNumpy(
        T *NUMPY_ARRAY,
        int NUMPY_ARRAY_DIM1,
        int NUMPY_ARRAY_DIM2
) {
    CuArrayError err;
    this->m_ = NUMPY_ARRAY_DIM1;
    this->n_ = NUMPY_ARRAY_DIM2;
    this->size_ = this->m_ * this->n_;
    this->bytes_ = this->size_ * sizeof(T);
    if (this->allocatedHost_ == 1) {
        err = this->deallocateHost();
    }
    if (this->allocatedDevice_ == 1) {
        err = this->deallocateDevice();
    }
    err = this->allocateHost();
    std::copy(NUMPY_ARRAY, NUMPY_ARRAY + this->size_, this->host_);
    return err;
}

template<typename T>
T CuArray<T>::at(
        int i,
        int j
) const {
    return this->host_[i * this->n_ + j];
}

template<typename T>
CuArrayError CuArray<T>::at(
        T value,
        int i,
        int j
) {
    if (i < this->m_ && j < this->n_) {
        this->host_[i * this->n_ + j] = value;
        return 0;
    }
    return 1;
}

template<typename T>
CuArrayError CuArray<T>::load(const std::string &fname) {
    cnpy::NpyArray npyArray = cnpy::npy_load(fname);
    auto err = this->fromNumpy(
            npyArray.data<T>(),
            npyArray.shape[0],
            npyArray.shape[1]
    );
    return err;
}

template<typename T>
void CuArray<T>::save(const std::string &fname) {
    cnpy::npy_save(
            fname,
            this->host_,
            {static_cast<unsigned long>(this->m_),
             static_cast<unsigned long>(this->n_)},
            "w"
    );
}

template<typename T>
void CuArray<T>::toNumpy(
        T **NUMPY_ARRAY,
        int **NUMPY_ARRAY_DIM1,
        int **NUMPY_ARRAY_DIM2
) {
    *NUMPY_ARRAY_DIM1 = new int;
    *NUMPY_ARRAY_DIM2 = new int;
    *(NUMPY_ARRAY_DIM1[0]) = this->m_;
    *(NUMPY_ARRAY_DIM2[0]) = this->n_;
    *NUMPY_ARRAY = new T[this->size_];
    std::copy(this->host_, this->host_ + this->size_, *NUMPY_ARRAY);
}

template<typename T>
void CuArray<T>::toNumpy(
        T **NUMPY_ARRAY,
        int **NUMPY_ARRAY_DIM1
) {
    *NUMPY_ARRAY_DIM1 = new int;
    *(NUMPY_ARRAY_DIM1)[0] = this->n_;
    *NUMPY_ARRAY = new T[this->size_];
    std::copy(this->host_, this->host_ + this->size_, *NUMPY_ARRAY);
}

template
class CuArray<int>;

template
class CuArray<float>;

template
class CuArray<double>;

