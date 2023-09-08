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
    if (this->allocatedHost_ == 1 && this->owner_) {
        this->deallocateHost();
    }
    if (this->allocatedDevice_ == 1) {
        this->deallocateDevice();
    }
    this->n_ = n;
    this->m_ = m;
    this->owner_ = 1;
    this->size_ = n * m;
    this->bytes_ = m * n * sizeof(T);
    this->allocatedDevice_ = 0;
    this->allocatedHost_ = 1;
    this->host_ = new T[this->size_];
    std::fill(
            this->host_,
            this->host_ + this->size_,
            0
    );
    this->device_ = nullptr;
    return 0;
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
    this->owner_ = 1;
    this->host_ = new T[this->size_];
    std::copy(host, host + this->size_, this->host_);
    this->device_ = nullptr;
    return 0;
}

template<typename T>
CuArray<T>::~CuArray() {
    if (this->allocatedHost_ == 1 && this->owner_) {
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
    if (this->allocatedHost_ == 0 && this->owner_) {
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
    if (this->allocatedHost_ == 1 && this->owner_) {
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
    this->owner_ = 1;
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
    this->owner_ = 1;
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
T CuArray<T>::get(
        int i,
        int j
) const {
    return this->host_[i * this->n_ + j];
}

template<typename T>
CuArrayError CuArray<T>::set(
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
    *(NUMPY_ARRAY_DIM1)[0] = this->size_;
    *NUMPY_ARRAY = new T[this->size_];
    std::copy(this->host_, this->host_ + this->size_, *NUMPY_ARRAY);
}

template<typename T>
CuArrayError CuArray<T>::fromCuArrayShallowCopy(CuArray<T> *cuArray, int start, int end, int m, int n) {
    if (((end - start) + 1) * cuArray->n_ != m * n) {
        return 1;
    } else {
        this->owner_ = 0;
        this->m_ = m;
        this->n_ = n;
        this->size_ = this->m_ * this->n_;
        this->bytes_ = this->size_ * sizeof(T);
        this->allocatedHost_ = 1;
        this->allocatedDevice_ = 0;
        this->host_ = cuArray->host_ + start * cuArray->n_;
        return 0;
    }
}

template<typename T>
CuArrayError
CuArray<T>::fromCuArrayDeepCopy(CuArray<T> *cuArray, int start, int end, int m, int n) {
    if (((end - start) + 1) * cuArray->n_ != m * n) {
        return 1;
    } else {
        this->owner_ = 1;
        this->m_ = m;
        this->n_ = n;
        this->size_ = this->m_ * this->n_;
        this->bytes_ = this->size_ * sizeof(T);
        this->allocatedHost_ = 1;
        this->allocatedDevice_ = 0;
        this->host_ = new T[this->size_];
        std::copy(cuArray->host_ + start * cuArray->n_,
                  cuArray->host_ + (end + 1) * cuArray->n_,
                  this->host_
        );
        return 0;
    }
}

template<typename T>
int CuArray<T>::owner() const {
    return this->owner_;
}

template<typename T>
CuArray<T> *CuArray<T>::sort(int i) {
    auto cuArray = new CuArray<T>();
    cuArray->fromCuArrayDeepCopy(
            this,
            i,
            i,
            1, this->n()
    );
    std::sort(
            cuArray->host_,
            cuArray->host_ + cuArray->n(),
            std::greater<T>()
    );
    return cuArray;
}

template<typename T>
CuArray<int> *CuArray<T>::argsort(int i) {
    auto cuArray = new CuArray<int>();
    cuArray->init(1, this->n());
    std::iota(
            cuArray->host(),
            cuArray->host() + cuArray->n(),
            0
    );
    std::sort(
            cuArray->host(),
            cuArray->host() + cuArray->n(),
            [this, i](int a, int b) {
                return this->get(i, a) > this->get(i, b);
            }
    );
    return cuArray;
}

template<typename T>
CuArrayRow<T>::CuArrayRow(
        CuArray<T> *cuArray,
        int i
) {
    this->data_ = (cuArray->host() + i * cuArray->n());
    this->n_ = cuArray->n();
}

template<typename T>
T &CuArrayRow<T>::operator[](int i) const {
    return this->data_[i];
}

template<typename T>
int CuArrayRow<T>::n() const {
    return this->n_;
}

template<typename T>
T* CuArrayRow<T>::data() const {
    return this->data_;
}


template
class CuArray<int>;

template
class CuArray<float>;

template
class CuArray<double>;

template
class CuArrayRow<int>;

template
class CuArrayRow<float>;

template
class CuArrayRow<double>;

