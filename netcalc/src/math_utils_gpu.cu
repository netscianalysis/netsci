//
// Created by astokely on 5/16/23.
//
#include "math_utils.h"

__device__ void mathUtilsWarpReduce(
        volatile float *s_a,
        unsigned int localThreadIndex
) {
    s_a[localThreadIndex] += s_a[localThreadIndex + 32];
    s_a[localThreadIndex] += s_a[localThreadIndex + 16];
    s_a[localThreadIndex] += s_a[localThreadIndex + 8];
    s_a[localThreadIndex] += s_a[localThreadIndex + 4];
    s_a[localThreadIndex] += s_a[localThreadIndex + 2];
    s_a[localThreadIndex] += s_a[localThreadIndex + 1];
}


__global__ void meanKernel(
        const float *a,
        float *u,
        int m,
        int n
) {
    volatile __shared__ float s_u[1024];
    for (unsigned int blockIndex = blockIdx.x; blockIndex < m; blockIndex += gridDim.x) {
        auto localThreadIndex = threadIdx.x;
        auto index = localThreadIndex;
        s_u[localThreadIndex] = 0.0;
        __syncthreads();
        while (index < n) {
            s_u[localThreadIndex] += a[blockIndex * n + index];
            index += blockDim.x;
        }
        __syncthreads();

        if (localThreadIndex < 512) {
            s_u[localThreadIndex] += s_u[localThreadIndex + 512];
        }
        __syncthreads();
        if (localThreadIndex < 256) {
            s_u[localThreadIndex] += s_u[localThreadIndex + 256];
        }
        __syncthreads();
        if (localThreadIndex < 128) {
            s_u[localThreadIndex] += s_u[localThreadIndex + 128];
        }
        __syncthreads();
        if (localThreadIndex < 64) {
            s_u[localThreadIndex] += s_u[localThreadIndex + 64];
        }
        __syncthreads();

        if (localThreadIndex < 32) {
            mathUtilsWarpReduce(s_u, localThreadIndex);
        }
        if (localThreadIndex == 0) {
            u[blockIndex] = s_u[0] / ((float) n);
        }
        __syncthreads();
    }
}

__global__ void standardDeviationKernel(
        const float *a,
        float *u,
        float *sigma,
        int m,
        int n
) {
    volatile __shared__ float s_u[1024];
    volatile __shared__ float s_sigma[1024];
    for (unsigned int blockIndex = blockIdx.x; blockIndex < m; blockIndex += gridDim.x) {
        auto localThreadIndex = threadIdx.x;
        auto index = localThreadIndex;
        s_u[localThreadIndex] = 0.0;
        __syncthreads();
        while (index < n) {
            s_u[localThreadIndex] += a[blockIndex * n + index];
            index += blockDim.x;
        }
        __syncthreads();

        if (localThreadIndex < 512) {
            s_u[localThreadIndex] += s_u[localThreadIndex + 512];
        }
        __syncthreads();
        if (localThreadIndex < 256) {
            s_u[localThreadIndex] += s_u[localThreadIndex + 256];
        }
        __syncthreads();
        if (localThreadIndex < 128) {
            s_u[localThreadIndex] += s_u[localThreadIndex + 128];
        }
        __syncthreads();
        if (localThreadIndex < 64) {
            s_u[localThreadIndex] += s_u[localThreadIndex + 64];
        }
        __syncthreads();

        if (localThreadIndex < 32) {
            mathUtilsWarpReduce(s_u, localThreadIndex);
        }
        if (localThreadIndex == 0) {
            u[blockIndex] = s_u[0] / ((float) n);
        }
        __syncthreads();
    }
    __syncthreads();
    for (unsigned int blockIndex = blockIdx.x; blockIndex < m; blockIndex += gridDim.x) {
        auto localThreadIndex = threadIdx.x;
        auto index = localThreadIndex;
        s_sigma[localThreadIndex] = 0.0;
        __syncthreads();
        while (index < n) {
            s_sigma[localThreadIndex] += powf(a[blockIndex * n + index] - u[blockIndex], 2.0);
            index += blockDim.x;
        }
        __syncthreads();

        if (localThreadIndex < 512) {
            s_sigma[localThreadIndex] += s_sigma[localThreadIndex + 512];
        }
        __syncthreads();
        if (localThreadIndex < 256) {
            s_sigma[localThreadIndex] += s_sigma[localThreadIndex + 256];
        }
        __syncthreads();
        if (localThreadIndex < 128) {
            s_sigma[localThreadIndex] += s_sigma[localThreadIndex + 128];
        }
        __syncthreads();
        if (localThreadIndex < 64) {
            s_sigma[localThreadIndex] += s_sigma[localThreadIndex + 64];
        }
        __syncthreads();

        if (localThreadIndex < 32) {
            mathUtilsWarpReduce(s_sigma, localThreadIndex);
        }
        if (localThreadIndex == 0) {
            sigma[blockIndex] = sqrt(s_sigma[0] / ((float) n));
        }
        __syncthreads();
    }
    __syncthreads();
}

void meanGpu(
        CuArray<float> *a,
        CuArray<float> *u,
        int m,
        int n
) {
    if (!a->allocatedDevice()) {
        a->allocateDevice();
        a->toDevice();
    }
    if (!u->allocatedDevice()) {
        u->allocateDevice();
    }
    auto blocksPerGrid = m;
    auto threadsPerBlock = 1024;
    meanKernel<<<
    blocksPerGrid,
    threadsPerBlock
    >>>(a->device(), u->device(), m, n);
    u->toHost();
}

void standardDeviationGpu(
        CuArray<float> *a,
        CuArray<float> *u,
        CuArray<float> *sigma,
        int m,
        int n
) {
    if (!a->allocatedDevice()) {
        a->allocateDevice();
        a->toDevice();
    }
    if (!u->allocatedDevice()) {
        u->allocateDevice();
    }
    if (!sigma->allocatedDevice()) {
        sigma->allocateDevice();
    }
    auto blocksPerGrid = m;
    auto threadsPerBlock = 1024;
    standardDeviationKernel<<<
    blocksPerGrid,
    threadsPerBlock
    >>>(a->device(), u->device(), sigma->device(), m, n);
    u->toHost();
    sigma->toHost();
}
