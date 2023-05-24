//
// Created by astokely on 5/10/23.
//
#include <numeric>
#include <iostream>
#include "hedetniemi.h"

__global__ void foundAllShortestPathsKernel(
        const int *foundShortestPath,
        int *foundAllShortestPaths,
        int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = tid;
    __shared__ int s_foundAllShortestPaths[1024];
    s_foundAllShortestPaths[tid] = 0;
    __syncthreads();
    while (index < n) {
        s_foundAllShortestPaths[tid] += foundShortestPath[index];
        index += blockDim.x;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_foundAllShortestPaths[tid] += s_foundAllShortestPaths[
                    tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        foundAllShortestPaths[0] = s_foundAllShortestPaths[0];
    }
}

__global__

void correlationToAdjacencyKernel(
        float *A,
        const float *C,
        int n
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        if (i == j) {
            A[i * n + j] = 0.0;
        } else if (C[i * n + j] == 0.0) {
            A[i * n + j] = INFINITY;
        } else {
            A[i * n + j] = (float) 1.0 / C[i * n + j];
        }
    }
}

__global__ void
hedetniemiShortestPathsKernel(
        const float *A,
        float *H,
        int *foundShortestPath,
        int n
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float cij = INFINITY;
        for (int k = 0; k < n; k++) {
            auto AikHkjSum = A[j * n + k] + H[k * n + i];
            if (AikHkjSum < cij) {
                cij = AikHkjSum;
            }
        }
        if (cij == H[i * n + j]) {
            foundShortestPath[i * n + j] = 1;
        }
        H[i * n + j] = cij;
    }
}

__global__ void
hedetniemiRecoverShortestPathsKernel(
        const float *A,
        float *H,
        int *paths,
        int n,
        int maxPathLength
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        for (int k = 0; k < maxPathLength; k++) {
            paths[i * n * maxPathLength + j * maxPathLength + k] = -1;
        }
        int pos = j;
        int c = 0;
        float ssp = H[i * n + j];
        while (pos != i) {
            for (int k = 0; k < n; k++) {
                float delta = std::abs(
                        (H[i * n + k] + A[pos * n + k]) - ssp);
                if (delta < 0.0001 && pos != k) {
                    pos = k;
                    ssp = H[i * n + pos];
                    paths[i * n * maxPathLength + j * maxPathLength +
                          c] = pos;
                    c++;
                }
            }
        }
    }
}

int hedetniemiShortestPathsGpu(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths
) {
    int maxPathLength = 0;
    auto foundShortestPath = new CuArray<int>();
    foundShortestPath->init(A->m(),
                            A->n());
    foundShortestPath->allocateDevice();
    foundShortestPath->toDevice();
    auto foundAllShortestPaths = new CuArray<int>();
    foundAllShortestPaths->init(1,
                                1);
    foundAllShortestPaths->allocateDevice();
    foundAllShortestPaths->toDevice();
    H->fromCuArrayDeepCopy(
            A,
            0,
            A->m() - 1,
            A->m(),
            A->n()
    );
    int n = A->n();
    if (!A->allocateDevice()) {
        A->allocateDevice();
        A->toDevice();
    }
    H->allocateDevice();
    H->toDevice();
    int threadsPerBlock = 16;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock,
               threadsPerBlock);
    dim3 grid(blocksPerGrid,
              blocksPerGrid);
    int foundShortestPathSize = n * n;
    void *args1[] = {
            &A->device(),
            &H->device(),
            &foundShortestPath->device(),
            &n,
    };
    void *args2[] = {
            &foundShortestPath->device(),
            &foundAllShortestPaths->device(),
            &foundShortestPathSize,
    };
    while (foundAllShortestPaths->get(0,
                                      0) < n * n) {
        cudaLaunchKernel((void *) hedetniemiShortestPathsKernel,
                         grid,
                         block,
                         args1
        );
        cudaLaunchKernel((void *) foundAllShortestPathsKernel,
                         1,
                         1024,
                         args2
        );
        maxPathLength++;
        foundAllShortestPaths->toHost();
    }
    paths->init(
            n,
            n * maxPathLength
    );
    paths->allocateDevice();
    paths->toDevice();
    void *args3[] = {
            &A->device(),
            &H->device(),
            &paths->device(),
            &n,
            &maxPathLength,
    };
    cudaLaunchKernel((void *) hedetniemiRecoverShortestPathsKernel,
                     grid,
                     block,
                     args3
    );
    H->toHost();
    paths->toHost();
    delete foundShortestPath;
    delete foundAllShortestPaths;
    return maxPathLength;
}

void correlationToAdjacencyGpu(
        CuArray<float> *A,
        CuArray<float> *C,
        int n
) {
    A->allocateDevice();
    if (!C->allocatedDevice()) {
        C->allocateDevice();
    }
    C->toDevice();
    A->toDevice();
    int threadsPerBlock = 16;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock,
               threadsPerBlock);
    dim3 grid(blocksPerGrid,
              blocksPerGrid);
    auto kernel = correlationToAdjacencyKernel;
    void *args[] = {&A->device(), &C->device(), &n};
    cudaLaunchKernel((void *) kernel,
                     grid,
                     block,
                     args);
    A->toHost();
}