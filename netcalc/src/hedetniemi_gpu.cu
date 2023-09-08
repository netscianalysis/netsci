//
// Created by astokely on 5/10/23.
//
#include <numeric>
#include <limits>
#include <iostream>
#include "hedetniemi.h"


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

__global__ void hedetniemiAllShortestPathLengthsKernel(
        const float *A,
        float *H,
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
        H[i * n + j] = cij;
    }
}

__global__ void
hedetniemiAllShortestPathsPart1Kernel(
        const float *A,
        float *H,
        float *Hi,
        int n,
        int maxPathLength,
        int p
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float cij = INFINITY;
        for (int k = 0; k < n; k++) {
            auto AikHkjSum = A[j * n + k] + H[k * n + i];
            if (AikHkjSum < cij + .0001) {
                cij = AikHkjSum;
            }
        }
        Hi[p * n * n + i * n + j] = H[i * n + j];
        H[i * n + j] = cij;
    }
}

__global__ void hedetniemiAllShortestPathsKernelPart2(
        const float *H,
        const float *Hi,
        const float *A,
        int *paths,
        int n,
        int maxPathLength
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        float length = H[i * n + j];
        int kj = j;
        for (int p = 0; p < maxPathLength; p++) {
            for (int k = 0; k < n; k++) {
                float a = A[k * n + kj];
                float h = Hi[n * n * (maxPathLength - 2 - p) + i * n +
                             k];
                if (k != kj) {
                    if (std::fabs(a + h - length) < .0001) {
                        paths[i * n * maxPathLength + p * n + j] = k;
                        length = h;
                        kj = k;
                        break;
                    }
                }
            }
        }
    }
}

void netcalc::hedetniemiAllShortestPathLengthsGpu(
        CuArray<float> *A,
        CuArray<float> *H,
        int maxPathLength
) {
    H->fromCuArrayDeepCopy(
            A,
            0,
            A->m() - 1,
            A->m(),
            A->n()
    );
    int numNodes = A->n();
    A->allocateDevice();
    A->toDevice();
    H->allocateDevice();
    H->toDevice();
    int threadsPerBlock = 16;
    int blocksPerGrid =
            (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock,
               threadsPerBlock);
    dim3 grid(blocksPerGrid,
              blocksPerGrid);
    void *hedetniemiAllShortestPathLengthsKernelArgs[] = {
            &A->device(),
            &H->device(),
            &numNodes
    };
    for (int p = 0; p < maxPathLength; p++) {
        cudaLaunchKernel((void *) hedetniemiAllShortestPathLengthsKernel,
                         grid,
                         block,
                         hedetniemiAllShortestPathLengthsKernelArgs
        );
    }
    H->toHost();
}

void netcalc::hedetniemiAllShortestPathsGpu(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        int maxPathLength
) {
    H->fromCuArrayDeepCopy(
            A,
            0,
            A->m() - 1,
            A->m(),
            A->n()
    );
    auto Hi = new CuArray<float>;
    Hi->init(
            A->m() * maxPathLength,
            A->n()
    );
    Hi->allocateDevice();
    Hi->toDevice();
    int numNodes = A->n();
    A->allocateDevice();
    A->toDevice();
    H->allocateDevice();
    H->toDevice();
    int threadsPerBlock = 16;
    int blocksPerGrid =
            (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock,
               threadsPerBlock);
    dim3 grid(blocksPerGrid,
              blocksPerGrid);
    paths->init(
            numNodes * maxPathLength,
            numNodes
    );
    for (int _ = 0; _ < paths->size(); _++) {
        paths->host()[_] = -1;
    }
    paths->allocateDevice();
    paths->toDevice();
    for (int p = 0; p < maxPathLength; p++) {
        void *hedetniemiAllShortestPathsPart1KernelArgs[] = {
                &A->device(),
                &H->device(),
                &Hi->device(),
                &numNodes,
                &maxPathLength,
                &p
        };
        cudaLaunchKernel((void *) hedetniemiAllShortestPathsPart1Kernel,
                         grid,
                         block,
                         hedetniemiAllShortestPathsPart1KernelArgs
        );
    }
    void *hedetniemiAllShortestPathsPart2KernelArgs[] = {
            &H->device(),
            &Hi->device(),
            &A->device(),
            &paths->device(),
            &numNodes,
            &maxPathLength
    };
    cudaLaunchKernel((void *) hedetniemiAllShortestPathsKernelPart2,
                     grid,
                     block,
                     hedetniemiAllShortestPathsPart2KernelArgs
    );
    H->toHost();
    paths->toHost();
    A->deallocateDevice();
    H->deallocateDevice();
    paths->deallocateDevice();
    delete Hi;
}

void netcalc::correlationToAdjacencyGpu(
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