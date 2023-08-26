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
    __syncthreads();
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
        int *paths,
        int *nodeIdx,
        int n,
        float tolerance
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        int node = -1;
        int idx = nodeIdx[i * n + j];
        if (idx == 0) {
            paths[i * n * 5 + j * 5 + idx] = i;
            for (int _ = 1; _ < 5; _++) {
                paths[i * n * 5 + j * 5 + _] = -1;
            }
            nodeIdx[i * n + j] += 1;
            idx += 1;
        }

        float cij = INFINITY;
        for (int k = 0; k < n; k++) {
            auto AikHkjSum = A[j * n + k] + H[k * n + i];
            if (AikHkjSum < cij) {
                cij = AikHkjSum;
                node = k;
            }
        }
        if (std::fabs(cij - H[i * n + j]) < tolerance) {
            foundShortestPath[i * n + j] = 1;
        }
        else {
            paths[i * n * 5 + j * 5 + idx] = node;
            nodeIdx[i * n + j] += 1;
        }
        H[i * n + j] = cij;
    }
}

void netcalc::hedetniemiShortestPathsGpu(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        float tolerance
) {
    auto foundShortestPath = new CuArray<int>();
    foundShortestPath->init(A->m(),
                            A->n());
    foundShortestPath->allocateDevice();
    foundShortestPath->toDevice();
    auto nodeIdx = new CuArray<int>();
    nodeIdx->init(A->m(),
                  A->n());
    nodeIdx->allocateDevice();
    nodeIdx->toDevice();
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
    int numNodes = A->n();
    if (!A->allocateDevice()) {
        A->allocateDevice();
        A->toDevice();
    }
    H->allocateDevice();
    H->toDevice();
    int threadsPerBlock = 16;
    int blocksPerGrid =
            (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock,
               threadsPerBlock);
    dim3 grid(blocksPerGrid,
              blocksPerGrid);
    int foundShortestPathSize = numNodes * numNodes;
    paths->init(
            numNodes,
            numNodes * 5
    );
    paths->allocateDevice();
    paths->toDevice();
    paths->allocateDevice();
    paths->toDevice();
    void *hedetniemiShortestPathsKernelArgs[] = {
            &A->device(),
            &H->device(),
            &foundShortestPath->device(),
            &paths->device(),
            &nodeIdx->device(),
            &numNodes,
            &tolerance,
    };
    void *foundAllShortestPathsKernelArgs[] = {
            &foundShortestPath->device(),
            &foundAllShortestPaths->device(),
            &foundShortestPathSize,
    };
    while (foundAllShortestPaths->get(0,
                                      0) < numNodes * numNodes) {
        cudaLaunchKernel((void *) hedetniemiShortestPathsKernel,
                         grid,
                         block,
                         hedetniemiShortestPathsKernelArgs
        );
        cudaLaunchKernel((void *) foundAllShortestPathsKernel,
                         1,
                         1024,
                         foundAllShortestPathsKernelArgs
        );
        foundAllShortestPaths->toHost();
    }
    H->toHost();
    paths->toHost();
    delete foundShortestPath;
    delete foundAllShortestPaths;
    delete nodeIdx;
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