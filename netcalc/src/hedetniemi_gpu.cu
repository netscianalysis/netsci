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
        int n,
        float tolerance
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
        if (std::fabs(cij - H[i * n + j]) < tolerance) {
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
        int longestShortestPathNodeCount,
        float tolerance
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        for (int k = 0; k < longestShortestPathNodeCount; k++) {
            paths[i * n * longestShortestPathNodeCount +
                  j * longestShortestPathNodeCount + k] = -1;
        }
        unsigned int vi = i;
        unsigned int vk = j;
        int ijSspNodeCount = 1;
        float ikSspLength = H[vi * n + vk];
        paths[i * n * longestShortestPathNodeCount +
              j * longestShortestPathNodeCount +
              ijSspNodeCount - 1] = vk;
        while (vk != vi) {
            ijSspNodeCount++;
            for (int vp = 0; vp < n; vp++) {
                float deltaSspIpKp = std::abs(
                        (H[vi * n + vp]
                         + A[vk * n + vp])
                        - ikSspLength
                );
                if (deltaSspIpKp < tolerance && vk != vp) {
                    vk = vp;
                    ikSspLength = H[i * n + vk];
                    paths[vi * n * longestShortestPathNodeCount +
                          j * longestShortestPathNodeCount +
                          ijSspNodeCount - 1] = vk;
                }
            }
        }
        for (int jiSspNodeIndex = 0; jiSspNodeIndex < ijSspNodeCount / 2; jiSspNodeIndex++) {
            int jiSspNode = paths[i * n * longestShortestPathNodeCount +
                             j * longestShortestPathNodeCount + jiSspNodeIndex];
            paths[i * n * longestShortestPathNodeCount +
                  j * longestShortestPathNodeCount + jiSspNodeIndex] =
                    paths[i * n * longestShortestPathNodeCount +
                          j * longestShortestPathNodeCount +
                          ijSspNodeCount - jiSspNodeIndex - 1];
            paths[i * n * longestShortestPathNodeCount +
                  j * longestShortestPathNodeCount +
                  ijSspNodeCount - jiSspNodeIndex - 1] = jiSspNode;
        }
    }
}

void netcalc::hedetniemiShortestPathsGpu(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        float tolerance
) {
    int longestShortestPathNodeCount = 0;
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
    void *hedetniemiShortestPathsKernelArgs[] = {
            &A->device(),
            &H->device(),
            &foundShortestPath->device(),
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
        longestShortestPathNodeCount++;
    }
    longestShortestPathNodeCount++;
    paths->init(
            numNodes,
            numNodes * (longestShortestPathNodeCount)
    );
    paths->allocateDevice();
    paths->toDevice();
    void *hedetniemiRecoverShortestPathsKernelArgs[] = {
            &A->device(),
            &H->device(),
            &paths->device(),
            &numNodes,
            &longestShortestPathNodeCount,
            &tolerance,
    };
    cudaLaunchKernel((void *) hedetniemiRecoverShortestPathsKernel,
                     grid,
                     block,
                     hedetniemiRecoverShortestPathsKernelArgs
    );
    H->toHost();
    paths->toHost();
    delete foundShortestPath;
    delete foundAllShortestPaths;
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