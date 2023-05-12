//
// Created by astokely on 5/10/23.
//
#include "hedetniemi.h"

template<int maxPathLength>
__global__ void hedetniemiKernel(const float *A, float *H, int *paths, int n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    int r_paths[maxPathLength];
    for (int p = 0; p < maxPathLength; p++) {
        float hij = H[i * n + j];
        if (i < n && j < n) {
            float sum = 9999.99999;
            for (int k = 0; k < n; k++) {
                float tmp = A[i * n + k] + H[k * n + j];
                if (tmp < sum) {
                    sum = tmp;
                    r_paths[p] = k;
                }
            }
            if (sum != hij) {
                H[i * n + j] = sum;
            } else {
                for (int e = p - 1; e >= 0; e--) {
                    paths[i * n * maxPathLength + j * maxPathLength + e] = r_paths[e];
                }
                break;
            }
        }
    }
}


void gpuHedetniemi(CuArray<float> *A, CuArray<float> *H, CuArray<int> *paths) {
    H->fromCuArrayDeepCopy(
            A,
            0,
            A->m() - 1,
            A->m(),
            A->n()
    );
    int n = A->n();
    int maxPathLength = 10;
    A->allocateDevice();
    H->allocateDevice();
    paths->allocateDevice();
    A->toDevice();
    H->toDevice();
    paths->toDevice();
    //Calculate the number of blocks and threads for a 2D grid
    int threadsPerBlock = 32;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid(blocksPerGrid, blocksPerGrid);
    hedetniemiKernel<10><<<grid, block>>>(A->device(), H->device(), paths->device(), n);
    H->toHost();
    paths->toHost();
}