//
// Created by astokely on 8/28/23.
//
#include "cutoff_matrix.h"

__global__ void cutoffMatrixKernel(
        const float *A,
        float *Z,
        int m,
        int n,
        int d,
        float cutoff
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    auto size = (float) n;
    if (i < m && j < m) {
        float udist = 0.0;
        for (int k = 0; k < n; k++) {
            float distsq = 0.0;
            for (int l = 0; l < d; l++) {
                distsq +=
                        (A[i * d * n + l * n + k] - A[j * n * d + l *
                        n + k]) *
                        (A[i * d * n + l * n + k] - A[j * n * d + l *
                        n + k]);
            }
            udist += sqrt(distsq);
        }
        udist /= size;
        if (udist > cutoff) {
            Z[i * m + j] = 0.0;
        } else {
            Z[i * m + j] = 1.0;
        }
    }
}

void netcalc::cutoffMatrixGpu(
        CuArray<float> *A,
        CuArray<float> *Z,
        int m,
        int n,
        int d,
        float cutoff
) {
    A->allocateDevice();
    A->toDevice();
    Z->init(m,
            m);
    Z->allocateDevice();
    Z->toDevice();
    int threadsPerBlock = 16;
    int blocksPerGrid =
            (m + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock,
               threadsPerBlock);
    dim3 grid(blocksPerGrid,
              blocksPerGrid);
    void *cutoffMatrixKernelArgs[] = {
            (void *) &A->device(),
            (void *) &Z->device(),
            (void *) &m,
            (void *) &n,
            (void *) &d,
            (void *) &cutoff
    };
    cudaLaunchKernel(
            (const void *) cutoffMatrixKernel,
            grid,
            block,
            cutoffMatrixKernelArgs
    );
    Z->toHost();
}