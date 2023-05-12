//
// Created by astokely on 5/10/23.
//
#include <map>
#include "hedetniemi.h"

__global__ void correlationToAdjacencyKernel(
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
            A[i * n + j] = (float) RAND_MAX;
        } else {
            A[i * n + j] = (float) 1.0 / C[i * n + j];
        }
    }
}

template<int maxPathLength>
__global__ void hedetniemiShortestPathsKernel(
        const float *A,
        float *H,
        int *paths,
        int n
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    int r_paths[maxPathLength];
    for (int pathLength = 0; pathLength < maxPathLength; pathLength++) {
        if (i < n && j < n) {
            float hij = H[i * n + j];
            auto cij = (float) RAND_MAX;
            for (int k = 0; k < n; k++) {
                auto AikHkjSum = A[i * n + k] + H[k * n + j];
                if (AikHkjSum < cij) {
                    cij = AikHkjSum;
                    r_paths[pathLength] = k;
                }
            }
            if (cij != hij) {
                H[i * n + j] = cij;
            } else {
                for (int edgeIndex = pathLength - 1;
                     edgeIndex >= 0; edgeIndex--) {
                    paths[i * n * maxPathLength + j * maxPathLength +
                          edgeIndex] = r_paths[edgeIndex];
                }
                break;
            }
        }
    }
}

template<int maxPathLength>
__global__ void hedetniemiShortestPathLengthsKernel(
        const float *A,
        float *H,
        int n
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    for (int pathLength = 0; pathLength < maxPathLength; pathLength++) {
        if (i < n && j < n) {
            float hij = H[i * n + j];
            auto cij = (float) RAND_MAX;
            for (int k = 0; k < n; k++) {
                auto AikHkjSum = A[i * n + k] + H[k * n + j];
                if (AikHkjSum < cij) {
                    cij = AikHkjSum;
                }
            }
            if (cij != hij) {
                H[i * n + j] = cij;
            } else {
                for (int edgeIndex = pathLength - 1;
                     edgeIndex >= 0; edgeIndex--) {
                }
                break;
            }
        }
    }
}
std::map<int, void *> hedetniemiShortestPathLengthsKernels() {
    return std::map<int, void *>{
            {1,  (void *) hedetniemiShortestPathLengthsKernel<1>},
            {2,  (void *) hedetniemiShortestPathLengthsKernel<2>},
            {3,  (void *) hedetniemiShortestPathLengthsKernel<3>},
            {4,  (void *) hedetniemiShortestPathLengthsKernel<4>},
            {5,  (void *) hedetniemiShortestPathLengthsKernel<5>},
            {6,  (void *) hedetniemiShortestPathLengthsKernel<6>},
            {7,  (void *) hedetniemiShortestPathLengthsKernel<7>},
            {8,  (void *) hedetniemiShortestPathLengthsKernel<8>},
            {9,  (void *) hedetniemiShortestPathLengthsKernel<9>},
            {10, (void *) hedetniemiShortestPathLengthsKernel<10>},
            {11, (void *) hedetniemiShortestPathLengthsKernel<11>},
            {12, (void *) hedetniemiShortestPathLengthsKernel<12>},
            {13, (void *) hedetniemiShortestPathLengthsKernel<13>},
            {14, (void *) hedetniemiShortestPathLengthsKernel<14>},
            {15, (void *) hedetniemiShortestPathLengthsKernel<15>},
            {16, (void *) hedetniemiShortestPathLengthsKernel<16>},
            {17, (void *) hedetniemiShortestPathLengthsKernel<17>},
            {18, (void *) hedetniemiShortestPathLengthsKernel<18>},
            {19, (void *) hedetniemiShortestPathLengthsKernel<19>},
            {20, (void *) hedetniemiShortestPathLengthsKernel<20>},
            {21, (void *) hedetniemiShortestPathLengthsKernel<21>},
            {22, (void *) hedetniemiShortestPathLengthsKernel<22>},
            {23, (void *) hedetniemiShortestPathLengthsKernel<23>},
            {24, (void *) hedetniemiShortestPathLengthsKernel<24>},
            {25, (void *) hedetniemiShortestPathLengthsKernel<25>},
            {26, (void *) hedetniemiShortestPathLengthsKernel<26>},
            {27, (void *) hedetniemiShortestPathLengthsKernel<27>},
            {28, (void *) hedetniemiShortestPathLengthsKernel<28>},
            {29, (void *) hedetniemiShortestPathLengthsKernel<29>},
            {30, (void *) hedetniemiShortestPathLengthsKernel<30>},
            {31, (void *) hedetniemiShortestPathLengthsKernel<31>},
            {32, (void *) hedetniemiShortestPathLengthsKernel<32>},
            {33, (void *) hedetniemiShortestPathLengthsKernel<33>},
            {34, (void *) hedetniemiShortestPathLengthsKernel<34>},
            {35, (void *) hedetniemiShortestPathLengthsKernel<35>},
            {36, (void *) hedetniemiShortestPathLengthsKernel<36>},
            {37, (void *) hedetniemiShortestPathLengthsKernel<37>},
            {38, (void *) hedetniemiShortestPathLengthsKernel<38>},
            {39, (void *) hedetniemiShortestPathLengthsKernel<39>},
            {40, (void *) hedetniemiShortestPathLengthsKernel<40>},
            {41, (void *) hedetniemiShortestPathLengthsKernel<41>},
            {42, (void *) hedetniemiShortestPathLengthsKernel<42>},
            {43, (void *) hedetniemiShortestPathLengthsKernel<43>},
            {44, (void *) hedetniemiShortestPathLengthsKernel<44>},
            {45, (void *) hedetniemiShortestPathLengthsKernel<45>},
            {46, (void *) hedetniemiShortestPathLengthsKernel<46>},
            {47, (void *) hedetniemiShortestPathLengthsKernel<47>},
            {48, (void *) hedetniemiShortestPathLengthsKernel<48>},
            {49, (void *) hedetniemiShortestPathLengthsKernel<49>},
            {50, (void *) hedetniemiShortestPathLengthsKernel<50>},
            {51, (void *) hedetniemiShortestPathLengthsKernel<51>},
            {52, (void *) hedetniemiShortestPathLengthsKernel<52>},
            {53, (void *) hedetniemiShortestPathLengthsKernel<53>},
            {54, (void *) hedetniemiShortestPathLengthsKernel<54>},
            {55, (void *) hedetniemiShortestPathLengthsKernel<55>},
            {56, (void *) hedetniemiShortestPathLengthsKernel<56>},
            {57, (void *) hedetniemiShortestPathLengthsKernel<57>},
            {58, (void *) hedetniemiShortestPathLengthsKernel<58>},
            {59, (void *) hedetniemiShortestPathLengthsKernel<59>},
            {60, (void *) hedetniemiShortestPathLengthsKernel<60>},
            {61, (void *) hedetniemiShortestPathLengthsKernel<61>},
            {62, (void *) hedetniemiShortestPathLengthsKernel<62>},
            {63, (void *) hedetniemiShortestPathLengthsKernel<63>},
            {64, (void *) hedetniemiShortestPathLengthsKernel<64>},
            {65, (void *) hedetniemiShortestPathLengthsKernel<65>},
            {66, (void *) hedetniemiShortestPathLengthsKernel<66>},
            {67, (void *) hedetniemiShortestPathLengthsKernel<67>},
            {68, (void *) hedetniemiShortestPathLengthsKernel<68>},
            {69, (void *) hedetniemiShortestPathLengthsKernel<69>},
            {70, (void *) hedetniemiShortestPathLengthsKernel<70>},
            {71, (void *) hedetniemiShortestPathLengthsKernel<71>},
            {72, (void *) hedetniemiShortestPathLengthsKernel<72>},
            {73, (void *) hedetniemiShortestPathLengthsKernel<73>},
            {74, (void *) hedetniemiShortestPathLengthsKernel<74>},
            {75, (void *) hedetniemiShortestPathLengthsKernel<75>},
            {76, (void *) hedetniemiShortestPathLengthsKernel<76>},
            {77, (void *) hedetniemiShortestPathLengthsKernel<77>},
            {78, (void *) hedetniemiShortestPathLengthsKernel<78>},
            {79, (void *) hedetniemiShortestPathLengthsKernel<79>},
            {80, (void *) hedetniemiShortestPathLengthsKernel<80>},
            {81, (void *) hedetniemiShortestPathLengthsKernel<81>},
            {82, (void *) hedetniemiShortestPathLengthsKernel<82>},
            {83, (void *) hedetniemiShortestPathLengthsKernel<83>},
            {84, (void *) hedetniemiShortestPathLengthsKernel<84>},
            {85, (void *) hedetniemiShortestPathLengthsKernel<85>},
            {86, (void *) hedetniemiShortestPathLengthsKernel<86>},
            {87, (void *) hedetniemiShortestPathLengthsKernel<87>},
            {88, (void *) hedetniemiShortestPathLengthsKernel<88>},
            {89, (void *) hedetniemiShortestPathLengthsKernel<89>},
            {90, (void *) hedetniemiShortestPathLengthsKernel<90>},
            {91, (void *) hedetniemiShortestPathLengthsKernel<91>},
            {92, (void *) hedetniemiShortestPathLengthsKernel<92>},
            {93, (void *) hedetniemiShortestPathLengthsKernel<93>},
            {94, (void *) hedetniemiShortestPathLengthsKernel<94>},
            {95, (void *) hedetniemiShortestPathLengthsKernel<95>},
            {96, (void *) hedetniemiShortestPathLengthsKernel<96>},
            {97, (void *) hedetniemiShortestPathLengthsKernel<97>},
            {98, (void *) hedetniemiShortestPathLengthsKernel<98>},
            {99, (void *) hedetniemiShortestPathLengthsKernel<99>}
    };

};

std::map<int, void *> hedetniemiShortestPathsKernels() {
    return std::map<int, void *>{
            {1,  (void *) hedetniemiShortestPathsKernel<1>},
            {2,  (void *) hedetniemiShortestPathsKernel<2>},
            {3,  (void *) hedetniemiShortestPathsKernel<3>},
            {4,  (void *) hedetniemiShortestPathsKernel<4>},
            {5,  (void *) hedetniemiShortestPathsKernel<5>},
            {6,  (void *) hedetniemiShortestPathsKernel<6>},
            {7,  (void *) hedetniemiShortestPathsKernel<7>},
            {8,  (void *) hedetniemiShortestPathsKernel<8>},
            {9,  (void *) hedetniemiShortestPathsKernel<9>},
            {10, (void *) hedetniemiShortestPathsKernel<10>},
            {11, (void *) hedetniemiShortestPathsKernel<11>},
            {12, (void *) hedetniemiShortestPathsKernel<12>},
            {13, (void *) hedetniemiShortestPathsKernel<13>},
            {14, (void *) hedetniemiShortestPathsKernel<14>},
            {15, (void *) hedetniemiShortestPathsKernel<15>},
            {16, (void *) hedetniemiShortestPathsKernel<16>},
            {17, (void *) hedetniemiShortestPathsKernel<17>},
            {18, (void *) hedetniemiShortestPathsKernel<18>},
            {19, (void *) hedetniemiShortestPathsKernel<19>},
            {20, (void *) hedetniemiShortestPathsKernel<20>},
            {21, (void *) hedetniemiShortestPathsKernel<21>},
            {22, (void *) hedetniemiShortestPathsKernel<22>},
            {23, (void *) hedetniemiShortestPathsKernel<23>},
            {24, (void *) hedetniemiShortestPathsKernel<24>},
            {25, (void *) hedetniemiShortestPathsKernel<25>},
            {26, (void *) hedetniemiShortestPathsKernel<26>},
            {27, (void *) hedetniemiShortestPathsKernel<27>},
            {28, (void *) hedetniemiShortestPathsKernel<28>},
            {29, (void *) hedetniemiShortestPathsKernel<29>},
            {30, (void *) hedetniemiShortestPathsKernel<30>},
            {31, (void *) hedetniemiShortestPathsKernel<31>},
            {32, (void *) hedetniemiShortestPathsKernel<32>},
            {33, (void *) hedetniemiShortestPathsKernel<33>},
            {34, (void *) hedetniemiShortestPathsKernel<34>},
            {35, (void *) hedetniemiShortestPathsKernel<35>},
            {36, (void *) hedetniemiShortestPathsKernel<36>},
            {37, (void *) hedetniemiShortestPathsKernel<37>},
            {38, (void *) hedetniemiShortestPathsKernel<38>},
            {39, (void *) hedetniemiShortestPathsKernel<39>},
            {40, (void *) hedetniemiShortestPathsKernel<40>},
            {41, (void *) hedetniemiShortestPathsKernel<41>},
            {42, (void *) hedetniemiShortestPathsKernel<42>},
            {43, (void *) hedetniemiShortestPathsKernel<43>},
            {44, (void *) hedetniemiShortestPathsKernel<44>},
            {45, (void *) hedetniemiShortestPathsKernel<45>},
            {46, (void *) hedetniemiShortestPathsKernel<46>},
            {47, (void *) hedetniemiShortestPathsKernel<47>},
            {48, (void *) hedetniemiShortestPathsKernel<48>},
            {49, (void *) hedetniemiShortestPathsKernel<49>},
            {50, (void *) hedetniemiShortestPathsKernel<50>},
            {51, (void *) hedetniemiShortestPathsKernel<51>},
            {52, (void *) hedetniemiShortestPathsKernel<52>},
            {53, (void *) hedetniemiShortestPathsKernel<53>},
            {54, (void *) hedetniemiShortestPathsKernel<54>},
            {55, (void *) hedetniemiShortestPathsKernel<55>},
            {56, (void *) hedetniemiShortestPathsKernel<56>},
            {57, (void *) hedetniemiShortestPathsKernel<57>},
            {58, (void *) hedetniemiShortestPathsKernel<58>},
            {59, (void *) hedetniemiShortestPathsKernel<59>},
            {60, (void *) hedetniemiShortestPathsKernel<60>},
            {61, (void *) hedetniemiShortestPathsKernel<61>},
            {62, (void *) hedetniemiShortestPathsKernel<62>},
            {63, (void *) hedetniemiShortestPathsKernel<63>},
            {64, (void *) hedetniemiShortestPathsKernel<64>},
            {65, (void *) hedetniemiShortestPathsKernel<65>},
            {66, (void *) hedetniemiShortestPathsKernel<66>},
            {67, (void *) hedetniemiShortestPathsKernel<67>},
            {68, (void *) hedetniemiShortestPathsKernel<68>},
            {69, (void *) hedetniemiShortestPathsKernel<69>},
            {70, (void *) hedetniemiShortestPathsKernel<70>},
            {71, (void *) hedetniemiShortestPathsKernel<71>},
            {72, (void *) hedetniemiShortestPathsKernel<72>},
            {73, (void *) hedetniemiShortestPathsKernel<73>},
            {74, (void *) hedetniemiShortestPathsKernel<74>},
            {75, (void *) hedetniemiShortestPathsKernel<75>},
            {76, (void *) hedetniemiShortestPathsKernel<76>},
            {77, (void *) hedetniemiShortestPathsKernel<77>},
            {78, (void *) hedetniemiShortestPathsKernel<78>},
            {79, (void *) hedetniemiShortestPathsKernel<79>},
            {80, (void *) hedetniemiShortestPathsKernel<80>},
            {81, (void *) hedetniemiShortestPathsKernel<81>},
            {82, (void *) hedetniemiShortestPathsKernel<82>},
            {83, (void *) hedetniemiShortestPathsKernel<83>},
            {84, (void *) hedetniemiShortestPathsKernel<84>},
            {85, (void *) hedetniemiShortestPathsKernel<85>},
            {86, (void *) hedetniemiShortestPathsKernel<86>},
            {87, (void *) hedetniemiShortestPathsKernel<87>},
            {88, (void *) hedetniemiShortestPathsKernel<88>},
            {89, (void *) hedetniemiShortestPathsKernel<89>},
            {90, (void *) hedetniemiShortestPathsKernel<90>},
            {91, (void *) hedetniemiShortestPathsKernel<91>},
            {92, (void *) hedetniemiShortestPathsKernel<92>},
            {93, (void *) hedetniemiShortestPathsKernel<93>},
            {94, (void *) hedetniemiShortestPathsKernel<94>},
            {95, (void *) hedetniemiShortestPathsKernel<95>},
            {96, (void *) hedetniemiShortestPathsKernel<96>},
            {97, (void *) hedetniemiShortestPathsKernel<97>},
            {98, (void *) hedetniemiShortestPathsKernel<98>},
            {99, (void *) hedetniemiShortestPathsKernel<99>}
    };

};


void hedetniemiShortestPathsGpu(
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
    int n = A->n();
    A->allocateDevice();
    H->allocateDevice();
    paths->allocateDevice();
    A->toDevice();
    H->toDevice();
    paths->toDevice();
    //Calculate the number of blocks and threads for a 2D grid
    int threadsPerBlock = 16;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid(blocksPerGrid, blocksPerGrid);
    auto kernel = hedetniemiShortestPathsKernels()[maxPathLength];
    void *args[] = {&A->device(), &H->device(), &paths->device(), &n};
    cudaLaunchKernel((void *) kernel, grid, block, args);
    H->toHost();
    paths->toHost();
}

void hedetniemiShortestPathLengthsGpu(
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
    int n = A->n();
    A->allocateDevice();
    H->allocateDevice();
    A->toDevice();
    H->toDevice();
    int threadsPerBlock = 16;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid(blocksPerGrid, blocksPerGrid);
    auto kernel = hedetniemiShortestPathsKernels()[maxPathLength];
    void *args[] = {&A->device(), &H->device(), &n};
    cudaLaunchKernel((void *) kernel, grid, block, args);
    H->toHost();
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
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid(blocksPerGrid, blocksPerGrid);
    auto kernel = correlationToAdjacencyKernel;
    void *args[] = {&A->device(), &C->device(), &n};
    cudaLaunchKernel((void *) kernel, grid, block, args);
    A->toHost();
}