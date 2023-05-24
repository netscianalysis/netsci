#include "mutual_information.h"
#include "psi.h"
#include <map>
#include <vector>
#include <iostream>
#include <curand_kernel.h>

__global__ void initCurandKernel(
        curandState *state,
        int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, i, 0, &state[i]);
}



__device__ void warpReduce(
        volatile int *s_a,
        int localThreadIndex
) {
    s_a[localThreadIndex] += s_a[localThreadIndex + 32];
    s_a[localThreadIndex] += s_a[localThreadIndex + 16];
    s_a[localThreadIndex] += s_a[localThreadIndex + 8];
    s_a[localThreadIndex] += s_a[localThreadIndex + 4];
    s_a[localThreadIndex] += s_a[localThreadIndex + 2];
    s_a[localThreadIndex] += s_a[localThreadIndex + 1];
}

__device__ void minWarpReduce(
        volatile float *s_min,
        volatile int *s_argMin,
        int localThreadIndex
) {
    if (s_min[localThreadIndex] > s_min[localThreadIndex + 32]) {
        s_min[localThreadIndex] = s_min[localThreadIndex + 32];
        s_argMin[localThreadIndex] = s_argMin[localThreadIndex + 32];
    }
    if (s_min[localThreadIndex] > s_min[localThreadIndex + 16]) {
        s_min[localThreadIndex] = s_min[localThreadIndex + 16];
        s_argMin[localThreadIndex] = s_argMin[localThreadIndex + 16];
    }
    if (s_min[localThreadIndex] > s_min[localThreadIndex + 8]) {
        s_min[localThreadIndex] = s_min[localThreadIndex + 8];
        s_argMin[localThreadIndex] = s_argMin[localThreadIndex + 8];
    }
    if (s_min[localThreadIndex] > s_min[localThreadIndex + 4]) {
        s_min[localThreadIndex] = s_min[localThreadIndex + 4];
        s_argMin[localThreadIndex] = s_argMin[localThreadIndex + 4];
    }
    if (s_min[localThreadIndex] > s_min[localThreadIndex + 2]) {
        s_min[localThreadIndex] = s_min[localThreadIndex + 2];
        s_argMin[localThreadIndex] = s_argMin[localThreadIndex + 2];
    }
    if (s_min[localThreadIndex] > s_min[localThreadIndex + 1]) {
        s_min[localThreadIndex] = s_min[localThreadIndex + 1];
        s_argMin[localThreadIndex] = s_argMin[localThreadIndex + 1];
    }
}


template<int XRegisterSize>
__global__ void mutualInformation2X1D_kernel(
        const float *Xa,
        const float *Xb,
        int k,
        int n,
        int *nXa,
        int *nXb
) {
    __shared__ int s_argMin[1024];
    __shared__  float s_min[1024];
    __shared__  float s_epsXa[1];
    __shared__  float s_epsXb[1];
    __shared__ int s_nXa[1024];
    __shared__ int s_nXb[1024];
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float r_Xa[XRegisterSize];
        float r_Xb[XRegisterSize];
        float r_Xai = Xa[i];
        float r_Xbi = Xb[i];
        unsigned int localThreadIndex = threadIdx.x;
        unsigned int index = localThreadIndex;
        s_nXa[localThreadIndex] = 0;
        s_nXb[localThreadIndex] = 0;
        if (localThreadIndex == 0) {
            s_epsXa[localThreadIndex] = 0.0;
            s_epsXb[localThreadIndex] = 0.0;
        }
        int registerIndex = 0;
        while (index < n) {
            r_Xa[registerIndex] = Xa[index];
            r_Xb[registerIndex] = Xb[index];
            index += blockDim.x;
            registerIndex++;
        }
        __syncthreads();
        for (int j = 0; j < k + 1; j++) {
            index = localThreadIndex;
            auto localMin = (float) RAND_MAX;
            int localArgMin = 0;
            registerIndex = 0;
            while (index < n) {
                float dX = fmax(
                        abs(r_Xai - r_Xa[registerIndex]),
                        abs(r_Xbi - r_Xb[registerIndex])
                );
                if (dX < localMin) {
                    localMin = dX;
                    localArgMin = index;
                }
                registerIndex++;
                index += blockDim.x;
            }
            s_min[localThreadIndex] = localMin;
            s_argMin[localThreadIndex] = localArgMin;
            __syncthreads();
            if (localThreadIndex < 512) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 512]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    512];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 512];
                }
            }
            __syncthreads();
            if (localThreadIndex < 256) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 256]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    256];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 256];
                }
            }
            __syncthreads();
            if (localThreadIndex < 128) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 128]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    128];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 128];
                }
            }
            __syncthreads();
            if (localThreadIndex < 64) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 64]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    64];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 64];
                }
            }
            __syncthreads();
            if (localThreadIndex < 32) {
                minWarpReduce(s_min, s_argMin, localThreadIndex);
            }
            __syncthreads();
            if ((s_argMin[0] -
                 ((s_argMin[0] / blockDim.x) * blockDim.x)) ==
                localThreadIndex) {
                if (j > 0) {
                    s_epsXa[0] = fmax(s_epsXa[0], abs(r_Xai -
                                                      r_Xa[s_argMin[0] /
                                                           blockDim.x]));
                    s_epsXb[0] = fmax(s_epsXb[0], abs(r_Xbi -
                                                      r_Xb[s_argMin[0] /
                                                           blockDim.x]));
                }
                r_Xa[s_argMin[0] / blockDim.x] = (float) RAND_MAX;
                r_Xb[s_argMin[0] / blockDim.x] = (float) RAND_MAX;
            }
            __syncthreads();
        }
        index = localThreadIndex;
        registerIndex = 0;
        while (index < n) {
            s_nXa[localThreadIndex] += (
                                               abs(r_Xa[registerIndex] -
                                                   r_Xai) <= s_epsXa[0]
                                       ) ? 1 : 0;
            s_nXb[localThreadIndex] += (
                                               abs(r_Xb[registerIndex] -
                                                   r_Xbi) <= s_epsXb[0]
                                       ) ? 1 : 0;
            registerIndex++;
            index += blockDim.x;
        }
        __syncthreads();
        if (localThreadIndex < 512) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 512];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 512];
        }
        __syncthreads();
        if (localThreadIndex < 256) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 256];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 256];
        }
        __syncthreads();
        if (localThreadIndex < 128) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 128];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 128];
        }
        __syncthreads();
        if (localThreadIndex < 64) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 64];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 64];
        }
        __syncthreads();

        if (localThreadIndex < 32) {
            warpReduce(s_nXa, localThreadIndex);
            warpReduce(s_nXb, localThreadIndex);
        }
        if (localThreadIndex == 0) {
            nXa[i] = s_nXa[0] + k;
            nXb[i] = s_nXb[0] + k;
        }
        __syncthreads();
    }
}


template<int XRegisterSize>
__global__ void mutualInformation2X2D_kernel(
        const float *Xa,
        const float *Xb,
        int k,
        int n,
        int *nXa,
        int *nXb
) {
    __shared__ int s_argMin[1024];
    __shared__  float s_min[1024];
    __shared__  float s_epsXa[1];
    __shared__  float s_epsXb[1];
    __shared__ int s_nXa[1024];
    __shared__ int s_nXb[1024];
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float r_Xa[2 * XRegisterSize];
        float r_Xb[2 * XRegisterSize];
        float r_Xai[2] = {
                Xa[i], Xa[i + n]
        };
        float r_Xbi[2] = {
                Xb[i], Xb[i + n]
        };
        unsigned int localThreadIndex = threadIdx.x;
        unsigned int index = localThreadIndex;
        s_nXa[localThreadIndex] = 0;
        s_nXb[localThreadIndex] = 0;
        if (localThreadIndex == 0) {
            s_epsXa[localThreadIndex] = 0.0;
            s_epsXb[localThreadIndex] = 0.0;
        }
        int registerIndex = 0;
        while (index < n) {
            r_Xa[registerIndex] = Xa[index];
            r_Xa[registerIndex + XRegisterSize] = Xa[index + n];
            r_Xb[registerIndex] = Xb[index];
            r_Xb[registerIndex + XRegisterSize] = Xb[index + n];
            index += blockDim.x;
            registerIndex++;
        }
        __syncthreads();
        for (int j = 0; j < k + 1; j++) {
            index = localThreadIndex;
            auto localMin = (float) RAND_MAX;
            int localArgMin = 0;
            registerIndex = 0;
            while (index < n) {
                float dX = fmax(
                        fmax(
                                abs(r_Xai[0] - r_Xa[registerIndex]),
                                abs(r_Xai[1] -
                                    r_Xa[registerIndex + XRegisterSize])
                        ),
                        fmax(
                                abs(r_Xbi[0] - r_Xb[registerIndex]),
                                abs(r_Xbi[1] -
                                    r_Xb[registerIndex + XRegisterSize])
                        )
                );
                if (dX < localMin) {
                    localMin = dX;
                    localArgMin = index;
                }
                registerIndex++;
                index += blockDim.x;
            }
            s_min[localThreadIndex] = localMin;
            s_argMin[localThreadIndex] = localArgMin;
            __syncthreads();
            if (localThreadIndex < 512) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 512]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    512];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 512];
                }
            }
            __syncthreads();
            if (localThreadIndex < 256) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 256]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    256];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 256];
                }
            }
            __syncthreads();
            if (localThreadIndex < 128) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 128]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    128];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 128];
                }
            }
            __syncthreads();
            if (localThreadIndex < 64) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 64]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    64];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 64];
                }
            }
            __syncthreads();
            if (localThreadIndex < 32) {
                minWarpReduce(s_min, s_argMin, localThreadIndex);
            }
            __syncthreads();
            if ((s_argMin[0] -
                 ((s_argMin[0] / blockDim.x) * blockDim.x)) ==
                localThreadIndex) {
                if (j > 0) {
                    s_epsXa[0] = fmax(
                            s_epsXa[0],
                            abs(r_Xai[0]
                                - r_Xa[s_argMin[0] / blockDim.x]
                            )
                    );
                    s_epsXa[0] = fmax(
                            s_epsXa[0],
                            abs(r_Xai[1]
                                - r_Xa[(s_argMin[0] / blockDim.x) +
                                       XRegisterSize]
                            )
                    );
                    s_epsXb[0] = fmax(
                            s_epsXb[0],
                            abs(r_Xbi[0]
                                - r_Xb[s_argMin[0] / blockDim.x]
                            )
                    );
                    s_epsXb[0] = fmax(
                            s_epsXb[0],
                            abs(r_Xbi[1]
                                - r_Xb[s_argMin[0] / blockDim.x +
                                       XRegisterSize]
                            )
                    );
                }
                r_Xa[s_argMin[0] / blockDim.x] = (float) RAND_MAX;
                r_Xa[s_argMin[0] / blockDim.x +
                     XRegisterSize] = (float) RAND_MAX;
                r_Xb[s_argMin[0] / blockDim.x] = (float) RAND_MAX;
                r_Xb[s_argMin[0] / blockDim.x +
                     XRegisterSize] = (float) RAND_MAX;
            }
            __syncthreads();
        }
        index = localThreadIndex;
        registerIndex = 0;
        while (index < n) {
            s_nXa[localThreadIndex] += (fmax(abs(r_Xa[registerIndex]
                                                 - r_Xai[0]),
                                             abs(r_Xa[registerIndex +
                                                      XRegisterSize] -
                                                 r_Xai[1])) <=
                                        s_epsXa[0]
                                       ) ? 1 : 0;
            s_nXb[localThreadIndex] += (fmax(abs(r_Xb[registerIndex]
                                                 - r_Xbi[0]),
                                             abs(r_Xb[registerIndex +
                                                      XRegisterSize] -
                                                 r_Xbi[1])) <=
                                        s_epsXb[0]
                                       ) ? 1 : 0;
            registerIndex++;
            index += blockDim.x;
        }
        __syncthreads();
        if (localThreadIndex < 512) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 512];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 512];
        }
        __syncthreads();
        if (localThreadIndex < 256) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 256];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 256];
        }
        __syncthreads();
        if (localThreadIndex < 128) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 128];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 128];
        }
        __syncthreads();
        if (localThreadIndex < 64) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 64];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 64];
        }
        __syncthreads();

        if (localThreadIndex < 32) {
            warpReduce(s_nXa, localThreadIndex);
            warpReduce(s_nXb, localThreadIndex);
        }
        if (localThreadIndex == 0) {
            nXa[i] = s_nXa[0] + k;
            nXb[i] = s_nXb[0] + k;
        }
        __syncthreads();
    }
}


template<int XRegisterSize>
__global__ void mutualInformation2X3D_kernel(
        const float *Xa,
        const float *Xb,
        int k,
        int n,
        int *nXa,
        int *nXb
) {
    __shared__ int s_argMin[1024];
    __shared__  float s_min[1024];
    __shared__  float s_epsXa[1];
    __shared__  float s_epsXb[1];
    __shared__ int s_nXa[1024];
    __shared__ int s_nXb[1024];
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float r_Xa[3 * XRegisterSize];
        float r_Xb[3 * XRegisterSize];
        float r_Xai[3] = {
                Xa[i], Xa[i + n], Xa[i + 2 * n]
        };
        float r_Xbi[3] = {
                Xb[i], Xb[i + n], Xb[i + 2 * n]
        };
        unsigned int localThreadIndex = threadIdx.x;
        unsigned int index = localThreadIndex;
        s_nXa[localThreadIndex] = 0;
        s_nXb[localThreadIndex] = 0;
        if (localThreadIndex == 0) {
            s_epsXa[localThreadIndex] = 0.0;
            s_epsXb[localThreadIndex] = 0.0;
        }
        int registerIndex = 0;
        while (index < n) {
            r_Xa[registerIndex] = Xa[index];
            r_Xa[registerIndex + XRegisterSize] = Xa[index + n];
            r_Xa[registerIndex + 2 * XRegisterSize] = Xa[index + 2 * n];
            r_Xb[registerIndex] = Xb[index];
            r_Xb[registerIndex + XRegisterSize] = Xb[index + n];
            r_Xb[registerIndex + 2 * XRegisterSize] = Xb[index + 2 * n];
            index += blockDim.x;
            registerIndex++;
        }
        __syncthreads();
        for (int j = 0; j < k + 1; j++) {
            index = localThreadIndex;
            auto localMin = (float) RAND_MAX;
            int localArgMin = 0;
            registerIndex = 0;
            while (index < n) {
                float dX = fmax(
                        fmax(
                                abs(r_Xai[0] - r_Xa[registerIndex]),
                                fmax(
                                        abs(r_Xai[1] -
                                            r_Xa[registerIndex +
                                                 XRegisterSize]),
                                        abs(r_Xai[2] -
                                            r_Xa[registerIndex +
                                                 2 *
                                                 XRegisterSize])
                                )

                        ),
                        fmax(
                                abs(r_Xbi[0] - r_Xb[registerIndex]),
                                fmax(abs(r_Xbi[1] -
                                         r_Xb[registerIndex +
                                              XRegisterSize]),
                                     abs(r_Xbi[2] -
                                         r_Xb[registerIndex +
                                              2 * XRegisterSize])
                                )
                        )
                );
                if (dX < localMin) {
                    localMin = dX;
                    localArgMin = index;
                }
                registerIndex++;
                index += blockDim.x;
            }
            s_min[localThreadIndex] = localMin;
            s_argMin[localThreadIndex] = localArgMin;
            __syncthreads();
            if (localThreadIndex < 512) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 512]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    512];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 512];
                }
            }
            __syncthreads();
            if (localThreadIndex < 256) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 256]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    256];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 256];
                }
            }
            __syncthreads();
            if (localThreadIndex < 128) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 128]) {
                    s_min[localThreadIndex] = s_min[localThreadIndex +
                                                    128];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 128];
                }
            }
            __syncthreads();
            if (localThreadIndex < 64) {
                if (s_min[localThreadIndex] >
                    s_min[localThreadIndex + 64]) {
                    s_min[localThreadIndex] = s_min[
                            localThreadIndex + 64];
                    s_argMin[localThreadIndex] = s_argMin[
                            localThreadIndex + 64];
                }
            }
            __syncthreads();
            if (localThreadIndex < 32) {
                minWarpReduce(s_min, s_argMin, localThreadIndex);
            }
            __syncthreads();
            if ((s_argMin[0] -
                 ((s_argMin[0] / blockDim.x) * blockDim.x)) ==
                localThreadIndex) {
                if (j > 0) {
                    s_epsXa[0] = fmax(
                            s_epsXa[0],
                            abs(r_Xai[0]
                                - r_Xa[s_argMin[0] / blockDim.x]
                            )
                    );
                    s_epsXa[0] = fmax(
                            s_epsXa[0],
                            abs(r_Xai[1]
                                - r_Xa[(s_argMin[0] / blockDim.x) +
                                       XRegisterSize]
                            )
                    );
                    s_epsXa[0] = fmax(
                            s_epsXa[0],
                            abs(r_Xai[2]
                                - r_Xa[(s_argMin[0] / blockDim.x) +
                                       2 * XRegisterSize]
                            )
                    );
                    s_epsXb[0] = fmax(
                            s_epsXb[0],
                            abs(r_Xbi[0]
                                - r_Xb[s_argMin[0] / blockDim.x]
                            )
                    );
                    s_epsXb[0] = fmax(
                            s_epsXb[0],
                            abs(r_Xbi[1]
                                - r_Xb[s_argMin[0] / blockDim.x +
                                       XRegisterSize]
                            )
                    );
                    s_epsXb[0] = fmax(
                            s_epsXb[0],
                            abs(r_Xbi[2]
                                - r_Xb[s_argMin[0] / blockDim.x +
                                       2 * XRegisterSize]
                            )
                    );
                }
                r_Xa[s_argMin[0] / blockDim.x] = (float) RAND_MAX;
                r_Xa[s_argMin[0] / blockDim.x +
                     XRegisterSize] = (float) RAND_MAX;
                r_Xa[s_argMin[0] / blockDim.x +
                     2 * XRegisterSize] = (float) RAND_MAX;
                r_Xb[s_argMin[0] / blockDim.x] = (float) RAND_MAX;
                r_Xb[s_argMin[0] / blockDim.x +
                     XRegisterSize] = (float) RAND_MAX;
                r_Xb[s_argMin[0] / blockDim.x +
                     2 * XRegisterSize] = (float) RAND_MAX;
            }
            __syncthreads();
        }
        index = localThreadIndex;
        registerIndex = 0;
        while (index < n) {
            s_nXa[localThreadIndex] += (fmax(abs(r_Xa[registerIndex]
                                                 - r_Xai[0]),
                                             fmax(
                                                     abs(r_Xa[registerIndex +
                                                              XRegisterSize] -
                                                         r_Xai[1]),
                                                     abs(r_Xa[registerIndex +
                                                              2 *
                                                              XRegisterSize] -
                                                         r_Xai[2])
                                             )) <=
                                        s_epsXa[0]
                                       ) ? 1 : 0;
            s_nXb[localThreadIndex] += (fmax(abs(r_Xb[registerIndex]
                                                 - r_Xbi[0]),
                                             fmax(
                                                     abs(r_Xb[registerIndex +
                                                              XRegisterSize] -
                                                         r_Xbi[1]),
                                                     abs(r_Xb[registerIndex +
                                                              2 *
                                                              XRegisterSize] -
                                                         r_Xbi[2])
                                             )) <=
                                        s_epsXb[0]
                                       ) ? 1 : 0;
            registerIndex++;
            index += blockDim.x;
        }
        __syncthreads();
        if (localThreadIndex < 512) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 512];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 512];
        }
        __syncthreads();
        if (localThreadIndex < 256) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 256];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 256];
        }
        __syncthreads();
        if (localThreadIndex < 128) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 128];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 128];
        }
        __syncthreads();
        if (localThreadIndex < 64) {
            s_nXa[localThreadIndex] += s_nXa[localThreadIndex + 64];
            s_nXb[localThreadIndex] += s_nXb[localThreadIndex + 64];
        }
        __syncthreads();

        if (localThreadIndex < 32) {
            warpReduce(s_nXa, localThreadIndex);
            warpReduce(s_nXb, localThreadIndex);
        }
        if (localThreadIndex == 0) {
            nXa[i] = s_nXa[0] + k;
            nXb[i] = s_nXb[0] + k;
        }
        __syncthreads();
    }
}

std::map<int, void *> mutualInformation2X1DKernels() {
    return std::map<int, void *>{
            {1,   (void *) mutualInformation2X1D_kernel<1>},
            {2,   (void *) mutualInformation2X1D_kernel<2>},
            {3,   (void *) mutualInformation2X1D_kernel<3>},
            {4,   (void *) mutualInformation2X1D_kernel<4>},
            {5,   (void *) mutualInformation2X1D_kernel<5>},
            {6,   (void *) mutualInformation2X1D_kernel<6>},
            {7,   (void *) mutualInformation2X1D_kernel<7>},
            {8,   (void *) mutualInformation2X1D_kernel<8>},
            {9,   (void *) mutualInformation2X1D_kernel<9>},
            {10,  (void *) mutualInformation2X1D_kernel<10>},
            {11,  (void *) mutualInformation2X1D_kernel<11>},
            {12,  (void *) mutualInformation2X1D_kernel<12>},
            {13,  (void *) mutualInformation2X1D_kernel<13>},
            {14,  (void *) mutualInformation2X1D_kernel<14>},
            {15,  (void *) mutualInformation2X1D_kernel<15>},
            {16,  (void *) mutualInformation2X1D_kernel<16>},
            {17,  (void *) mutualInformation2X1D_kernel<17>},
            {18,  (void *) mutualInformation2X1D_kernel<18>},
            {19,  (void *) mutualInformation2X1D_kernel<19>},
            {20,  (void *) mutualInformation2X1D_kernel<20>},
            {21,  (void *) mutualInformation2X1D_kernel<21>},
            {22,  (void *) mutualInformation2X1D_kernel<22>},
            {23,  (void *) mutualInformation2X1D_kernel<23>},
            {24,  (void *) mutualInformation2X1D_kernel<24>},
            {25,  (void *) mutualInformation2X1D_kernel<25>},
            {26,  (void *) mutualInformation2X1D_kernel<26>},
            {27,  (void *) mutualInformation2X1D_kernel<27>},
            {28,  (void *) mutualInformation2X1D_kernel<28>},
            {29,  (void *) mutualInformation2X1D_kernel<29>},
            {30,  (void *) mutualInformation2X1D_kernel<30>},
            {31,  (void *) mutualInformation2X1D_kernel<31>},
            {32,  (void *) mutualInformation2X1D_kernel<32>},
            {33,  (void *) mutualInformation2X1D_kernel<33>},
            {34,  (void *) mutualInformation2X1D_kernel<34>},
            {35,  (void *) mutualInformation2X1D_kernel<35>},
            {36,  (void *) mutualInformation2X1D_kernel<36>},
            {37,  (void *) mutualInformation2X1D_kernel<37>},
            {38,  (void *) mutualInformation2X1D_kernel<38>},
            {39,  (void *) mutualInformation2X1D_kernel<39>},
            {40,  (void *) mutualInformation2X1D_kernel<40>},
            {41,  (void *) mutualInformation2X1D_kernel<41>},
            {42,  (void *) mutualInformation2X1D_kernel<42>},
            {43,  (void *) mutualInformation2X1D_kernel<43>},
            {44,  (void *) mutualInformation2X1D_kernel<44>},
            {45,  (void *) mutualInformation2X1D_kernel<45>},
            {46,  (void *) mutualInformation2X1D_kernel<46>},
            {47,  (void *) mutualInformation2X1D_kernel<47>},
            {48,  (void *) mutualInformation2X1D_kernel<48>},
            {49,  (void *) mutualInformation2X1D_kernel<49>},
            {50,  (void *) mutualInformation2X1D_kernel<50>},
            {51,  (void *) mutualInformation2X1D_kernel<51>},
            {52,  (void *) mutualInformation2X1D_kernel<52>},
            {53,  (void *) mutualInformation2X1D_kernel<53>},
            {54,  (void *) mutualInformation2X1D_kernel<54>},
            {55,  (void *) mutualInformation2X1D_kernel<55>},
            {56,  (void *) mutualInformation2X1D_kernel<56>},
            {57,  (void *) mutualInformation2X1D_kernel<57>},
            {58,  (void *) mutualInformation2X1D_kernel<58>},
            {59,  (void *) mutualInformation2X1D_kernel<59>},
            {60,  (void *) mutualInformation2X1D_kernel<60>},
            {61,  (void *) mutualInformation2X1D_kernel<61>},
            {62,  (void *) mutualInformation2X1D_kernel<62>},
            {63,  (void *) mutualInformation2X1D_kernel<63>},
            {64,  (void *) mutualInformation2X1D_kernel<64>},
            {65,  (void *) mutualInformation2X1D_kernel<65>},
            {66,  (void *) mutualInformation2X1D_kernel<66>},
            {67,  (void *) mutualInformation2X1D_kernel<67>},
            {68,  (void *) mutualInformation2X1D_kernel<68>},
            {69,  (void *) mutualInformation2X1D_kernel<69>},
            {70,  (void *) mutualInformation2X1D_kernel<70>},
            {71,  (void *) mutualInformation2X1D_kernel<71>},
            {72,  (void *) mutualInformation2X1D_kernel<72>},
            {73,  (void *) mutualInformation2X1D_kernel<73>},
            {74,  (void *) mutualInformation2X1D_kernel<74>},
            {75,  (void *) mutualInformation2X1D_kernel<75>},
            {76,  (void *) mutualInformation2X1D_kernel<76>},
            {77,  (void *) mutualInformation2X1D_kernel<77>},
            {78,  (void *) mutualInformation2X1D_kernel<78>},
            {79,  (void *) mutualInformation2X1D_kernel<79>},
            {80,  (void *) mutualInformation2X1D_kernel<80>},
            {81,  (void *) mutualInformation2X1D_kernel<81>},
            {82,  (void *) mutualInformation2X1D_kernel<82>},
            {83,  (void *) mutualInformation2X1D_kernel<83>},
            {84,  (void *) mutualInformation2X1D_kernel<84>},
            {85,  (void *) mutualInformation2X1D_kernel<85>},
            {86,  (void *) mutualInformation2X1D_kernel<86>},
            {87,  (void *) mutualInformation2X1D_kernel<87>},
            {88,  (void *) mutualInformation2X1D_kernel<88>},
            {89,  (void *) mutualInformation2X1D_kernel<89>},
            {90,  (void *) mutualInformation2X1D_kernel<90>},
            {91,  (void *) mutualInformation2X1D_kernel<91>},
            {92,  (void *) mutualInformation2X1D_kernel<92>},
            {93,  (void *) mutualInformation2X1D_kernel<93>},
            {94,  (void *) mutualInformation2X1D_kernel<94>},
            {95,  (void *) mutualInformation2X1D_kernel<95>},
            {96,  (void *) mutualInformation2X1D_kernel<96>},
            {97,  (void *) mutualInformation2X1D_kernel<97>},
            {98,  (void *) mutualInformation2X1D_kernel<98>},
            {99,  (void *) mutualInformation2X1D_kernel<99>},
            {100, (void *) mutualInformation2X1D_kernel<100>},
            {101, (void *) mutualInformation2X1D_kernel<101>},
    };
}

std::map<int, void *> mutualInformation2X2DKernels() {
    return std::map<int, void *>{
            {1,   (void *) mutualInformation2X2D_kernel<1>},
            {2,   (void *) mutualInformation2X2D_kernel<2>},
            {3,   (void *) mutualInformation2X2D_kernel<3>},
            {4,   (void *) mutualInformation2X2D_kernel<4>},
            {5,   (void *) mutualInformation2X2D_kernel<5>},
            {6,   (void *) mutualInformation2X2D_kernel<6>},
            {7,   (void *) mutualInformation2X2D_kernel<7>},
            {8,   (void *) mutualInformation2X2D_kernel<8>},
            {9,   (void *) mutualInformation2X2D_kernel<9>},
            {10,  (void *) mutualInformation2X2D_kernel<10>},
            {11,  (void *) mutualInformation2X2D_kernel<11>},
            {12,  (void *) mutualInformation2X2D_kernel<12>},
            {13,  (void *) mutualInformation2X2D_kernel<13>},
            {14,  (void *) mutualInformation2X2D_kernel<14>},
            {15,  (void *) mutualInformation2X2D_kernel<15>},
            {16,  (void *) mutualInformation2X2D_kernel<16>},
            {17,  (void *) mutualInformation2X2D_kernel<17>},
            {18,  (void *) mutualInformation2X2D_kernel<18>},
            {19,  (void *) mutualInformation2X2D_kernel<19>},
            {20,  (void *) mutualInformation2X2D_kernel<20>},
            {21,  (void *) mutualInformation2X2D_kernel<21>},
            {22,  (void *) mutualInformation2X2D_kernel<22>},
            {23,  (void *) mutualInformation2X2D_kernel<23>},
            {24,  (void *) mutualInformation2X2D_kernel<24>},
            {25,  (void *) mutualInformation2X2D_kernel<25>},
            {26,  (void *) mutualInformation2X2D_kernel<26>},
            {27,  (void *) mutualInformation2X2D_kernel<27>},
            {28,  (void *) mutualInformation2X2D_kernel<28>},
            {29,  (void *) mutualInformation2X2D_kernel<29>},
            {30,  (void *) mutualInformation2X2D_kernel<30>},
            {31,  (void *) mutualInformation2X2D_kernel<31>},
            {32,  (void *) mutualInformation2X2D_kernel<32>},
            {33,  (void *) mutualInformation2X2D_kernel<33>},
            {34,  (void *) mutualInformation2X2D_kernel<34>},
            {35,  (void *) mutualInformation2X2D_kernel<35>},
            {36,  (void *) mutualInformation2X2D_kernel<36>},
            {37,  (void *) mutualInformation2X2D_kernel<37>},
            {38,  (void *) mutualInformation2X2D_kernel<38>},
            {39,  (void *) mutualInformation2X2D_kernel<39>},
            {40,  (void *) mutualInformation2X2D_kernel<40>},
            {41,  (void *) mutualInformation2X2D_kernel<41>},
            {42,  (void *) mutualInformation2X2D_kernel<42>},
            {43,  (void *) mutualInformation2X2D_kernel<43>},
            {44,  (void *) mutualInformation2X2D_kernel<44>},
            {45,  (void *) mutualInformation2X2D_kernel<45>},
            {46,  (void *) mutualInformation2X2D_kernel<46>},
            {47,  (void *) mutualInformation2X2D_kernel<47>},
            {48,  (void *) mutualInformation2X2D_kernel<48>},
            {49,  (void *) mutualInformation2X2D_kernel<49>},
            {50,  (void *) mutualInformation2X2D_kernel<50>},
            {51,  (void *) mutualInformation2X2D_kernel<51>},
            {52,  (void *) mutualInformation2X2D_kernel<52>},
            {53,  (void *) mutualInformation2X2D_kernel<53>},
            {54,  (void *) mutualInformation2X2D_kernel<54>},
            {55,  (void *) mutualInformation2X2D_kernel<55>},
            {56,  (void *) mutualInformation2X2D_kernel<56>},
            {57,  (void *) mutualInformation2X2D_kernel<57>},
            {58,  (void *) mutualInformation2X2D_kernel<58>},
            {59,  (void *) mutualInformation2X2D_kernel<59>},
            {60,  (void *) mutualInformation2X2D_kernel<60>},
            {61,  (void *) mutualInformation2X2D_kernel<61>},
            {62,  (void *) mutualInformation2X2D_kernel<62>},
            {63,  (void *) mutualInformation2X2D_kernel<63>},
            {64,  (void *) mutualInformation2X2D_kernel<64>},
            {65,  (void *) mutualInformation2X2D_kernel<65>},
            {66,  (void *) mutualInformation2X2D_kernel<66>},
            {67,  (void *) mutualInformation2X2D_kernel<67>},
            {68,  (void *) mutualInformation2X2D_kernel<68>},
            {69,  (void *) mutualInformation2X2D_kernel<69>},
            {70,  (void *) mutualInformation2X2D_kernel<70>},
            {71,  (void *) mutualInformation2X2D_kernel<71>},
            {72,  (void *) mutualInformation2X2D_kernel<72>},
            {73,  (void *) mutualInformation2X2D_kernel<73>},
            {74,  (void *) mutualInformation2X2D_kernel<74>},
            {75,  (void *) mutualInformation2X2D_kernel<75>},
            {76,  (void *) mutualInformation2X2D_kernel<76>},
            {77,  (void *) mutualInformation2X2D_kernel<77>},
            {78,  (void *) mutualInformation2X2D_kernel<78>},
            {79,  (void *) mutualInformation2X2D_kernel<79>},
            {80,  (void *) mutualInformation2X2D_kernel<80>},
            {81,  (void *) mutualInformation2X2D_kernel<81>},
            {82,  (void *) mutualInformation2X2D_kernel<82>},
            {83,  (void *) mutualInformation2X2D_kernel<83>},
            {84,  (void *) mutualInformation2X2D_kernel<84>},
            {85,  (void *) mutualInformation2X2D_kernel<85>},
            {86,  (void *) mutualInformation2X2D_kernel<86>},
            {87,  (void *) mutualInformation2X2D_kernel<87>},
            {88,  (void *) mutualInformation2X2D_kernel<88>},
            {89,  (void *) mutualInformation2X2D_kernel<89>},
            {90,  (void *) mutualInformation2X2D_kernel<90>},
            {91,  (void *) mutualInformation2X2D_kernel<91>},
            {92,  (void *) mutualInformation2X2D_kernel<92>},
            {93,  (void *) mutualInformation2X2D_kernel<93>},
            {94,  (void *) mutualInformation2X2D_kernel<94>},
            {95,  (void *) mutualInformation2X2D_kernel<95>},
            {96,  (void *) mutualInformation2X2D_kernel<96>},
            {97,  (void *) mutualInformation2X2D_kernel<97>},
            {98,  (void *) mutualInformation2X2D_kernel<98>},
            {99,  (void *) mutualInformation2X2D_kernel<99>},
            {100, (void *) mutualInformation2X2D_kernel<100>},
            {101, (void *) mutualInformation2X2D_kernel<101>},
    };
}

std::map<int, void *> mutualInformation2X3DKernels() {
    return std::map<int, void *>{
            {1,   (void *) mutualInformation2X3D_kernel<1>},
            {2,   (void *) mutualInformation2X3D_kernel<2>},
            {3,   (void *) mutualInformation2X3D_kernel<3>},
            {4,   (void *) mutualInformation2X3D_kernel<4>},
            {5,   (void *) mutualInformation2X3D_kernel<5>},
            {6,   (void *) mutualInformation2X3D_kernel<6>},
            {7,   (void *) mutualInformation2X3D_kernel<7>},
            {8,   (void *) mutualInformation2X3D_kernel<8>},
            {9,   (void *) mutualInformation2X3D_kernel<9>},
            {10,  (void *) mutualInformation2X3D_kernel<10>},
            {11,  (void *) mutualInformation2X3D_kernel<11>},
            {12,  (void *) mutualInformation2X3D_kernel<12>},
            {13,  (void *) mutualInformation2X3D_kernel<13>},
            {14,  (void *) mutualInformation2X3D_kernel<14>},
            {15,  (void *) mutualInformation2X3D_kernel<15>},
            {16,  (void *) mutualInformation2X3D_kernel<16>},
            {17,  (void *) mutualInformation2X3D_kernel<17>},
            {18,  (void *) mutualInformation2X3D_kernel<18>},
            {19,  (void *) mutualInformation2X3D_kernel<19>},
            {20,  (void *) mutualInformation2X3D_kernel<20>},
            {21,  (void *) mutualInformation2X3D_kernel<21>},
            {22,  (void *) mutualInformation2X3D_kernel<22>},
            {23,  (void *) mutualInformation2X3D_kernel<23>},
            {24,  (void *) mutualInformation2X3D_kernel<24>},
            {25,  (void *) mutualInformation2X3D_kernel<25>},
            {26,  (void *) mutualInformation2X3D_kernel<26>},
            {27,  (void *) mutualInformation2X3D_kernel<27>},
            {28,  (void *) mutualInformation2X3D_kernel<28>},
            {29,  (void *) mutualInformation2X3D_kernel<29>},
            {30,  (void *) mutualInformation2X3D_kernel<30>},
            {31,  (void *) mutualInformation2X3D_kernel<31>},
            {32,  (void *) mutualInformation2X3D_kernel<32>},
            {33,  (void *) mutualInformation2X3D_kernel<33>},
            {34,  (void *) mutualInformation2X3D_kernel<34>},
            {35,  (void *) mutualInformation2X3D_kernel<35>},
            {36,  (void *) mutualInformation2X3D_kernel<36>},
            {37,  (void *) mutualInformation2X3D_kernel<37>},
            {38,  (void *) mutualInformation2X3D_kernel<38>},
            {39,  (void *) mutualInformation2X3D_kernel<39>},
            {40,  (void *) mutualInformation2X3D_kernel<40>},
            {41,  (void *) mutualInformation2X3D_kernel<41>},
            {42,  (void *) mutualInformation2X3D_kernel<42>},
            {43,  (void *) mutualInformation2X3D_kernel<43>},
            {44,  (void *) mutualInformation2X3D_kernel<44>},
            {45,  (void *) mutualInformation2X3D_kernel<45>},
            {46,  (void *) mutualInformation2X3D_kernel<46>},
            {47,  (void *) mutualInformation2X3D_kernel<47>},
            {48,  (void *) mutualInformation2X3D_kernel<48>},
            {49,  (void *) mutualInformation2X3D_kernel<49>},
            {50,  (void *) mutualInformation2X3D_kernel<50>},
            {51,  (void *) mutualInformation2X3D_kernel<51>},
            {52,  (void *) mutualInformation2X3D_kernel<52>},
            {53,  (void *) mutualInformation2X3D_kernel<53>},
            {54,  (void *) mutualInformation2X3D_kernel<54>},
            {55,  (void *) mutualInformation2X3D_kernel<55>},
            {56,  (void *) mutualInformation2X3D_kernel<56>},
            {57,  (void *) mutualInformation2X3D_kernel<57>},
            {58,  (void *) mutualInformation2X3D_kernel<58>},
            {59,  (void *) mutualInformation2X3D_kernel<59>},
            {60,  (void *) mutualInformation2X3D_kernel<60>},
            {61,  (void *) mutualInformation2X3D_kernel<61>},
            {62,  (void *) mutualInformation2X3D_kernel<62>},
            {63,  (void *) mutualInformation2X3D_kernel<63>},
            {64,  (void *) mutualInformation2X3D_kernel<64>},
            {65,  (void *) mutualInformation2X3D_kernel<65>},
            {66,  (void *) mutualInformation2X3D_kernel<66>},
            {67,  (void *) mutualInformation2X3D_kernel<67>},
            {68,  (void *) mutualInformation2X3D_kernel<68>},
            {69,  (void *) mutualInformation2X3D_kernel<69>},
            {70,  (void *) mutualInformation2X3D_kernel<70>},
            {71,  (void *) mutualInformation2X3D_kernel<71>},
            {72,  (void *) mutualInformation2X3D_kernel<72>},
            {73,  (void *) mutualInformation2X3D_kernel<73>},
            {74,  (void *) mutualInformation2X3D_kernel<74>},
            {75,  (void *) mutualInformation2X3D_kernel<75>},
            {76,  (void *) mutualInformation2X3D_kernel<76>},
            {77,  (void *) mutualInformation2X3D_kernel<77>},
            {78,  (void *) mutualInformation2X3D_kernel<78>},
            {79,  (void *) mutualInformation2X3D_kernel<79>},
            {80,  (void *) mutualInformation2X3D_kernel<80>},
            {81,  (void *) mutualInformation2X3D_kernel<81>},
            {82,  (void *) mutualInformation2X3D_kernel<82>},
            {83,  (void *) mutualInformation2X3D_kernel<83>},
            {84,  (void *) mutualInformation2X3D_kernel<84>},
            {85,  (void *) mutualInformation2X3D_kernel<85>},
            {86,  (void *) mutualInformation2X3D_kernel<86>},
            {87,  (void *) mutualInformation2X3D_kernel<87>},
            {88,  (void *) mutualInformation2X3D_kernel<88>},
            {89,  (void *) mutualInformation2X3D_kernel<89>},
            {90,  (void *) mutualInformation2X3D_kernel<90>},
            {91,  (void *) mutualInformation2X3D_kernel<91>},
            {92,  (void *) mutualInformation2X3D_kernel<92>},
            {93,  (void *) mutualInformation2X3D_kernel<93>},
            {94,  (void *) mutualInformation2X3D_kernel<94>},
            {95,  (void *) mutualInformation2X3D_kernel<95>},
            {96,  (void *) mutualInformation2X3D_kernel<96>},
            {97,  (void *) mutualInformation2X3D_kernel<97>},
            {98,  (void *) mutualInformation2X3D_kernel<98>},
            {99,  (void *) mutualInformation2X3D_kernel<99>},
            {100, (void *) mutualInformation2X3D_kernel<100>},
            {101, (void *) mutualInformation2X3D_kernel<101>},
    };
}

float netcalc::mutualInformationGpu(
        CuArray<float> *Xa,
        CuArray<float> *Xb,
        int k,
        int n,
        int xd,
        int d
) {
    void *kernel;
    if (xd == 2 && d == 1) {
        kernel = mutualInformation2X1DKernels()[
                (n / 1024) + 1
        ];
    }
    if (xd == 2 && d == 2) {
        kernel = mutualInformation2X2DKernels()[
                (n / 1024) + 1
        ];
    }
    if (xd == 2 && d == 3) {
        kernel = mutualInformation2X3DKernels()[
                (n / 1024) + 1
        ];
    }
    auto psi = new CuArray<float>;
    psi->init(1, n + 1);
    generatePsi(
            psi,
            n
    );
    auto nXa = new CuArray<int>;
    nXa->init(1, n);
    auto nXb = new CuArray<int>;
    nXb->init(1, n);
    Xa->allocateDevice();
    Xb->allocateDevice();
    Xa->toDevice();
    Xb->toDevice();
    nXb->allocateDevice();
    nXa->allocateDevice();
    void *args[] = {
            (void *) &Xa->device(),
            (void *) &Xb->device(),
            (void *) &k,
            (void *) &n,
            (void *) &nXa->device(),
            (void *) &nXb->device()
    };
    auto gridSize = (n < 65000) ? n : 65000;
    auto blockSize = 1024;
    cudaLaunchKernel(
            kernel,
            gridSize,
            blockSize,
            args
    );
    nXa->toHost();
    nXb->toHost();
    float averageXaXbPsiK = 0.0;
    for (int i = 0; i < n; i++) {
        averageXaXbPsiK +=
                psi->get(0, nXa->host()[i])
                + psi->get(0, nXb->host()[i]);
    }
    averageXaXbPsiK /= (float) n;
    float mutualInformation =
            psi->host()[n] + psi->host()[k]
            - (float) (1.0 / (float) k) - averageXaXbPsiK;
    delete psi;
    delete nXa;
    delete nXb;
    return mutualInformation;
}


