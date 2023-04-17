

%rename(fromNumpy1D) CuArray<float>::fromNumpy(
        float *NUMPY_ARRAY,
        int NUMPY_ARRAY_DIM1
);
%rename(fromNumpy2D) CuArray<float>::fromNumpy(
        float *NUMPY_ARRAY,
        int NUMPY_ARRAY_DIM1,
        int NUMPY_ARRAY_DIM2
);
%rename(fromNumpy1D) CuArray<int>::fromNumpy(
        int *NUMPY_ARRAY,
        int NUMPY_ARRAY_DIM1
);
%rename(fromNumpy2D) CuArray<int>::fromNumpy(
        int *NUMPY_ARRAY,
        int NUMPY_ARRAY_DIM1,
        int NUMPY_ARRAY_DIM2
);
%rename(toNumpy1D) CuArray<float>::toNumpy(
        float **NUMPY_ARRAY,
        int **NUMPY_ARRAY_DIM1
);
%rename(toNumpy2D) CuArray<float>::toNumpy(
        float **NUMPY_ARRAY,
        int **NUMPY_ARRAY_DIM1,
        int **NUMPY_ARRAY_DIM2
);
%rename(toNumpy1D) CuArray<int>::toNumpy(
        int **NUMPY_ARRAY,
        int **NUMPY_ARRAY_DIM1
);
%rename(toNumpy2D) CuArray<int>::toNumpy(
        int **NUMPY_ARRAY,
        int **NUMPY_ARRAY_DIM1,
        int **NUMPY_ARRAY_DIM2
);
%rename(fromCuArray) fromCuArrayDeepCopy;