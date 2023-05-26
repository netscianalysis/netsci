%include "numpy.i"
%init %{
    import_array();
%}


%fragment("FreeCap", "header") {
void FreeCap (PyObject *cap) {
    void *array = (void *) PyCapsule_GetPointer(
            cap,
            NULL
    );
    if (array
        != NULL) {
        free(array);
    }
}

}

%typemap(in, numinputs = 0
) (
int **NUMPY_ARRAY,
int    **NUMPY_ARRAY_DIM1
)
(
int *NUMPY_ARRAY_tmp,
int    *NUMPY_ARRAY_DIM1_tmp
) {
$1 = &NUMPY_ARRAY_tmp;
$2 = &NUMPY_ARRAY_DIM1_tmp;
}

%typemap(argout, fragment = "FreeCap"
) (
int **NUMPY_ARRAY,
int    **NUMPY_ARRAY_DIM1
) {
npy_intp dims[1] = {(*($2))[0]};
PyObject      *obj   = PyArray_SimpleNewFromData(
        1,
        dims,
        NPY_INT32,
        (void *) (*$1));
PyArrayObject *array = (PyArrayObject *) obj;
PyObject      *cap   = PyCapsule_New((void *) (*$1),
                                     NULL,
                                     FreeCap
);
PyArray_SetBaseObject(array, cap
);
$result = SWIG_Python_AppendOutput(
        $result,
        obj
);
free(*$2);
}
