%module netsci

%{
#define SWIG_FILE_WITH_INIT
#include "cuarray.h"
%}

%include "numpy.i"
%init %{
import_array();
%}
%include "python_cuarray_typemaps.i"

%include cuarray.h

%template(IntCuArray) CuArray<int>;
%template(FloatCuArray) CuArray<float>;
%template(DoubleCuArray) CuArray<double>;
