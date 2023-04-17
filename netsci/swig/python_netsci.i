%module netsci

%include "std_string.i"
%{
#define SWIG_FILE_WITH_INIT
#include "mutual_information.h"
#include "generalized_correlation.h"
%}

%include mutual_information.h
%include generalized_correlation.h
