SET(CLEAN_FILES_AND_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/*
    ${CMAKE_CURRENT_SOURCE_DIR}/python/build
    ${CMAKE_CURRENT_SOURCE_DIR}/python/cuarray.egg-info
    ${CMAKE_CURRENT_SOURCE_DIR}/python/dist
    ${CMAKE_CURRENT_SOURCE_DIR}/python/cuarray/*.so*
    ${CMAKE_CURRENT_SOURCE_DIR}/python/cuarray/cuarray.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/cuarray/*.cxx*
    ${CMAKE_CURRENT_SOURCE_DIR}/tcl/*
    )
SET_DIRECTORY_PROPERTIES(
        PROPERTIES
        ADDITIONAL_CLEAN_FILES "${CLEAN_FILES_AND_DIRS}"
        )
SET_SOURCE_FILES_PROPERTIES(
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_cuarray.i
        PROPERTIES
        CPLUSPLUS ON
        SWIG_MODULE_NAME cuarray
        )

SET_SOURCE_FILES_PROPERTIES(
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/tcl_cuarray.i
        PROPERTIES
        CPLUSPLUS ON
        SWIG_MODULE_NAME cuarray
        )

ADD_LIBRARY(cuarray SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cuarray.cu
            )
ADD_DEPENDENCIES(cuarray cnpy)
SET_TARGET_PROPERTIES(
        cuarray
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${CUARRAY_LINK_LIBRARIES}"
        )
IF (${BUILD_PYTHON_BINDINGS})
    SWIG_ADD_LIBRARY(python_cuarray
                     LANGUAGE python
                     OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/cuarray
                     SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_cuarray.i
                     )
    SET_TARGET_PROPERTIES(
            python_cuarray
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/cuarray
            LINK_LIBRARIES "${CUARRAY_PYTHON_LINK_LIBRARIES}"
            )
    SET(PIP_UNINSTALL_CUARRAY_ARGS "uninstall" "cuarray" "-y")
    SET(PYTHON_CUARRAY_INSTALL_ARGS "setup.py" "install")
    ADD_CUSTOM_TARGET(PythonCuArrayInstall
                      COMMAND pip ${PIP_UNINSTALL_CUARRAY_ARGS}
                      COMMAND ${Python3_EXECUTABLE} ${PYTHON_CUARRAY_INSTALL_ARGS}
                      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/cuarray/python
                      )
ENDIF ()

IF (${BUILD_TCL_BINDINGS})
    SWIG_ADD_LIBRARY(tcl_cuarray
                     LANGUAGE tcl
                     OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tcl
                     SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/swig/tcl_cuarray.i
                     )
    SET_TARGET_PROPERTIES(
            tcl_cuarray
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tcl
            LINK_LIBRARIES "${CUARRAY_TCL_LINK_LIBRARIES}"
            PREFIX ""
            OUTPUT_NAME "cuarray"
            SUFFIX ".so"
            )
    SWIG_LINK_LIBRARIES(tcl_cuarray cuarray)
    IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tcl/tcl_cuarrayTCL_wrap.cxx)
        FILE(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/tcl/tcl_cuarrayTCL_wrap.cxx)
    ENDIF ()
    IF (${CREATE_VMD_TCL_PACKAGE})
        SET(ENV{VMD_TCL_INDEX_PATH} "${VMD_LIB_DIR}/scripts/vmd/tclIndex")
        SET(ENV{TCL_CUARRAY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/tcl")
        SET(ENV{TCL_CUARRAY_LIB_PATH} "$ENV{TCL_CUARRAY_DIR}/cuarray.so")
        SET(ENV{TCL_CUARRAY_SCRIPT_PATH} "$ENV{TCL_CUARRAY_DIR}/cuarray.tcl")
        EXECUTE_PROCESS(
                COMMAND tclsh ${CMAKE_CURRENT_SOURCE_DIR}/tcl/create_vmd_package.tcl
                )
    ENDIF (${CREATE_VMD_TCL_PACKAGE})
ENDIF (${BUILD_TCL_BINDINGS})
