SET(CLEAN_FILES_AND_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/*
    ${CMAKE_CURRENT_SOURCE_DIR}/build
    ${CMAKE_CURRENT_SOURCE_DIR}/tcl/*
    )
SET_DIRECTORY_PROPERTIES(
        PROPERTIES
        ADDITIONAL_CLEAN_FILES "${CLEAN_FILES_AND_DIRS}"
        )
SET_SOURCE_FILES_PROPERTIES(
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/tcl_netvis.i
        PROPERTIES
        CPLUSPLUS ON
        SWIG_MODULE_NAME netvis
        )

ADD_LIBRARY(netvis SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/path.cpp
            )
ADD_DEPENDENCIES(netvis cuarray)
SET_TARGET_PROPERTIES(
        netvis
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${NETVIS_LINK_LIBRARIES}"
        )

IF (${BUILD_TCL_BINDINGS})
    SWIG_ADD_LIBRARY(tcl_netvis
                     LANGUAGE tcl
                     OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tcl
                     SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/swig/tcl_netvis.i
                     )
    SET_TARGET_PROPERTIES(
            tcl_netvis
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tcl
            LINK_LIBRARIES "${NETVIS_TCL_LINK_LIBRARIES}"
            PREFIX ""
            OUTPUT_NAME "netvis"
            SUFFIX ".so"
            )
    SWIG_LINK_LIBRARIES(tcl_netvis netvis)
    IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tcl/tcl_netvisTCL_wrap.cxx)
        FILE(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/tcl/tcl_netvisTCL_wrap.cxx)
    ENDIF ()
    IF (${CREATE_VMD_TCL_PACKAGE})
        SET(ENV{VMD_TCL_INDEX_PATH} "${VMD_LIB_DIR}/scripts/vmd/tclIndex")
        SET(ENV{TCL_NETVIS_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/tcl")
        SET(ENV{TCL_NETVIS_LIB_PATH} "$ENV{TCL_NETVIS_DIR}/netvis.so")
        SET(ENV{TCL_CUARRAY_LIB_PATH} "${CMAKE_SOURCE_DIR}/cuarray/tcl/cuarray.so")
        EXECUTE_PROCESS(
                COMMAND tclsh ${CMAKE_CURRENT_SOURCE_DIR}/tcl/create_vmd_package.tcl
                )
    ENDIF (${CREATE_VMD_TCL_PACKAGE})
ENDIF (${BUILD_TCL_BINDINGS})
