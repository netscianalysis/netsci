
SET(CLEAN_FILES_AND_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/*
    ${CMAKE_CURRENT_SOURCE_DIR}/python/build
    ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc.egg-info
    ${CMAKE_CURRENT_SOURCE_DIR}/python/dist
    ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc/*.so*
    ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc/netcalc.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc/*.cxx*
    )
SET_DIRECTORY_PROPERTIES(
        PROPERTIES
        ADDITIONAL_CLEAN_FILES "${CLEAN_FILES_AND_DIRS}"
        )

SET_SOURCE_FILES_PROPERTIES(
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netcalc.i
        PROPERTIES
        CPLUSPLUS ON
        SWIG_MODULE_NAME netcalc
        )

ADD_LIBRARY(netcalc SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mutual_information_gpu.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/math_utils_gpu.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/math_utils.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/generalized_correlation_gpu.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mutual_information_cpu.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/generalized_correlation_cpu.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mutual_information.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/hedetniemi_cpu.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/hedetniemi.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/hedetniemi_gpu.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/generalized_correlation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/psi.cpp
            )
ADD_DEPENDENCIES(netcalc cuarray)
SET_TARGET_PROPERTIES(
        netcalc
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${NETCALC_LINK_LIBRARIES}"
        )
IF (${BUILD_PYTHON_BINDINGS})
    SWIG_ADD_LIBRARY(python_netcalc
                     LANGUAGE python
                     OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc
                     SOURCES
                     ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netcalc.i
                     )
    ADD_DEPENDENCIES(python_netcalc netcalc)
    SET_TARGET_PROPERTIES(
            python_netcalc
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc
            LINK_LIBRARIES "${NETCALC_PYTHON_LINK_LIBRARIES}"
            )

    SET(PIP_UNINSTALL_NETCALC_ARGS "uninstall" "netcalc" "-y")
    SET(PYTHON_NETCALC_INSTALL_ARGS "setup.py" "install")
    ADD_CUSTOM_TARGET(PythonNetCalcInstall
                      COMMAND pip ${PIP_UNINSTALL_NETCALC_ARGS}
                      COMMAND ${Python3_EXECUTABLE} ${PYTHON_NETCALC_INSTALL_ARGS}
                      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/netcalc/python
                      )
ENDIF (${BUILD_PYTHON_BINDINGS})

