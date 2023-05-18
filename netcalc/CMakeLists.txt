
set(CLEAN_FILES_AND_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/*
        ${CMAKE_CURRENT_SOURCE_DIR}/python/build
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc.egg-info
        ${CMAKE_CURRENT_SOURCE_DIR}/python/dist
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc/*.so*
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc/netcalc.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc/*.cxx*
        )
set_directory_properties(
        PROPERTIES
        ADDITIONAL_CLEAN_FILES "${CLEAN_FILES_AND_DIRS}"
)

set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netcalc.i
        PROPERTIES
        CPLUSPLUS ON
        SWIG_MODULE_NAME netcalc
)

add_library(netcalc SHARED
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
add_dependencies(netcalc cuarray)
set_target_properties(
        netcalc
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${NETCALC_LINK_LIBRARIES}"
)
swig_add_library(python_netcalc
        LANGUAGE python
        OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netcalc.i
        )
set_target_properties(
        python_netcalc
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/netcalc
        LINK_LIBRARIES "${NETCALC_PYTHON_LINK_LIBRARIES}"
)

set(PIP_UNINSTALL_NETCALC_ARGS "uninstall" "netcalc" "-y")
set(PYTHON_NETCALC_INSTALL_ARGS "setup.py" "install")
add_custom_target(PythonNetCalcInstall
        COMMAND pip ${PIP_UNINSTALL_NETCALC_ARGS}
        COMMAND ${Python3_EXECUTABLE} ${PYTHON_NETCALC_INSTALL_ARGS}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/netcalc/python
        )
