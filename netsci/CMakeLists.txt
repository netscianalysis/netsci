
set(CLEAN_FILES_AND_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/*
        ${CMAKE_CURRENT_SOURCE_DIR}/python/build
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netsci.egg-info
        ${CMAKE_CURRENT_SOURCE_DIR}/python/dist
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netsci/*.so*
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netsci/netsci.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netsci/*.cxx*
        )
set_directory_properties(
        PROPERTIES
        ADDITIONAL_CLEAN_FILES "${CLEAN_FILES_AND_DIRS}"
)

set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netsci.i
        PROPERTIES
        CPLUSPLUS ON
        SWIG_MODULE_NAME netsci
)

add_library(netsci SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_mutual_information.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_generalized_correlation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_mutual_information.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_generalized_correlation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mutual_information.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/generalized_correlation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/psi.cpp
        )
add_dependencies(netsci cuarray)
set_target_properties(
        netsci
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${NETSCI_LINK_LIBRARIES}"
)
swig_add_library(python_netsci
        LANGUAGE python
        OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/netsci
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netsci.i
        )
set_target_properties(
        python_netsci
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/netsci
        LINK_LIBRARIES "${NETSCI_PYTHON_LINK_LIBRARIES}"
)

set(PIP_UNINSTALL_NETSCI_ARGS "uninstall" "netsci" "-y")
set(PYTHON_NETSCI_INSTALL_ARGS "setup.py" "install")
add_custom_target(PythonNetSciInstall
        COMMAND pip ${PIP_UNINSTALL_NETSCI_ARGS}
        COMMAND ${Python3_EXECUTABLE} ${PYTHON_NETSCI_INSTALL_ARGS}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/netsci/python
        )
