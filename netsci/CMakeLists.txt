set_directory_properties(
        PROPERTIES
        ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/lib/*"
)

set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netsci.i
        PROPERTIES
        CPLUSPLUS ON
        SWIG_MODULE_NAME netsci
)

add_library(netsci SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_mutual_information.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_mutual_information.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/psi.cpp
        )
add_dependencies(netsci cuarray)
set_target_properties(
        netsci
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 61
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${NETSCI_LINK_LIBRARIES}"
)
swig_add_library(python_netsci
        LANGUAGE python
        OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netsci.i
        )
set_target_properties(
        python_netsci
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${NETSCI_PYTHON_LINK_LIBRARIES}"
)
