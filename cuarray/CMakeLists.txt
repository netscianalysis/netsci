set_directory_properties(
        PROPERTIES
        ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/lib/*"
)

set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_cuarray.i
        PROPERTIES
        CPLUSPLUS ON
        SWIG_MODULE_NAME cuarray
)

add_library(cuarray SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuarray.cu
        )
add_dependencies(cuarray cnpy)
set_target_properties(
        cuarray
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${CUARRARY_LINK_LIBRARIES}"
)
swig_add_library(python_cuarray
        LANGUAGE python
        OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_cuarray.i
        )
set_target_properties(
        python_cuarray
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${CUARRARY_PYTHON_LINK_LIBRARIES}"
)