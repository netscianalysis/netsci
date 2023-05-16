
set(CLEAN_FILES_AND_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/*
        ${CMAKE_CURRENT_SOURCE_DIR}/python/build
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netchem.egg-info
        ${CMAKE_CURRENT_SOURCE_DIR}/python/dist
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netchem/*.so*
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netchem/netchem.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/netchem/*.cxx*
        )
set_directory_properties(
        PROPERTIES
        ADDITIONAL_CLEAN_FILES "${CLEAN_FILES_AND_DIRS}"
)

set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netchem.i
        PROPERTIES
        CPLUSPLUS ON
        SWIG_MODULE_NAME netchem
)

add_library(netchem SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/serializer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dcd/dcd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/atom.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/node.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/atoms.cpp
        )
add_dependencies(netchem cuarray)
set_target_properties(
        netchem
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        LINK_LIBRARIES "${NETCHEM_LINK_LIBRARIES}"
)
swig_add_library(python_netchem
        LANGUAGE python
        OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/netchem
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/swig/python_netchem.i
        )
set_target_properties(
        python_netchem
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/netchem
        LINK_LIBRARIES "${NETCHEM_PYTHON_LINK_LIBRARIES}"
)

set(PIP_UNINSTALL_NETCHEM_ARGS "uninstall" "netchem" "-y")
set(PYTHON_NETCHEM_INSTALL_ARGS "setup.py" "install")
add_custom_target(PythonNetChemInstall
        COMMAND pip ${PIP_UNINSTALL_NETCHEM_ARGS}
        COMMAND ${Python3_EXECUTABLE} ${PYTHON_NETCHEM_INSTALL_ARGS}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/netchem/python
        )
