CMAKE_MINIMUM_REQUIRED(VERSION 3.21)
PROJECT(netsci LANGUAGES CXX CUDA)
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release)

#Absolute path to conda environment
SET(CONDA_DIR "" CACHE PATH "Path to conda environment")

SET(VMD_LIB_DIR "" CACHE PATH "Path to VMD lib directory")

SET(CREATE_VMD_TCL_PACKAGE OFF CACHE BOOL "If ON, the CuArray TCL library is added to VMD's tcl package list.")

SET(CUDA_ARCHITECTURE "52" CACHE STRING "CUDA architecture to compile for. Set this equal to a valid CUDA architecture less than or equal to 10X the compute capability of your GPU.")

SET(BUILD_NETSCI ON CACHE BOOL "Build all NetSci toolkit libraries.")

SET(BUILD_PYTHON_BINDINGS ON CACHE BOOL "Build python bindings for cuarray, netcalc, and netchem")
SET(BUILD_TCL_BINDINGS OFF CACHE BOOL "Build tcl bindings for cuarray")

SET(BUILD_SAMPLES ON CACHE BOOL "Compiles examples for CuArray, NetCalc, and NetChem")

SET(GENERATE_DOCS OFF CACHE BOOL "Generate documentation for CuArray, NetCalc, and NetChem")

#CMake build type defaults to Release
SET(CMAKE_BUILD_TYPE Release)

IF (${BUILD_NETSCI})
    #Download google test zip file
    INCLUDE(FetchContent)
    FETCHCONTENT_DECLARE(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
            )
    FETCHCONTENT_MAKEAVAILABLE(googletest)

#Make sure Boost is installed
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem system)
#Print boost include directories
MESSAGE(${Boost_INCLUDE_DIRS})
#Make sure CUDA is installed
FIND_PACKAGE(CUDA REQUIRED)

#Find Python and NumPy include directories
FIND_PACKAGE(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

#Make sure SWIG is installed
FIND_PACKAGE(SWIG REQUIRED)

#Path to "UseSWIG.cmake" file
INCLUDE(${SWIG_USE_FILE})

#Make sure ZLIB is installed
FIND_PACKAGE(ZLIB REQUIRED)

#Global include directories used by all build targets
SET(
        INCLUDE_DIRS
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
        ${CONDA_DIR}/include
        ${CMAKE_SOURCE_DIR}/cnpy/include
        ${CMAKE_SOURCE_DIR}/netcalc/include
        ${CMAKE_SOURCE_DIR}/netchem/include
        ${CMAKE_SOURCE_DIR}/netchem/include/json/include
        ${CMAKE_SOURCE_DIR}/netchem/include
        ${CMAKE_SOURCE_DIR}/cuarray/include
        "${CUDA_INCLUDE_DIRS}"
        )
INCLUDE_DIRECTORIES("${INCLUDE_DIRS}")

#Libraries cnpy is linked against
SET(
        CNPY_LINK_LIBRARIES
        ${ZLIB_LIBRARIES}
        )

#Libraries cuarray is linked against
SET(
        CUARRAY_LINK_LIBRARIES
        "${CNPY_LINK_LIBRARIES}"
        "${Boost_LIBRARIES}"
        "${CUDA_LIBRARIES}"
        cnpy
        )
#Libraries the python cuarray swig module is linked against
SET(
        CUARRAY_PYTHON_LINK_LIBRARIES
        "${CUARRAY_LINK_LIBRARIES}"
        cuarray
        )
#Libraries the tcl cuarray swig module is linked against
SET(
        CUARRAY_TCL_LINK_LIBRARIES
        "${CUARRAY_LINK_LIBRARIES}"
        cuarray
        )
#Libraries the cuarray tests are linked against
SET(
        CUARRAY_TEST_LINK_LIBRARIES
        "${CUARRAY_LINK_LIBRARIES}"
        cuarray
        gtest
        )

#Libraries netcalc is linked against
SET(
        NETCALC_LINK_LIBRARIES
        "${CUARRAY_LINK_LIBRARIES}"
        "${Boost_LIBRARIES}"
        cuarray
        )
#Libraries the python netcalc swig module is linked against
SET(
        NETCALC_PYTHON_LINK_LIBRARIES
        "${NETCALC_LINK_LIBRARIES}"
        netcalc
        )
#Libraries the netcalc tests are linked against
SET(
        NETCALC_TEST_LINK_LIBRARIES
        "${NETCALC_LINK_LIBRARIES}"
        netcalc
        gtest
        )

SET(
        NETCALC_SCRATCH_LINK_LIBRARIES
        "${NETCALC_LINK_LIBRARIES}"
        netcalc
        )

#Libraries netchem is linked against
SET(
        NETCHEM_LINK_LIBRARIES
        "${NETCALC_LINK_LIBRARIES}"
        "${Boost_LIBRARIES}"
        cuarray
        )

#Libraries the python netchem swig module is linked against
SET(
        NETCHEM_PYTHON_LINK_LIBRARIES
        "${NETCHEM_LINK_LIBRARIES}"
        netchem
        )

#Libraries the netchem tests are linked against
SET(
        NETCHEM_TEST_LINK_LIBRARIES
        "${NETCHEM_LINK_LIBRARIES}"
        netchem
        gtest
        )

SET(
        NETCHEM_SCRATCH_LINK_LIBRARIES
        "${NETCHEM_LINK_LIBRARIES}"
        netchem
        )

#Add cnpy src directory to the build
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/cnpy)
#Add cuarray src directory to the build
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/cuarray)
#Add netcalc src directory to the build
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/netcalc)
#Add netchem src directory to the build
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/netchem)

    #Write all unit tests to the build directory
    ENABLE_TESTING()
    #add tests directory to the build
    ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/tests)
    IF (${BUILD_PYTHON_BINDINGS})
        ADD_CUSTOM_TARGET(
                python
                DEPENDS PythonCuArrayInstall PythonNetCalcInstall PythonNetChemInstall
                )
    ENDIF (${BUILD_PYTHON_BINDINGS})
ENDIF (${BUILD_NETSCI})

IF (${GENERATE_DOCS})
    #Make sure sure Doxygen is installed for building documentation
    FIND_PACKAGE(Doxygen REQUIRED)

    SET(DOXYGEN_PROJECT_NAME NetSci)
    SET(DOXYGEN_EXAMPLE_PATH
        ${CMAKE_SOURCE_DIR}/cuarray/samples/cpp/src
        ${CMAKE_SOURCE_DIR}/cuarray/samples/python
        ${CMAKE_SOURCE_DIR}/netchem/samples/python
        ${CMAKE_SOURCE_DIR}/netcalc/samples/python
        )
    SET(DOXYGEN_ALIASES
        [[PythonExample{1}=<b>Python Example</b><br>\include \1]]
        [[CppExample{1}=<b>C++ Example</b><br>\include \1]]
        [[type{1}=<!-- \1 -->]]
        [[function{1}=<!-- \1 -->]]
        [[const=<!--  -->]]
        [[AutoDocIgnore=<!--  -->]]
        )
    SET(DOXYGEN_EXTRACT_PRIVATE YES)
    SET(DOXYGEN_FULL_PATH_NAMES NO)
    SET(DOXYGEN_FILE_PATTERNS *.cu)
    SET(DOXYGEN_EXTENSION_MAPPING cu=c++)
    SET(DOXYGEN_ENABLE_PREPROCESSING NO)
    SET(DOXYGEN_STRIP_CODE_COMMENTS NO)
    SET(DOXYGEN_COMPACT_LATEX YES)
    SET(DOXYGEN_EXTRACT_PRIVATE YES)
    SET(DOXYGEN_GENERATE_DOCBOOK YES)
    SET(DOXYGEN_GENERATE_LATEX YES)
    SET(DOXYGEN_SHOW_FILES NO)
    SET(DOXYGEN_USE_PDFLATEX YES)
    SET(DOXYGEN_FULL_PATH_NAMES NO)
    SET(DOXYGEN_GENERATE_XML TRUE)
    SET(DOXYGEN_USE_MATHJAX YES)
    SET(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/docs)
    LIST(APPEND NETSCI_DOCUMENTATION_INPUT_FILES
         ${CMAKE_SOURCE_DIR}/doxygen/index.h
         ${CMAKE_SOURCE_DIR}/cuarray/include/cuarray.h
         ${CMAKE_SOURCE_DIR}/netcalc/include/mutual_information.h
         ${CMAKE_SOURCE_DIR}/netcalc/include/generalized_correlation.h
         ${CMAKE_SOURCE_DIR}/netcalc/include/hedetniemi.h
         ${CMAKE_SOURCE_DIR}/netchem/include/network.h
         ${CMAKE_SOURCE_DIR}/netchem/include/node.h
         ${CMAKE_SOURCE_DIR}/netchem/include/atom.h
         ${CMAKE_SOURCE_DIR}/netchem/include/atoms.h
         )
    DOXYGEN_ADD_DOCS(
            NetSci_docs
            ${NETSCI_DOCUMENTATION_INPUT_FILES}
            )
    SET(DOCUMENTATION_ARGS "Doxyfile.NetSci_docs")
    ADD_CUSTOM_TARGET(docs
                      COMMAND ${DOXYGEN_EXECUTABLE} ${DOCUMENTATION_ARGS}
                      )
ENDIF (${GENERATE_DOCS})

