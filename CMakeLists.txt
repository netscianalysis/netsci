CMAKE_MINIMUM_REQUIRED(VERSION 3.21)
PROJECT(netsci LANGUAGES CXX CUDA)
enable_testing()

SET(CMAKE_CXX_STANDARD 23)
SET(CMAKE_BUILD_TYPE Release)

include_directories(external)

#Absolute path to conda environment
SET(CONDA_DIR "" CACHE PATH "Path to conda environment")

SET(CUDA_ARCHITECTURE "89" CACHE STRING "CUDA architecture to compile for. Set this equal to a valid CUDA architecture less than or equal to 10X the compute capability of your GPU.")

SET(BUILD_NETSCI ON CACHE BOOL "Build all NetSci toolkit libraries.")

SET(BUILD_PYTHON_BINDINGS ON CACHE BOOL "Build python bindings for cuarray, netcalc, and netchem")

SET(BUILD_SAMPLES ON CACHE BOOL "Compiles examples for CuArray, NetCalc, and NetChem")

#CMake build type defaults to Release
SET(CMAKE_BUILD_TYPE Release)

IF (${BUILD_NETSCI})

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
            ${CMAKE_SOURCE_DIR}/netcalc/src
            ${CMAKE_SOURCE_DIR}/netchem/src
            ${CMAKE_SOURCE_DIR}/netchem/src/json/include
            ${CMAKE_SOURCE_DIR}/netchem/src
            ${CMAKE_SOURCE_DIR}/cuarray/src
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
    #Libraries the cuarray tests are linked against
    SET(
            CUARRAY_TEST_LINK_LIBRARIES
            "${CUARRAY_LINK_LIBRARIES}"
            cuarray
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
