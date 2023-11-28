cmake_minimum_required(VERSION 3.21)
project(netsci LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

#Absolute path to conda environment
set(CONDA_DIR "" CACHE PATH "Path to conda environment")

set(CUDA_ARCHITECTURE "52" CACHE STRING "CUDA architecture to compile for. Set this equal to a valid CUDA architecture less than or equal to 10X the compute capability of your GPU.")

#CMake build type defaults to Release
set(CMAKE_BUILD_TYPE Release)

#Download google test zip file
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

#Make sure Boost is installed
find_package(Boost REQUIRED COMPONENTS filesystem system)
#Print boost include directories
message(${Boost_INCLUDE_DIRS})
#Make sure CUDA is installed
find_package(CUDA REQUIRED)

#Find Python and NumPy include directories
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

#Make sure SWIG is installed
find_package(SWIG REQUIRED)

#Path to "UseSWIG.cmake" file
INCLUDE(${SWIG_USE_FILE})

#Make sure ZLIB is installed
FIND_PACKAGE(ZLIB REQUIRED)

#Global include directories used by all build targets
set(
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
include_directories("${INCLUDE_DIRS}")

#Libraries cnpy is linked against
set(
        CNPY_LINK_LIBRARIES
        ${ZLIB_LIBRARIES}
)

#Libraries cuarray is linked against
set(
        CUARRAY_LINK_LIBRARIES
        "${CNPY_LINK_LIBRARIES}"
        "${Boost_LIBRARIES}"
        "${CUDA_LIBRARIES}"
        cnpy
)
#Libraries the python cuarray swig module is linked against
set(
        CUARRAY_PYTHON_LINK_LIBRARIES
        "${CUARRAY_LINK_LIBRARIES}"
        cuarray
        )
#Libraries the tcl cuarray swig module is linked against
set(
        CUARRAY_TCL_LINK_LIBRARIES
        "${CUARRAY_LINK_LIBRARIES}"
        cuarray
)
#Libraries the cuarray tests are linked against
set (
        CUARRAY_TEST_LINK_LIBRARIES
        "${CUARRAY_LINK_LIBRARIES}"
        cuarray
        gtest
)

#Libraries netcalc is linked against
set(
        NETCALC_LINK_LIBRARIES
        "${CUARRAY_LINK_LIBRARIES}"
        "${Boost_LIBRARIES}"
        cuarray
)
#Libraries the python netcalc swig module is linked against
set(
        NETCALC_PYTHON_LINK_LIBRARIES
        "${NETCALC_LINK_LIBRARIES}"
        netcalc
)
#Libraries the netcalc tests are linked against
set (
        NETCALC_TEST_LINK_LIBRARIES
        "${NETCALC_LINK_LIBRARIES}"
        netcalc
        gtest
)

#Libraries netchem is linked against
set(
        NETCHEM_LINK_LIBRARIES
        "${NETCALC_LINK_LIBRARIES}"
        "${Boost_LIBRARIES}"
        cuarray
)

#Libraries the python netchem swig module is linked against
set(
        NETCHEM_PYTHON_LINK_LIBRARIES
        "${NETCHEM_LINK_LIBRARIES}"
        netchem
)

#Libraries the netchem tests are linked against
set (
        NETCHEM_TEST_LINK_LIBRARIES
        "${NETCHEM_LINK_LIBRARIES}"
        netchem
        gtest
)

#Add cnpy src directory to the build
add_subdirectory(${CMAKE_SOURCE_DIR}/cnpy)
#Add cuarray src directory to the build
add_subdirectory(${CMAKE_SOURCE_DIR}/cuarray)
#Add netcalc src directory to the build
add_subdirectory(${CMAKE_SOURCE_DIR}/netcalc)
#Add netchem src directory to the build
add_subdirectory(${CMAKE_SOURCE_DIR}/netchem)

#Write all unit tests to the build directory
enable_testing()
#add tests directory to the build
add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
add_custom_target(
        python
	DEPENDS PythonCuArrayInstall PythonNetCalcInstall PythonNetChemInstall
)
