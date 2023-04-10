cmake_minimum_required(VERSION 3.21)
project(netsci LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

#Absolute path to conda environment
set(CONDA_DIR "" CACHE PATH "Path to conda environment")

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

#Make sure CUDA is installed
find_package(CUDA REQUIRED)

#Find Python and NumPy include directories
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

#Make sure SWIG is installed
find_package(SWIG REQUIRED)

#Path to "UseSWIG.cmake" file
include(${SWIG_USE_FILE})

#Global include directories used by all build targets
set(
        INCLUDE_DIRS
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/cnpy/include
        ${CMAKE_SOURCE_DIR}/netsci/include
        ${CMAKE_SOURCE_DIR}/cuarray/include
        "${CUDA_INCLUDE_DIRS}"
)
include_directories("${INCLUDE_DIRS}")

#Libraries cnpy is linked against
set(
        CNPY_LINK_LIBRARIES
        ${CONDA_DIR}/lib/libz.so
)

#Libraries cuarray is linked against
set(
        CUARRAY_LINK_LIBRARIES
        "${CNPY_LINK_LIBRARIES}"
        "${Boost_LIBRARIES}"
        cnpy
)
#Libraries the python cuarray swig module is linked against
set(
        CUARRAY_PYTHON_LINK_LIBRARIES
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

#Libraries netsci is linked against
set(
        NETSCI_LINK_LIBRARIES
        "${CUARRAY_LINK_LIBRARIES}"
        "${Boost_LIBRARIES}"
        cuarray
)
#Libraries the python netsci swig module is linked against
set(
        NETSCI_PYTHON_LINK_LIBRARIES
        "${NETSCI_LINK_LIBRARIES}"
        netsci
)
#Libraries the netsci tests are linked against
set (
        NETSCI_TEST_LINK_LIBRARIES
        "${NETSCI_LINK_LIBRARIES}"
        netsci
        gtest
)

#Add cnpy src directory to the build
add_subdirectory(${CMAKE_SOURCE_DIR}/cnpy)
#Add cuarray src directory to the build
add_subdirectory(${CMAKE_SOURCE_DIR}/cuarray)
#Add netsci src directory to the build
add_subdirectory(${CMAKE_SOURCE_DIR}/netsci)

#Write all unit tests to the build directory
enable_testing()
#add tests directory to the build
add_subdirectory(${CMAKE_SOURCE_DIR}/tests)

#Scratch executable for quick manual testing
add_executable(main ${CMAKE_SOURCE_DIR}/main.cpp)
set_target_properties(
        main
        PROPERTIES
        LINK_LIBRARIES cuarray
)