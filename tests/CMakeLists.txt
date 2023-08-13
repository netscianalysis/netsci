
set(TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/tests/data)
message(${TEST_DATA_DIR})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/netcalc/cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuarray/cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/netchem/cpp)
