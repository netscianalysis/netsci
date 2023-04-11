
file(COPY ${CMAKE_SOURCE_DIR}/tests/data DESTINATION ${CMAKE_BINARY_DIR}/tests/netsci)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/netsci/cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuarray/cpp)
