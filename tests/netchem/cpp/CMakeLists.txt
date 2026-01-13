
set(NETCHEM_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# Copy $NETCHEM_TEST_DIR/data to $CMAKE_CURRENT_BINARY_DIR/data
file(COPY ${NETCHEM_TEST_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


#Netchem tests
add_executable(test_network_stride1 NetworkStride1_test.cpp)
target_link_libraries(test_network_stride1 PRIVATE netchem cuarray cnpy)
add_test(NAME test_network_stride1 COMMAND test_network_stride1 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_network_stride2 NetworkStride2_test.cpp)
target_link_libraries(test_network_stride2 PRIVATE netchem cuarray cnpy)
add_test(NAME test_network_stride2 COMMAND test_network_stride2 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_network_stride3 NetworkStride3_test.cpp)
target_link_libraries(test_network_stride3 PRIVATE netchem cuarray cnpy)
add_test(NAME test_network_stride3 COMMAND test_network_stride3 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_network_stride_coordinate_equality NetworkStrideCoordinateEquality_test.cpp)
target_link_libraries(test_network_stride_coordinate_equality PRIVATE netchem cuarray cnpy)
add_test(NAME test_network_stride_coordinate_equality COMMAND test_network_stride_coordinate_equality WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})