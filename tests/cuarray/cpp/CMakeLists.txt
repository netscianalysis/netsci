
set(CUARRAY_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})


add_executable(test_cuarray cuarray.test.cpp)
target_link_libraries(test_cuarray PRIVATE cuarray )
add_test(NAME test_cuarray COMMAND test_cuarray)

