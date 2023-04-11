
set(CUARRAY_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#CuArray<Float> tests
add_executable(
        FloatCuArray_test
        ${CUARRAY_TEST_DIR}/FloatCuArray_test.cpp
)
add_dependencies(FloatCuArray_test cuarray)
set_target_properties(
        FloatCuArray_test
        PROPERTIES
        LINK_LIBRARIES "${CUARRAY_TEST_LINK_LIBRARIES}"
)
add_test(
        NAME FloatCuArrayNoData_m
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_m*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)
add_test(
        NAME FloatCuArrayNoData_n
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_n*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayNoData_size
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_size*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayNoData_bytes
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_bytes*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME euArrayFloatNoData_allocatedHost
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_allocatedHost*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayNoData_allocatedDevice
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_allocatedDevice*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayNoData_host
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_host*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayNoData_toDeviceNegative
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_toDeviceNegative*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayNoData_toDevicePositive
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_toDevicePositive*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayNoData_toHostNegative
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_toHostNegative*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayNoData_toHostPositive
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayNoData_toHostPositive*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)


add_test(
        NAME FloatCuArrayWithData_m
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayWithData_m*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayWithData_n
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayWithData_n*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayWithData_size
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayWithData_size*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayWithData_bytes
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayWithData_bytes*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayWithData_allocatedHost
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayWithData_allocatedHost*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayWithData_allocatedDevice
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayWithData_allocatedDevice*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayWithData_host
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayWithData_host*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayWithData_at
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayWithData_at*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME FloatCuArrayWithData_allocateDevice
        COMMAND FloatCuArray_test --gtest_filter=*FloatCuArrayWithData_allocateDevice*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)
