
set(CUARRAY_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#CuArray<Float> tests
add_executable(
        CuArrayFloat_test
        ${CUARRAY_TEST_DIR}/CuArrayFloat_test.cpp
)
add_dependencies(CuArrayFloat_test cuarray)
set_target_properties(
        CuArrayFloat_test
        PROPERTIES
        LINK_LIBRARIES "${CUARRAY_TEST_LINK_LIBRARIES}"
)
add_test(
        NAME CuArrayFloatNoData_m
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_m*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)
add_test(
        NAME CuArrayFloatNoData_n
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_n*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatNoData_size
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_size*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatNoData_bytes
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_bytes*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME euArrayFloatNoData_allocatedHost
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_allocatedHost*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatNoData_allocatedDevice
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_allocatedDevice*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatNoData_host
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_host*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatNoData_toDeviceNegative
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_toDeviceNegative*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatNoData_toDevicePositive
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_toDevicePositive*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatNoData_toHostNegative
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_toHostNegative*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatNoData_toHostPositive
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatNoData_toHostPositive*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)


add_test(
        NAME CuArrayFloatWithData_m
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatWithData_m*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatWithData_n
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatWithData_n*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatWithData_size
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatWithData_size*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatWithData_bytes
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatWithData_bytes*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatWithData_allocatedHost
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatWithData_allocatedHost*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatWithData_allocatedDevice
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatWithData_allocatedDevice*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatWithData_host
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatWithData_host*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatWithData_at
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatWithData_at*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)

add_test(
        NAME CuArrayFloatWithData_allocateDevice
        COMMAND CuArrayFloat_test --gtest_filter=*CuArrayFloatWithData_allocateDevice*
        WORKING_DIRECTORY ${CUARRAY_TEST_DIR}
)
