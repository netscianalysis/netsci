
set(NETCHEM_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(COPY ${TEST_DATA_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
message(${TEST_DATA_DIR})
add_executable(
        NetworkStride1_test
        ${NETCHEM_TEST_DIR}/NetworkStride1_test.cpp)
add_dependencies(NetworkStride1_test netchem)
set_target_properties(
        NetworkStride1_test
        PROPERTIES
        LINK_LIBRARIES "${NETCHEM_TEST_LINK_LIBRARIES}"
)
add_executable(
        NetworkStride2_test
        ${NETCHEM_TEST_DIR}/NetworkStride2_test.cpp)
add_dependencies(NetworkStride2_test netchem)
set_target_properties(
        NetworkStride2_test
        PROPERTIES
        LINK_LIBRARIES "${NETCHEM_TEST_LINK_LIBRARIES}"
)
add_executable(
        NetworkStride3_test
        ${NETCHEM_TEST_DIR}/NetworkStride3_test.cpp)
add_dependencies(NetworkStride3_test netchem)
set_target_properties(
       NetworkStride3_test
       PROPERTIES
         LINK_LIBRARIES "${NETCHEM_TEST_LINK_LIBRARIES}"
)

add_executable(
        NetworkStrideCoordinateEquality_test
        ${NETCHEM_TEST_DIR}/NetworkStrideCoordinateEquality_test.cpp)
add_dependencies(NetworkStrideCoordinateEquality_test netchem)
set_target_properties(
        NetworkStrideCoordinateEquality_test
        PROPERTIES
        LINK_LIBRARIES "${NETCHEM_TEST_LINK_LIBRARIES}"
)

add_test(
        NAME NetworkStride1_numFrames
        COMMAND NetworkStride1_test --gtest_filter=*NetworkStride1_numFrames*
        WORKING_DIRECTORY ${NETCHEM_TEST_DIR}
)
add_test(
        NAME NetworkStride2_numFrames
        COMMAND NetworkStride2_test --gtest_filter=*NetworkStride2_numFrames*
        WORKING_DIRECTORY ${NETCHEM_TEST_DIR}
)
add_test(
        NAME NetworkStride3_numFrames
        COMMAND NetworkStride3_test --gtest_filter=*NetworkStride3_numFrames*
        WORKING_DIRECTORY ${NETCHEM_TEST_DIR}
)
add_test(
        NAME NetworkStrideCoordinateEqualityStride1Stride2
        COMMAND NetworkStrideCoordinateEquality_test --gtest_filter=*NetworkStrideCoordinateEqualityStride1Stride2*
        WORKING_DIRECTORY ${NETCHEM_TEST_DIR}
)
add_test(
        NAME NetworkStrideCoordinateEqualityStride1Stride3
        COMMAND NetworkStrideCoordinateEquality_test --gtest_filter=*NetworkStrideCoordinateEqualityStride1Stride3*
        WORKING_DIRECTORY ${NETCHEM_TEST_DIR}

)
