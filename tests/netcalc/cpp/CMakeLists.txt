
set(NETCALC_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(COPY ${TEST_DATA_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
message(${TEST_DATA_DIR})
#Mutual Information tests
add_executable(
        MutualInformation_test
        ${NETCALC_TEST_DIR}/MutualInformation_test.cpp)
add_dependencies(MutualInformation_test netcalc)
set_target_properties(
        MutualInformation_test
        PROPERTIES
        LINK_LIBRARIES "${NETCALC_TEST_LINK_LIBRARIES}"
)

add_test(
        NAME MutualInformation2X1D_1000n4k09covGaussian_GpuCpu
        COMMAND MutualInformation_test --gtest_filter=*MutualInformation2X1D_1000n4k09covGaussian_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME MutualInformation2X1D_2000n4k09covGaussian_GpuCpu
        COMMAND MutualInformation_test --gtest_filter=*MutualInformation2X1D_2000n4k09covGaussian_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME MutualInformation2X2D_1000n4k_GpuCpu
        COMMAND MutualInformation_test --gtest_filter=*MutualInformation2X2D_1000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME MutualInformation2X2D_2000n4k_GpuCpu
        COMMAND MutualInformation_test --gtest_filter=*MutualInformation2X2D_2000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME MutualInformation2X3D_1000n4k_GpuCpu
        COMMAND MutualInformation_test --gtest_filter=*MutualInformation2X3D_1000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME MutualInformation2X3D_2000n4k_GpuCpu
        COMMAND MutualInformation_test --gtest_filter=*MutualInformation2X3D_2000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)


#Generalized Correlation tests
add_executable(
        GeneralizedCorrelation_test
        ${NETCALC_TEST_DIR}/GeneralizedCorrelation_test.cpp
)
add_dependencies(GeneralizedCorrelation_test netcalc)
set_target_properties(
        GeneralizedCorrelation_test
        PROPERTIES
        LINK_LIBRARIES "${NETCALC_TEST_LINK_LIBRARIES}"
)

add_test(
        NAME GeneralizedCorrelation2X1D_1000n4k_GpuCpu
        COMMAND GeneralizedCorrelation_test --gtest_filter=*GeneralizedCorrelation_2X1D_1000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME GeneralizedCorrelation2X1D_2000n4k_GpuCpu
        COMMAND GeneralizedCorrelation_test --gtest_filter=*GeneralizedCorrelation_2X1D_2000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME GeneralizedCorrelation2X2D_1000n4k_GpuCpu
        COMMAND GeneralizedCorrelation_test --gtest_filter=*GeneralizedCorrelation_2X2D_1000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME GeneralizedCorrelation2X2D_2000n4k_GpuCpu
        COMMAND GeneralizedCorrelation_test --gtest_filter=*GeneralizedCorrelation_2X2D_2000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME GeneralizedCorrelation2X3D_1000n4k_GpuCpu
        COMMAND GeneralizedCorrelation_test --gtest_filter=*GeneralizedCorrelation_2X3D_1000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)

add_test(
        NAME GeneralizedCorrelation2X3D_2000n4k_GpuCpu
        COMMAND GeneralizedCorrelation_test --gtest_filter=*GeneralizedCorrelation_2X3D_2000n4k_GpuCpu*
        WORKING_DIRECTORY ${NETCALC_TEST_DIR}
)
