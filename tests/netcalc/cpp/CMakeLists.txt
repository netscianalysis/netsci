
set(NETCALC_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(COPY ${TEST_DATA_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
#Mutual Information tests
add_executable(test_mutual_information MutualInformation_test.cpp)
target_link_libraries(test_mutual_information PRIVATE netcalc cuarray cnpy)
add_test(NAME test_mutual_information COMMAND test_mutual_information)

add_executable(test_generalized_correlation GeneralizedCorrelation_test.cpp)
target_link_libraries(test_generalized_correlation PRIVATE netcalc cuarray cnpy)
add_test(NAME test_generalized_correlation COMMAND test_generalized_correlation)


