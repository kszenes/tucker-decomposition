set(This tests)
set(TEST_UTILS decompress.cpp host_contractions.cpp)
add_executable(${This} ${TEST_UTILS} compression_COO_test.cpp first_contraction_test.cpp)
target_link_libraries(${This} PUBLIC gtest_main tucker-lib $<LINK_ONLY:MKL::MKL>)
add_test(
    NAME ${This}
    COMMAND ${This}
)