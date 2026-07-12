add_executable(test-rocmfpx ../rocmfpx/test_rocmfpx.c)
target_include_directories(test-rocmfpx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../rocmfpx)
target_link_libraries(test-rocmfpx PRIVATE ggml-base)

if (NOT MSVC)
    target_link_libraries(test-rocmfpx PRIVATE m)
endif()

add_test(NAME test-rocmfpx COMMAND test-rocmfpx)
