cmake_minimum_required(VERSION 3.21)
project(dflash_cuda_architecture_tests LANGUAGES NONE)

enable_testing()

foreach(_case IN ITEMS permutations suffixes duplicates)
    add_test(
        NAME cuda_arch_${_case}
        COMMAND "${CMAKE_COMMAND}"
            "-DCASE=${_case}"
            -P "${CMAKE_CURRENT_LIST_DIR}/cuda_architectures_case.cmake")
endforeach()

add_test(
    NAME cuda_arch_errors
    COMMAND "${CMAKE_COMMAND}"
        -P "${CMAKE_CURRENT_LIST_DIR}/cuda_architectures_errors.cmake")
