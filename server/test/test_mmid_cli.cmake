if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED TEST_OUTPUT_DIR OR NOT DEFINED TEST_CASE)
    message(FATAL_ERROR "TEST_EXECUTABLE, TEST_OUTPUT_DIR and TEST_CASE are required")
endif()

if(TEST_CASE STREQUAL "parent_benchmark")
    foreach(mode IN ITEMS full mmid_only)
        set(args)
        if(mode STREQUAL "mmid_only")
            set(args --mmid-only)
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env
                DFLASH_MMID_BENCH_ITERS=1 DFLASH_MMID_BENCH_ROWS=64
                DFLASH_MMID_BENCH_TOP_K=4 --unset=DFLASH_MMID_TEST_WIDTH
                "${TEST_EXECUTABLE}" ${args}
            RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr
            TIMEOUT 60)
        if(NOT status STREQUAL "2" OR
           NOT stderr MATCHES "DFLASH_MMID_BENCH_ITERS is supported only with --child")
            message(FATAL_ERROR "${mode} did not reject benchmark mode (${status}):\n${stdout}\n${stderr}")
        endif()
    endforeach()
elseif(TEST_CASE STREQUAL "no_gpu")
    # Hide devices only in child processes; do not alter the runner's GPU state.
    set(no_gpu_env CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1)
    file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")
    foreach(mode IN ITEMS legacy grouped masked-fused)
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env ${no_gpu_env}
                "${TEST_EXECUTABLE}" --child "${mode}" "${TEST_OUTPUT_DIR}/${mode}.bin"
            RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr
            TIMEOUT 60)
        if(NOT status STREQUAL "77")
            message(FATAL_ERROR "${mode} did not skip with no GPU (${status}):\n${stdout}\n${stderr}")
        endif()
    endforeach()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env ${no_gpu_env}
            "${CMAKE_COMMAND}" "-DTEST_EXECUTABLE=${TEST_EXECUTABLE}"
                "-DTEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}/config"
                -P "${CMAKE_CURRENT_LIST_DIR}/test_mmid_benchmark_config.cmake"
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr
        TIMEOUT 60)
    if(NOT status STREQUAL "0" OR
       NOT stdout MATCHES "mmid-benchmark-config.*SKIP: no supported GPU" OR
       stdout MATCHES "configuration: PASS")
        message(FATAL_ERROR "configuration wrapper did not propagate GPU skip (${status}):\n${stdout}\n${stderr}")
    endif()
else()
    message(FATAL_ERROR "Unknown TEST_CASE: ${TEST_CASE}")
endif()
message(STATUS "MMID CLI ${TEST_CASE}: PASS")
