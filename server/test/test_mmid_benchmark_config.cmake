# Exercise the actual weight-generation, shape and timing paths in fresh
# processes so an inherited benchmark flag cannot weaken a parity check.
if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED TEST_OUTPUT_DIR)
    message(FATAL_ERROR "TEST_EXECUTABLE and TEST_OUTPUT_DIR are required")
endif()
file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")

foreach(mode IN ITEMS unset zero negative invalid positive)
    set(benchmark_env "--unset=DFLASH_MMID_BENCH_ITERS")
    if(mode STREQUAL "zero")
        set(benchmark_env "DFLASH_MMID_BENCH_ITERS=0")
    elseif(mode STREQUAL "negative")
        set(benchmark_env "DFLASH_MMID_BENCH_ITERS=-1")
    elseif(mode STREQUAL "invalid")
        set(benchmark_env "DFLASH_MMID_BENCH_ITERS=invalid")
    elseif(mode STREQUAL "positive")
        set(benchmark_env "DFLASH_MMID_BENCH_ITERS=1")
    endif()
    set(output "${TEST_OUTPUT_DIR}/${mode}.bin")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "${benchmark_env}"
            DFLASH_MMID_BENCH_K=512 DFLASH_MMID_BENCH_ROWS=64
            DFLASH_MMID_BENCH_EXPERTS=16 DFLASH_MMID_BENCH_TOP_K=4
            --unset=DFLASH_MMID_TEST_WIDTH
            "${TEST_EXECUTABLE}" --child masked-fused "${output}"
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr
        TIMEOUT 60)
    if(status STREQUAL "77" AND mode STREQUAL "unset")
        # cmake -P has no portable exit(77). Preserve the child's skip through
        # the marker registered with CTest's SKIP_REGULAR_EXPRESSION instead.
        message(STATUS "[mmid-benchmark-config] SKIP: no supported GPU")
        return()
    endif()
    if(NOT status STREQUAL "0")
        message(FATAL_ERROR "${mode} failed (${status}):\n${stdout}\n${stderr}")
    endif()
    file(SIZE "${output}" output_size)
    file(SHA256 "${output}" output_hash)
    if(mode STREQUAL "positive")
        # The positive control must apply the benchmark shape and run timing.
        if(NOT output_size EQUAL 32768 OR
           NOT stdout MATCHES "iterations=1 average_us=")
            message(FATAL_ERROR "positive benchmark did not run: ${stdout}")
        endif()
    else()
        if(NOT output_size EQUAL 131072 OR stdout MATCHES "average_us=")
            message(FATAL_ERROR "${mode} unexpectedly enabled benchmark mode: ${stdout}")
        endif()
        if(mode STREQUAL "unset")
            file(READ "${output}" output_hex HEX)
            if(NOT output_hex MATCHES "[1-9a-fA-F]")
                message(FATAL_ERROR "correctness fixture produced only zero output")
            endif()
            set(reference_hash "${output_hash}")
        elseif(NOT output_hash STREQUAL reference_hash)
            message(FATAL_ERROR "${mode} changed randomized correctness output")
        endif()
    endif()
endforeach()
message(STATUS "MMID benchmark configuration: PASS (unset, zero, negative, invalid, positive)")
