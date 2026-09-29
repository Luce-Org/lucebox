if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED TEST_OUTPUT_DIR)
    message(FATAL_ERROR "TEST_EXECUTABLE and TEST_OUTPUT_DIR are required")
endif()

file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")

# Each process initializes the cached environment switches independently.
# Test the default, the type-mask escape hatch, and the global grouped disable.
foreach(mode IN ITEMS legacy default mask-off)
    set(policy --unset=LUCE_MMID_GROUPED_TYPES LUCE_MMID_GROUPED=1)
    set(expect_grouped 0)
    if(mode STREQUAL "legacy")
        set(policy --unset=LUCE_MMID_GROUPED_TYPES LUCE_MMID_GROUPED=0)
    elseif(mode STREQUAL "mask-off")
        set(policy LUCE_MMID_GROUPED_TYPES=7 LUCE_MMID_GROUPED=1)
    else()
        set(expect_grouped 1)
    endif()
    set(path "${TEST_OUTPUT_DIR}/${mode}.bin")
    file(REMOVE "${path}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env ${policy}
            --unset=LUCE_MMID_GROUPED_DEVICE --unset=GGML_CUDA_DISABLE_FUSION
            --unset=LUCE_MMID_BENCH_ITERS
            LUCE_CUDA_MMVQ_MOE_KERNEL=1
            LUCE_CUDA_MMVQ_MOE_TOKENWISE=0 LUCE_MMID_TELEMETRY=1
            "${TEST_EXECUTABLE}" --q5-cases "${path}" "${expect_grouped}"
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr
        TIMEOUT 240)
    file(WRITE "${TEST_OUTPUT_DIR}/${mode}.log" "${stdout}\n${stderr}")
    if(status STREQUAL "77")
        message(STATUS "q5-mmid-grouped SKIP: requires NVIDIA sm_86")
        return()
    endif()
    if(NOT status STREQUAL "0")
        message(FATAL_ERROR "${mode} failed (${status}):\n${stdout}\n${stderr}")
    endif()
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "Missing output: ${path}")
    endif()
    if(NOT mode STREQUAL "legacy")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${TEST_OUTPUT_DIR}/legacy.bin" "${path}"
            RESULT_VARIABLE parity)
        if(NOT parity STREQUAL "0")
            message(FATAL_ERROR "Q5_0 parity failed: ${mode}")
        endif()
    endif()
endforeach()
message(STATUS "q5-mmid-grouped PASS: default/mask-off/legacy parity and dispatch")
