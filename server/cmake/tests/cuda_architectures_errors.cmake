cmake_minimum_required(VERSION 3.21)

function(assert_rejected label arch_list expected_error)
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DARCH_LIST=${arch_list}"
            -P "${CMAKE_CURRENT_LIST_DIR}/cuda_architectures_reject.cmake"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if(result EQUAL 0)
        message(FATAL_ERROR "${label}: malformed list '${arch_list}' was accepted")
    endif()
    set(output "${stdout}\n${stderr}")
    if(NOT output MATCHES "${expected_error}")
        message(FATAL_ERROR
            "${label}: expected error /${expected_error}/, got:\n${output}")
    endif()
endfunction()

assert_rejected("empty list" "" "must not be empty")
assert_rejected("leading empty entry" ";86" "contains an empty entry")
assert_rejected("middle empty entry" "86;;75" "contains an empty entry")
assert_rejected("trailing empty entry" "86;" "contains an empty entry")
assert_rejected("sm prefix" "sm_86" "Unsupported CUDA architecture")
assert_rejected("unknown suffix" "86-ptx" "Unsupported CUDA architecture")
assert_rejected("unknown architecture suffix" "120b" "Unsupported CUDA architecture")
assert_rejected("multiple architecture suffixes" "120af" "Unsupported CUDA architecture")
assert_rejected("suffixes out of order" "120-real-a" "Unsupported CUDA architecture")
assert_rejected("extra suffix" "90-real-extra" "Unsupported CUDA architecture")
assert_rejected("special native" "native" "Unsupported CUDA architecture")
assert_rejected("special all" "all" "Unsupported CUDA architecture")
assert_rejected("zero" "0" "Unsupported CUDA architecture")
