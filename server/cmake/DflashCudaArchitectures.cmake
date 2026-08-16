include_guard(GLOBAL)

# Parse a CMAKE_CUDA_ARCHITECTURES-style list into numeric SM values and find
# its true minimum. Numeric architectures may use CUDA's architecture-specific
# "a"/"f" suffixes and CMake's code-generation "-real"/"-virtual" suffixes.
# Special values such as "all" and "native" cannot produce a deterministic
# compile-time minimum and are intentionally rejected.
function(dflash_parse_cuda_architectures out_min_sm out_numeric_arches arch_list)
    if(NOT ARGC EQUAL 3)
        message(FATAL_ERROR
            "dflash_parse_cuda_architectures expects an output minimum, an "
            "output numeric list, and one quoted CUDA architecture list")
    endif()
    if("${arch_list}" STREQUAL "")
        message(FATAL_ERROR "CUDA architecture list must not be empty")
    endif()

    set(_dflash_numeric_arches "")
    set(_dflash_min_sm "")
    foreach(_dflash_arch IN LISTS arch_list)
        if("${_dflash_arch}" STREQUAL "")
            message(FATAL_ERROR
                "CUDA architecture list contains an empty entry: '${arch_list}'")
        endif()
        if(NOT "${_dflash_arch}" MATCHES "^([1-9][0-9]*)(a|f)?(-(real|virtual))?$")
            message(FATAL_ERROR
                "Unsupported CUDA architecture '${_dflash_arch}' in '${arch_list}'. "
                "Use a positive numeric SM with an optional a/f architecture suffix "
                "and an optional -real/-virtual code-generation suffix.")
        endif()

        set(_dflash_arch_sm "${CMAKE_MATCH_1}")
        list(APPEND _dflash_numeric_arches "${_dflash_arch_sm}")
        if("${_dflash_min_sm}" STREQUAL "")
            set(_dflash_min_sm "${_dflash_arch_sm}")
        elseif(_dflash_arch_sm LESS _dflash_min_sm)
            set(_dflash_min_sm "${_dflash_arch_sm}")
        endif()
    endforeach()

    if("${_dflash_min_sm}" STREQUAL "")
        message(FATAL_ERROR "CUDA architecture list must contain at least one entry")
    endif()

    set(${out_min_sm} "${_dflash_min_sm}" PARENT_SCOPE)
    set(${out_numeric_arches} "${_dflash_numeric_arches}" PARENT_SCOPE)
endfunction()
