cmake_minimum_required(VERSION 3.21)
include("${CMAKE_CURRENT_LIST_DIR}/../DflashCudaArchitectures.cmake")

function(assert_arch_parse arch_list expected_min expected_numeric)
    dflash_parse_cuda_architectures(actual_min actual_numeric "${arch_list}")
    if(NOT "${actual_min}" STREQUAL "${expected_min}")
        message(FATAL_ERROR
            "minimum for '${arch_list}': expected '${expected_min}', got '${actual_min}'")
    endif()
    if(NOT "${actual_numeric}" STREQUAL "${expected_numeric}")
        message(FATAL_ERROR
            "numeric list for '${arch_list}': expected '${expected_numeric}', "
            "got '${actual_numeric}'")
    endif()
endfunction()

if(CASE STREQUAL "permutations")
    assert_arch_parse("60;61;62;70;75;86;120" "60" "60;61;62;70;75;86;120")
    assert_arch_parse("120;86;75;70;62;61;60" "60" "120;86;75;70;62;61;60")
    assert_arch_parse("86;60;120;70;61;75;62" "60" "86;60;120;70;61;75;62")
elseif(CASE STREQUAL "suffixes")
    assert_arch_parse("121a-real;120f-virtual;90-virtual;100" "90" "121;120;90;100")
    assert_arch_parse("120a;86-virtual" "86" "120;86")
elseif(CASE STREQUAL "duplicates")
    assert_arch_parse("120a;86;120f-real;86-virtual" "86" "120;86;120;86")
else()
    message(FATAL_ERROR "unknown test CASE '${CASE}'")
endif()
