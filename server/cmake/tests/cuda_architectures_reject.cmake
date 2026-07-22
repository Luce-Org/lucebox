cmake_minimum_required(VERSION 3.21)
include("${CMAKE_CURRENT_LIST_DIR}/../DflashCudaArchitectures.cmake")

dflash_parse_cuda_architectures(actual_min actual_numeric "${ARCH_LIST}")
