cmake_minimum_required(VERSION 3.21)
project(lucebox_vulkan_cpu_tests LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Offline, read-only dependency source. Use the same revision as VulkanGGML.cmake.
set(LLAMA_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source, c061df19838ff60970faf54fd7e414953590125d")
if(NOT EXISTS "${LLAMA_SOURCE_DIR}/ggml/include/ggml.h")
    message(FATAL_ERROR "Set LLAMA_SOURCE_DIR to the pinned dependency source")
endif()
foreach(feature GGML_CPU GGML_VULKAN GGML_CUDA GGML_HIP GGML_METAL GGML_NATIVE
                LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_TOOLS LLAMA_BUILD_SERVER LLAMA_CURL)
    set(${feature} OFF CACHE BOOL "Disabled in CPU policy/ownership tests" FORCE)
endforeach()
add_subdirectory(${LLAMA_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/dependency EXCLUDE_FROM_ALL)
find_package(nlohmann_json CONFIG QUIET)
if(NOT TARGET nlohmann_json::nlohmann_json)
    # Reuse the pinned dependency's vendor header without downloading anything.
    add_library(vulkan_test_json INTERFACE)
    target_include_directories(vulkan_test_json INTERFACE ${LLAMA_SOURCE_DIR}/vendor)
    add_library(nlohmann_json::nlohmann_json ALIAS vulkan_test_json)
endif()
enable_testing()
include(../cpu_tests.cmake)
add_executable(test-greedy-scalar ../test_greedy.cpp)
target_compile_options(test-greedy-scalar PRIVATE -U__SSE2__)
add_test(NAME vulkan_greedy_scalar COMMAND test-greedy-scalar)
set_tests_properties(vulkan_greedy_scalar PROPERTIES PASS_REGULAR_EXPRESSION "fused-scalar")
# Fault-inject only at the device boundary. Link both real constructors against
# ggml-base; never link ggml-vulkan or initialize a real GPU in this suite.
add_executable(test-vulkan-ownership ../test_ownership.cpp ../qwen2_backend.cpp ../lfm2_backend.cpp
    ../../src/common/sampler.cpp ../../src/common/daemon_loop.cpp)
target_include_directories(test-vulkan-ownership PRIVATE .. ../../src)
target_link_libraries(test-vulkan-ownership PRIVATE ggml-base nlohmann_json::nlohmann_json)
target_compile_options(test-vulkan-ownership PRIVATE -ffunction-sections -fdata-sections)
target_link_options(test-vulkan-ownership PRIVATE -Wl,--gc-sections
    -Wl,--wrap=ggml_backend_vk_init
    -Wl,--wrap=ggml_backend_name
    -Wl,--wrap=ggml_backend_free
    -Wl,--wrap=ggml_backend_synchronize
    -Wl,--wrap=ggml_backend_alloc_ctx_tensors
    -Wl,--wrap=ggml_backend_buffer_free
    -Wl,--wrap=ggml_backend_buffer_set_usage
    -Wl,--wrap=ggml_backend_buffer_get_size
    -Wl,--wrap=ggml_backend_get_default_buffer_type
    -Wl,--wrap=ggml_gallocr_new
    -Wl,--wrap=ggml_gallocr_free
    -Wl,--wrap=ggml_backend_tensor_set
)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
add_test(NAME vulkan_constructor_ownership COMMAND test-vulkan-ownership ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
set_tests_properties(vulkan_constructor_ownership PROPERTIES TIMEOUT 30)
