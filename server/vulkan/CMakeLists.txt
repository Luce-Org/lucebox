cmake_minimum_required(VERSION 3.21)
project(lucebox_vulkan LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(../cmake/VulkanGGML.cmake)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)
set(S ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(J ${CMAKE_CURRENT_SOURCE_DIR}/../deps/llama.cpp/common)
add_library(lucebox_vulkan_core STATIC
 qwen2_backend.cpp lfm2_backend.cpp
 ${S}/engine/luce_engine.cpp ${S}/common/daemon_loop.cpp ${S}/common/sampler.cpp
 ${S}/common/concurrency/paged_kv_offload.cpp
 ${S}/common/concurrency/paged_kv_pool.cpp ${S}/common/concurrency/seq_slot_manager.cpp
 ${S}/common/pflash_drafter_ipc.cpp ${S}/common/backend_ipc.cpp
 ${S}/server/http_server.cpp ${S}/server/image_input.cpp ${S}/server/scheduler.cpp ${S}/server/tokenizer.cpp
 ${S}/server/chat_template.cpp ${S}/server/tool_parser.cpp ${S}/server/tool_hint.cpp
 ${S}/server/tool_memory.cpp ${S}/server/response_error.cpp
 ${S}/server/sse_emitter.cpp ${S}/server/prefix_cache.cpp ${S}/server/pin_friendly_prompt.cpp
 ${S}/server/disk_prefix_cache.cpp ${S}/server/freeze_history.cpp
 ${S}/server/prompt_normalize.cpp ${S}/server/model_card.cpp
 ${J}/jinja/lexer.cpp ${J}/jinja/parser.cpp ${J}/jinja/runtime.cpp
 ${J}/jinja/value.cpp ${J}/jinja/string.cpp ${J}/jinja/caps.cpp ${J}/unicode.cpp)
target_include_directories(lucebox_vulkan_core PUBLIC ${S} ${S}/common ${S}/server ${J} ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lucebox_vulkan_core PUBLIC ggml ggml-vulkan ggml-base nlohmann_json::nlohmann_json Threads::Threads)
add_executable(lucebox-vulkan main.cpp)
target_link_libraries(lucebox-vulkan PRIVATE lucebox_vulkan_core)
enable_testing()
add_executable(test-greedy test_greedy.cpp)
add_test(NAME vulkan_greedy COMMAND test-greedy)
add_executable(test-vulkan-policy test_policy.cpp)
target_link_libraries(test-vulkan-policy PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME vulkan_policy COMMAND test-vulkan-policy)
add_executable(test-vulkan-prefill test_prefill.cpp)
add_test(NAME vulkan_prefill COMMAND test-vulkan-prefill)
# Explicitly opt in on a monitored Vulkan host; CPU-only CTest stays GPU-free.
add_executable(test-vulkan-kquants test_kquants.cpp)
target_link_libraries(test-vulkan-kquants PRIVATE ggml ggml-vulkan ggml-base)
option(LUCE_VULKAN_GPU_TESTS "Enable hardware-dependent K-quant tests" OFF)
if(LUCE_VULKAN_GPU_TESTS)
    add_test(NAME vulkan_kquants COMMAND test-vulkan-kquants)
endif()
