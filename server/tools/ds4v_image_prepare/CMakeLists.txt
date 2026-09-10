cmake_minimum_required(VERSION 3.21)
project(ds4v_image_prepare LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
foreach(backend CPU CUDA HIP METAL VULKAN SYCL OPENCL CANN MUSA RPC BLAS WEBGPU HEXAGON ZENDNN ZDNN OPENVINO VIRTGPU VIRTGPU_BACKEND ACCELERATE OPENMP)
    set(GGML_${backend} OFF CACHE BOOL "" FORCE)
endforeach()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(../../deps/llama.cpp/ggml ggml EXCLUDE_FROM_ALL)

find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
        URL https://codeload.github.com/nlohmann/json/tar.gz/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
        URL_HASH SHA256=0dbc5e40a01ff142e7e68c03e85247a4dcede2f592d12d3677dee3664d17975a
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(json)
endif()

# Reuse the accepted target's pinned codec downloads and two-job build rule.
set(DS4V_PREPROCESS_WITH_CODECS ON CACHE BOOL "" FORCE)
add_subdirectory(../ds4v_preprocess_probe codecs EXCLUDE_FROM_ALL)
set(DS4V_JINJA ../../deps/llama.cpp/common/jinja)
add_executable(ds4v_image_prepare
    test.cpp compose.cpp
    ../../src/server/image_input.cpp
    ../../src/server/chat_template.cpp
    ../../src/server/tokenizer.cpp
    ../../src/deepseek4/deepseek4_vision_decode.cpp
    ../../src/deepseek4/deepseek4_vision_preprocess.cpp
    ../../src/deepseek4/deepseek4_image_prompt.cpp
    ${DS4V_JINJA}/lexer.cpp ${DS4V_JINJA}/parser.cpp
    ${DS4V_JINJA}/runtime.cpp ${DS4V_JINJA}/value.cpp
    ${DS4V_JINJA}/string.cpp ${DS4V_JINJA}/caps.cpp
    ../../deps/llama.cpp/common/unicode.cpp)
target_include_directories(ds4v_image_prepare PRIVATE ../../src ../../deps/llama.cpp/common)
target_link_libraries(ds4v_image_prepare PRIVATE ggml-base nlohmann_json::nlohmann_json ds4v_libjpeg ds4v_lodepng)
enable_testing()
set(DS4V_TOKENIZER_GGUF "" CACHE FILEPATH "Converter smoke GGUF; metadata only")
set(DS4V_SOURCE_FIXTURES "" CACHE PATH "Immutable preprocessing fixtures")
if(DS4V_TOKENIZER_GGUF AND DS4V_SOURCE_FIXTURES)
    add_test(NAME ds4v_image_prepare_composition COMMAND ds4v_image_prepare ${DS4V_TOKENIZER_GGUF} ${DS4V_SOURCE_FIXTURES})
endif()
