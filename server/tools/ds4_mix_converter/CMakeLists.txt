cmake_minimum_required(VERSION 3.21)
project(ds4_mix_converter LANGUAGES C CXX)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://codeload.github.com/nlohmann/json/tar.gz/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
    URL_HASH SHA256=0dbc5e40a01ff142e7e68c03e85247a4dcede2f592d12d3677dee3664d17975a)
FetchContent_MakeAvailable(nlohmann_json)
add_subdirectory(../../deps/llama.cpp/ggml ggml EXCLUDE_FROM_ALL)

add_executable(ds4_mix_converter ds4_mix_converter.cpp)
target_include_directories(ds4_mix_converter PRIVATE
    ../../deps/llama.cpp/ggml/include
    ../../deps/llama.cpp/ggml/rocmfpx)
target_link_libraries(ds4_mix_converter PRIVATE ggml-base nlohmann_json::nlohmann_json Threads::Threads)
if(UNIX)
    target_link_libraries(ds4_mix_converter PRIVATE m)
endif()
target_compile_options(ds4_mix_converter PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_ds4_mix_converter ../../test/test_ds4v_mix_converter.cpp ../../test/test_unit_main.cpp)
target_include_directories(test_ds4_mix_converter PRIVATE
    ../../deps/llama.cpp/ggml/include
    ../../deps/llama.cpp/ggml/rocmfpx)
target_link_libraries(test_ds4_mix_converter PRIVATE ggml-base)
if(UNIX)
    target_link_libraries(test_ds4_mix_converter PRIVATE m)
endif()
enable_testing()
add_test(NAME ds4_mix_converter_codec COMMAND test_ds4_mix_converter)

add_executable(test_ds4_mix_fitter ../../test/test_ds4_mix_fitter.cpp)
target_link_libraries(test_ds4_mix_fitter PRIVATE ggml-base nlohmann_json::nlohmann_json Threads::Threads)
target_include_directories(test_ds4_mix_fitter PRIVATE
    ../../deps/llama.cpp/ggml/include ../../deps/llama.cpp/ggml/rocmfpx)
add_test(NAME ds4_mix_fitter COMMAND test_ds4_mix_fitter)

add_executable(test_ds4_iq_converter ../../test/test_ds4_iq_converter.cpp)
target_link_libraries(test_ds4_iq_converter PRIVATE ggml-base nlohmann_json::nlohmann_json Threads::Threads)
target_include_directories(test_ds4_iq_converter PRIVATE
    ../../deps/llama.cpp/ggml/include ../../deps/llama.cpp/ggml/rocmfpx)
add_test(NAME ds4_iq_converter COMMAND test_ds4_iq_converter)
set_tests_properties(ds4_iq_converter PROPERTIES TIMEOUT 120)
