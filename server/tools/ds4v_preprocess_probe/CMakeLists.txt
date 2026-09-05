cmake_minimum_required(VERSION 3.20)
project(ds4v_preprocess_probe LANGUAGES CXX)

option(DS4V_PREPROCESS_WITH_CODECS "Build pinned JPEG/PNG decoder gate" ON)

add_executable(ds4v_preprocess_probe
    ds4v_preprocess_probe.cpp
    ../../src/deepseek4/deepseek4_vision_preprocess.cpp)
target_include_directories(ds4v_preprocess_probe PRIVATE ../../src/deepseek4)
target_compile_features(ds4v_preprocess_probe PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ds4v_preprocess_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(DS4V_PREPROCESS_WITH_CODECS)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/Ds4vImageCodecs.cmake")

    target_sources(ds4v_preprocess_probe PRIVATE
        ../../src/deepseek4/deepseek4_vision_decode.cpp)
    target_compile_definitions(ds4v_preprocess_probe PRIVATE DS4V_PREPROCESS_WITH_CODECS=1)
    target_link_libraries(ds4v_preprocess_probe PRIVATE ds4v_libjpeg ds4v_lodepng)
endif()

enable_testing()
add_test(NAME ds4v_preprocess_self_test COMMAND ds4v_preprocess_probe --self-test)
