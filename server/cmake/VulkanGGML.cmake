# Vulkan is a separate executable: never replace the CUDA/HIP ggml vendor tree.
include(FetchContent)
find_package(Git REQUIRED)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(luce_vulkan_ggml
    URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/c061df19838ff60970faf54fd7e414953590125d
    URL_HASH SHA256=d345a35f541d8f23ca519df580ed9f1ffcdd3a6f46a54ac97c80659fad8694d3
    SOURCE_SUBDIR ggml
    PATCH_COMMAND ${GIT_EXECUTABLE} apply --whitespace=error
        ${CMAKE_CURRENT_LIST_DIR}/patches/vulkan-w9100-rows.patch)
FetchContent_MakeAvailable(luce_vulkan_ggml)
