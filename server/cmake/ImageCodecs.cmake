# JPEG and PNG decoders behind common/vision/image_decode: libjpeg-turbo from
# its pinned release archive, lodepng from two files at a pinned commit. License
# texts are in ImageCodecs.NOTICES.md and the libjpeg-turbo archive.
include_guard(GLOBAL)

include(ExternalProject)

# DOWNLOAD_EXTRACT_TIMESTAMP exists from CMake 3.24; older releases would read
# it as part of URL_HASH.
set(IMAGE_CODEC_EXTRACT_TIMESTAMP)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(IMAGE_CODEC_EXTRACT_TIMESTAMP DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
endif()

set(LUCE_IMAGE_CODECS_CACHE_DIR "" CACHE PATH
    "Optional persistent directory to cache downloaded image codec sources")
set(_luce_jpeg_download_dir_arg)
if(LUCE_IMAGE_CODECS_CACHE_DIR)
    file(MAKE_DIRECTORY "${LUCE_IMAGE_CODECS_CACHE_DIR}")
    set(_luce_jpeg_download_dir_arg DOWNLOAD_DIR "${LUCE_IMAGE_CODECS_CACHE_DIR}")
endif()

set(IMAGE_CODEC_JPEG_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo-prefix)
set(IMAGE_CODEC_JPEG_ARCHIVE_NAME jpeg)
if(MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    set(IMAGE_CODEC_JPEG_ARCHIVE_NAME jpeg-static)
endif()
set(IMAGE_CODEC_JPEG_ARCHIVE
    ${IMAGE_CODEC_JPEG_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${IMAGE_CODEC_JPEG_ARCHIVE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
file(MAKE_DIRECTORY ${IMAGE_CODEC_JPEG_PREFIX}/include)
set(_luce_jpeg_cmake_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${IMAGE_CODEC_JPEG_PREFIX}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DENABLE_SHARED=OFF
    -DENABLE_STATIC=ON
    -DWITH_TOOLS=OFF
    -DWITH_TESTS=OFF
    -DWITH_SIMD=OFF
    -DWITH_TURBOJPEG=OFF)
if(CMAKE_C_COMPILER_LAUNCHER)
    list(APPEND _luce_jpeg_cmake_args
        "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}")
endif()
ExternalProject_Add(libjpeg_turbo_external
    URL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
    URL_HASH SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
    ${_luce_jpeg_download_dir_arg}
    ${IMAGE_CODEC_EXTRACT_TIMESTAMP}
    CMAKE_ARGS ${_luce_jpeg_cmake_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel 2
    BUILD_BYPRODUCTS ${IMAGE_CODEC_JPEG_ARCHIVE})
unset(_luce_jpeg_cmake_args)
unset(_luce_jpeg_download_dir_arg)
add_library(image_codec_jpeg STATIC IMPORTED GLOBAL)
set_target_properties(image_codec_jpeg PROPERTIES
    IMPORTED_LOCATION ${IMAGE_CODEC_JPEG_ARCHIVE}
    INTERFACE_INCLUDE_DIRECTORIES ${IMAGE_CODEC_JPEG_PREFIX}/include)
add_dependencies(image_codec_jpeg libjpeg_turbo_external)

# lodepng has no release archives and GitHub's commit archives are not
# byte-stable, so fetch its two files at a pinned commit (raw files are) and
# check each against its hash. Retried because CI runners drop downloads.
set(IMAGE_CODEC_PNG_COMMIT ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a)
if(LUCE_IMAGE_CODECS_CACHE_DIR)
    set(IMAGE_CODEC_PNG_DIR ${LUCE_IMAGE_CODECS_CACHE_DIR}/lodepng-${IMAGE_CODEC_PNG_COMMIT})
else()
    set(IMAGE_CODEC_PNG_DIR ${CMAKE_CURRENT_BINARY_DIR}/lodepng-${IMAGE_CODEC_PNG_COMMIT})
endif()
foreach(entry
        "lodepng.cpp=d98e1f40d303c1038a096ebf93b413a565a91cf2c72b9d2fa5c625c4279c3cb6"
        "lodepng.h=23c27abb06883ed98184d16d0b20771b526dca1e8e13236c2397316769c0dc8b")
    string(REPLACE "=" ";" entry "${entry}")
    list(GET entry 0 file)
    list(GET entry 1 hash)
    set(target ${IMAGE_CODEC_PNG_DIR}/${file})
    set(have "")
    foreach(attempt RANGE 1 3)
        if(EXISTS ${target})
            file(SHA256 ${target} have)
            if(have STREQUAL hash)
                break()
            endif()
            file(REMOVE ${target})
        endif()
        file(DOWNLOAD
            https://raw.githubusercontent.com/lvandeve/lodepng/${IMAGE_CODEC_PNG_COMMIT}/${file}
            ${target} TLS_VERIFY ON STATUS status)
    endforeach()
    if(EXISTS ${target})
        file(SHA256 ${target} have)
    endif()
    if(NOT have STREQUAL hash)
        message(FATAL_ERROR "lodepng: could not fetch ${file} with SHA256 ${hash} (${status})")
    endif()
endforeach()
add_library(image_codec_png STATIC ${IMAGE_CODEC_PNG_DIR}/lodepng.cpp)
target_include_directories(image_codec_png PUBLIC ${IMAGE_CODEC_PNG_DIR})

# Expose the original JPEG notices for production installation after the
# dependency build has downloaded its hash-verified source archive.
ExternalProject_Get_Property(libjpeg_turbo_external SOURCE_DIR)
set(IMAGE_CODEC_JPEG_SOURCE_DIR "${SOURCE_DIR}")
unset(SOURCE_DIR)
