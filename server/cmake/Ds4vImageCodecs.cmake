# Shared decoder dependencies used by production and the accepted preprocessing
# probe. Keep archive pins and codec options identical to the qualified build.
# License texts remain in tools/ds4v_preprocess_probe/THIRD_PARTY_NOTICES.md
# and the unmodified upstream archives.
include_guard(GLOBAL)

include(ExternalProject)
include(FetchContent)

set(DS4V_JPEG_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo-prefix)
set(DS4V_JPEG_ARCHIVE_NAME jpeg)
if(MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    set(DS4V_JPEG_ARCHIVE_NAME jpeg-static)
endif()
set(DS4V_JPEG_ARCHIVE
    ${DS4V_JPEG_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${DS4V_JPEG_ARCHIVE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
file(MAKE_DIRECTORY ${DS4V_JPEG_PREFIX}/include)
ExternalProject_Add(libjpeg_turbo_external
    URL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
    URL_HASH SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${DS4V_JPEG_PREFIX}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DENABLE_SHARED=OFF
        -DENABLE_STATIC=ON
        -DWITH_TOOLS=OFF
        -DWITH_TESTS=OFF
        -DWITH_SIMD=OFF
        -DWITH_TURBOJPEG=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel 2
    BUILD_BYPRODUCTS ${DS4V_JPEG_ARCHIVE})
add_library(ds4v_libjpeg STATIC IMPORTED GLOBAL)
set_target_properties(ds4v_libjpeg PROPERTIES
    IMPORTED_LOCATION ${DS4V_JPEG_ARCHIVE}
    INTERFACE_INCLUDE_DIRECTORIES ${DS4V_JPEG_PREFIX}/include)
add_dependencies(ds4v_libjpeg libjpeg_turbo_external)

FetchContent_Declare(lodepng
    URL https://github.com/lvandeve/lodepng/archive/ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a.tar.gz
    URL_HASH SHA256=c2459a3f9145258f901d262576f7a56ca08087d3b3efeee3ae033c0952120803
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(lodepng)
add_library(ds4v_lodepng STATIC ${lodepng_SOURCE_DIR}/lodepng.cpp)
target_include_directories(ds4v_lodepng PUBLIC ${lodepng_SOURCE_DIR})

# Expose the original JPEG notices for production installation after the
# dependency build has downloaded its hash-verified source archive.
ExternalProject_Get_Property(libjpeg_turbo_external SOURCE_DIR)
set(DS4V_JPEG_SOURCE_DIR "${SOURCE_DIR}")
unset(SOURCE_DIR)
