cmake_minimum_required(VERSION 3.21)  # 3.21 adds first-class HIP language support (project(LANGUAGES ... HIP))
set(DFLASH27B_GPU_BACKEND "cuda" CACHE STRING "GPU backend to build: cuda or hip")
set_property(CACHE DFLASH27B_GPU_BACKEND PROPERTY STRINGS cuda hip)
string(TOLOWER "${DFLASH27B_GPU_BACKEND}" DFLASH27B_GPU_BACKEND)
if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    set(DFLASH27B_USER_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    project(dflash27b LANGUAGES C CXX CUDA)
elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    project(dflash27b LANGUAGES C CXX HIP)
else()
    message(FATAL_ERROR "DFLASH27B_GPU_BACKEND must be 'cuda' or 'hip', got '${DFLASH27B_GPU_BACKEND}'")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# If we do not set this, ggml will output to bin and DLLs will not load
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# ROCm root for HIP builds (rpath + rocwmma header discovery).
if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    if(DEFINED ROCM_PATH)
        set(_dflash27b_rocm_root "${ROCM_PATH}")
    elseif(DEFINED ENV{ROCM_PATH})
        set(_dflash27b_rocm_root "$ENV{ROCM_PATH}")
    elseif(EXISTS "/opt/rocm")
        set(_dflash27b_rocm_root "/opt/rocm")
    else()
        set(_dflash27b_rocm_root "")
    endif()
endif()
# Bake portable rpath into all executables so bundled ggml backend libs / libggml-base
# are found regardless of LD_LIBRARY_PATH or stale /usr/local/lib (closes #31).
set(CMAKE_INSTALL_RPATH "$ORIGIN/deps/llama.cpp/ggml/src;$ORIGIN/deps/llama.cpp/ggml/src/ggml-cuda;$ORIGIN/deps/llama.cpp/ggml/src/ggml-hip;$ORIGIN/../deps/llama.cpp/ggml/src;$ORIGIN/../deps/llama.cpp/ggml/src/ggml-cuda;$ORIGIN/../deps/llama.cpp/ggml/src/ggml-hip")
if(DFLASH27B_GPU_BACKEND STREQUAL "hip" AND _dflash27b_rocm_root)
    list(APPEND CMAKE_BUILD_RPATH "${_dflash27b_rocm_root}/lib" "${_dflash27b_rocm_root}/lib64")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${_dflash27b_rocm_root}/lib;${_dflash27b_rocm_root}/lib64")
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ─── ggml (from the llama.cpp submodule) ─────────────────────────────
#
# We use only ggml from the vendored llama.cpp submodule. The drafter is
# loaded via our own custom Qwen3-0.6B forward
# (src/qwen3/qwen3_loader.cpp + src/qwen3/qwen3_graph.cpp)
# rather than libllama, so libllama is not built.
#
# No BLAS, no Metal, no Vulkan, no examples/tests/tools.

if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_HIP  OFF CACHE BOOL "" FORCE)
    set(DFLASH27B_GGML_BACKEND_TARGET ggml-cuda)
elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_HIP  ON  CACHE BOOL "" FORCE)
    set(GGML_HIP_RCCL OFF CACHE BOOL "" FORCE)
    set(DFLASH27B_GGML_BACKEND_TARGET ggml-hip)
    set(DFLASH27B_HIP_ARCHITECTURES "" CACHE STRING "HIP GPU targets, e.g. gfx906;gfx1100")
    if(DFLASH27B_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "${DFLASH27B_HIP_ARCHITECTURES}" CACHE STRING "" FORCE)
    endif()
endif()
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(GGML_METAL        OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN       OFF CACHE BOOL "" FORCE)
set(GGML_BLAS         OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL       OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "deps/llama.cpp submodule missing. Run: "
        "git submodule update --init --recursive")
endif()
# Asymmetric KV-quant (Q4_0 K with Q8_0 V etc) needs all fattn quant pairs.
# Build time grows ~3x. Set DFLASH27B_FA_ALL_QUANTS=OFF if you only run
# the spec_prefill demo (target_gen path uses standard quant pairs).
option(DFLASH27B_FA_ALL_QUANTS "Compile ggml-cuda fattn kernels for all KV-quant pairs" ON)
set(GGML_CUDA_FA_ALL_QUANTS ${DFLASH27B_FA_ALL_QUANTS} CACHE BOOL "" FORCE)

# HIP Phase 2 — opt-in rocWMMA flashprefill kernels (Strix Halo / gfx1151).
# Default OFF (Phase 1 = ggml q8 fallback). Set ON for the 1.7-2.7× compress
# speedup at 8K-32K context. Requires rocwmma headers installed under
# ${ROCM_PATH}/include (e.g. `sudo apt install rocwmma`).
option(DFLASH27B_HIP_SM80_EQUIV
    "HIP: build the rocWMMA flashprefill kernels (Phase 2). Requires rocwmma."
    OFF)

# Resolve the CUDA architecture list up-front so downstream logic (notably
# the consumer-Blackwell ggml workaround below) can inspect the actual arches
# nvcc will compile for. HIP builds resolve their own GPU architecture list
# (defaulting to gfx1151 / Strix Halo) and pass it through to the HIP
# toolchain and ggml backend.
if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    # Pascal (60/61/62), Volta (70), Turing (75) and Ampere (86) always;
    # Blackwell consumer (120) and Thor
    # (110 on CUDA 13+) added when nvcc supports them. DGX Spark /
    # GB10 is compute capability 12.1 (121), added at CUDA 12.9+.
    if(DFLASH27B_USER_CUDA_ARCHITECTURES)
        set(_dflash27b_archs "${DFLASH27B_USER_CUDA_ARCHITECTURES}")
    else()
        set(_dflash27b_archs "60;61;62;70;75;86")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND _dflash27b_archs "120")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
            list(APPEND _dflash27b_archs "110")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
            list(APPEND _dflash27b_archs "121")
        endif()
    endif()
elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    # User override precedence: -DDFLASH27B_HIP_ARCHITECTURES → -DAMDGPU_TARGETS
    # → gfx1151 default (Strix Halo).
    if(DFLASH27B_HIP_ARCHITECTURES)
        set(_dflash27b_archs "${DFLASH27B_HIP_ARCHITECTURES}")
    elseif(AMDGPU_TARGETS)
        set(_dflash27b_archs "${AMDGPU_TARGETS}")
    else()
        set(_dflash27b_archs "gfx1151")
    endif()
    # Make sure the HIP language picks up the resolved arch list.
    set(CMAKE_HIP_ARCHITECTURES "${_dflash27b_archs}" CACHE STRING "" FORCE)
else()
    set(_dflash27b_archs "")
endif()

# Consumer Blackwell workaround: skip sm_12x→sm_12xa replacement and FP4
# mmq kernels that can trigger illegal-instruction faults on consumer chips.
# By default, auto-enable when the resolved CUDA arch list includes a 12x
# entry. Set DFLASH27B_USE_BLACKWELL_CONSUMER_FIX=ON to force this behavior
# explicitly (for cross-compiles or custom arch lists).
option(DFLASH27B_USE_BLACKWELL_CONSUMER_FIX
    "Enable ggml consumer-Blackwell workaround (skip sm_12x→sm_12xa, exclude FP4 mmq kernels)" OFF)
if(DFLASH27B_USE_BLACKWELL_CONSUMER_FIX)
    set(_dflash_is_consumer_blackwell ON)
endif()

if(DFLASH27B_GPU_BACKEND STREQUAL "cuda" AND NOT DEFINED _dflash_is_consumer_blackwell)
    set(_dflash_is_consumer_blackwell OFF)
    # Iterate the resolved dflash27b arch list, not raw CMAKE_CUDA_ARCHITECTURES,
    # which is empty on the default path (the project supplies its own list above).
    foreach(_arch IN LISTS _dflash27b_archs)
        string(REGEX REPLACE "[^0-9]" "" _dflash_arch_num "${_arch}")
        if(_dflash_arch_num MATCHES "^12[0-9]$")
            set(_dflash_is_consumer_blackwell ON)
            break()
        endif()
    endforeach()
endif()

if(DFLASH27B_GPU_BACKEND STREQUAL "cuda" AND _dflash_is_consumer_blackwell)
    set(GGML_CUDA_BLACKWELL_CONSUMER ON CACHE BOOL
        "Skip sm_12X→sm_12Xa for consumer Blackwell (no FP4)" FORCE)
endif()
# Use only the ggml subtree of llama.cpp (skip libllama).
add_subdirectory(deps/llama.cpp/ggml EXCLUDE_FROM_ALL)

if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    # The vendored ggml HIP shim still uses a few CUDA spellings that are not
    # mapped in this submodule snapshot. Keep the compatibility layer in this
    # repo so the parent PR is reproducible without a dirty submodule checkout.
    target_compile_definitions(ggml-hip PRIVATE
        cublasSgemmStridedBatched=hipblasSgemmStridedBatched
        cudaStreamCaptureStatus=hipStreamCaptureStatus
        cudaStreamCaptureStatusNone=hipStreamCaptureStatusNone
        cudaStreamIsCapturing=hipStreamIsCapturing
    )
    target_include_directories(ggml-hip BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hip_compat)
endif()

if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    # The CUDA-only sources include <cuda_runtime.h> directly, so the toolkit
    # headers must be available when compiling the library.
    find_package(CUDAToolkit REQUIRED)
elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    find_package(hip REQUIRED)
endif()

# ─── dflash27b static library ──────────────────────────────────────

set(DFLASH27B_SRC_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/draft
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen35
    ${CMAKE_CURRENT_SOURCE_DIR}/src/laguna
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen3
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gemma4
)

add_library(dflash27b STATIC
    src/errors.cpp
    src/qwen35/gguf_target_loader.cpp
    src/qwen35/qwen35_target_graph.cpp
    src/draft/draft_gguf_loader.cpp
    src/draft/draft_safetensors_loader.cpp
    src/draft/draft_dflash_graph.cpp
    src/qwen3/qwen3_drafter.cpp
    src/qwen3/qwen3_loader.cpp
    src/qwen3/qwen3_graph.cpp
    src/qwen3/qwen3_backend.cpp
    src/qwen3/qwen3_daemon.cpp
    src/gemma4/gemma4_loader.cpp
    src/gemma4/gemma4_graph.cpp
    src/gemma4/gemma4_backend.cpp
    src/gemma4/gemma4_daemon.cpp
    src/flashprefill_q8.cpp
    src/kv_cache.cpp
    src/kv_quant.cpp
    src/delta_net_chunked.cpp
    # Laguna-XS.2 (Poolside) target arch
    src/laguna/laguna_target_loader.cpp
    src/laguna/laguna_target_graph.cpp
    src/laguna/laguna_daemon.cpp
    src/laguna/laguna_backend.cpp
    src/common/dflash_feature_ring.cpp
    src/common/dflash_capture.cpp
    src/common/dflash_draft_ipc.cpp
    src/common/dflash_draft_ipc_daemon.cpp
    src/common/dflash_draft_graph.cpp
    src/common/dflash_spec_decode.cpp
    src/qwen35/graph_builders.cpp
    src/qwen35/layer_split_forward.cpp
    src/qwen35/layer_split_daemon.cpp
    src/qwen35/qwen35_backend.cpp
    src/qwen35/qwen35_dflash_target.cpp
    src/qwen35/qwen35_layer_split_dflash_target.cpp
    src/qwen35/layer_split_daemon_loop.cpp
    src/qwen35/qwen35_daemon.cpp
    src/common/sampler.cpp
    src/common/daemon_loop.cpp
    src/common/gguf_inspect.cpp
    src/common/layer_split_utils.cpp
    src/common/ddtree.cpp
    src/common/peer_access.cpp
)
# BSA (Block-Sparse Attention) backs the speculative-prefill drafter scoring
# path. Default ON so prefill is fast out of the box. Turn OFF if you don't
# run the spec-prefill stack or are building without CUDA BF16 WMMA support
# (cutlass BSA kernels need sm_80+; the auto-detect below also disables BSA on
# legacy CUDA arches).
if(NOT DEFINED DFLASH27B_ENABLE_BSA)
    set(DFLASH27B_ENABLE_BSA ON)
endif()
if(DFLASH27B_GPU_BACKEND STREQUAL "hip" AND DFLASH27B_ENABLE_BSA AND NOT DFLASH27B_HIP_SM80_EQUIV)
    message(STATUS
        "BSA on HIP is backed by the rocWMMA sparse-FA kernel (Phase 2). "
        "Build with -DDFLASH27B_HIP_SM80_EQUIV=ON to enable; disabling BSA for this Phase 1 HIP build.")
    set(DFLASH27B_ENABLE_BSA OFF)
endif()

# Apply the arch list to local sources. HIP builds pass CMAKE_HIP_ARCHITECTURES
# through to the HIP toolchain/ggml backend and optionally build rocWMMA kernels.
if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    target_sources(dflash27b PRIVATE src/cuda_cross_device_copy.cpp)
    set_target_properties(dflash27b PROPERTIES CUDA_ARCHITECTURES "${_dflash27b_archs}")
    list(GET _dflash27b_archs 0 _dflash27b_cuda_min_sm)
    # Strip any trailing 'a' suffix (e.g. "121a" -> "121")
    string(REGEX REPLACE "[^0-9]" "" _dflash27b_cuda_min_sm "${_dflash27b_cuda_min_sm}")
    target_compile_definitions(dflash27b PRIVATE
        DFLASH27B_BACKEND_CUDA=1
        DFLASH27B_CUDA_MIN_SM=${_dflash27b_cuda_min_sm}
        DFLASH27B_MIN_SM=${_dflash27b_cuda_min_sm})
elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    set_target_properties(dflash27b PROPERTIES HIP_ARCHITECTURES "${_dflash27b_archs}")
    target_compile_definitions(dflash27b PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
endif()

# FlashPrefill custom kernels.
#   - CUDA sm_80+: BF16 WMMA (m16n16k16) + cp.async — flashprefill_kernels.cu
#   - CUDA sm_70–sm_79 (Volta/Turing): F16 WMMA (m16n8k16), cooperative loads — flashprefill_f16.cu
#   - CUDA sm_60–sm_69 (Pascal): scalar F16, no tensor cores — flashprefill_scalar.cu
#   - HIP Phase 1 (default): ggml q8 fallback, no custom kernels.
#   - HIP Phase 2 (DFLASH27B_HIP_SM80_EQUIV=ON): rocWMMA-native kernels.
# The dispatch in qwen3_graph.cpp checks buffer type at runtime:
#   BF16 buffers → bf16 WMMA kernel; F16 buffers → f16 WMMA kernel; else → ggml FA.
if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    if(DFLASH27B_HIP_SM80_EQUIV)
        find_path(DFLASH27B_ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp
            HINTS "${_dflash27b_rocm_root}/include" /opt/rocm/include
            NO_DEFAULT_PATH)
        if(NOT DFLASH27B_ROCWMMA_INCLUDE_DIR)
            message(FATAL_ERROR
                "DFLASH27B_HIP_SM80_EQUIV=ON but rocwmma/rocwmma.hpp not found. "
                "Install rocwmma (e.g. `sudo apt install rocwmma` on Debian/Ubuntu, "
                "or fetch headers from https://github.com/ROCm/rocWMMA), or rebuild "
                "with -DDFLASH27B_HIP_SM80_EQUIV=OFF (uses the slower q8 fallback).")
        endif()
        target_sources(dflash27b PRIVATE
            src/flashprefill_kernels.hip.cu
            src/flashprefill_select.cpp
            src/flashprefill.cpp)
        set_source_files_properties(src/flashprefill_kernels.hip.cu
            PROPERTIES LANGUAGE HIP)
        target_include_directories(dflash27b PRIVATE
            ${DFLASH27B_ROCWMMA_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
        target_compile_definitions(dflash27b PRIVATE
            DFLASH27B_HAVE_FLASHPREFILL=1
            DFLASH27B_CUDA_MIN_SM=80)
        message(STATUS "dflash: HIP Phase 2 - rocWMMA flashprefill kernels enabled")
    else()
        message(STATUS "dflash: HIP Phase 1 - ggml q8 fallback for flashprefill")
    endif()
elseif(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    target_sources(dflash27b PRIVATE
        src/flashprefill_select.cpp
        src/flashprefill.cpp)
    # Multi-arch: scan all resolved arches and compile every applicable
    # flashprefill kernel variant.  This lets a single binary run on mixed
    # GPUs (e.g. Volta sm_70 + Pascal sm_61) without "no kernel image" errors.
    set(_dflash_has_sm80 OFF)
    set(_dflash_has_sm70 OFF)
    set(_dflash_has_sm60 OFF)
    foreach(_arch IN LISTS _dflash27b_archs)
        string(REGEX REPLACE "[^0-9]" "" _arch_num "${_arch}")
        if(_arch_num GREATER_EQUAL 80)
            set(_dflash_has_sm80 ON)
        endif()
        if(_arch_num GREATER_EQUAL 70 AND _arch_num LESS 80)
            set(_dflash_has_sm70 ON)
        endif()
        if(_arch_num GREATER_EQUAL 60 AND _arch_num LESS 70)
            set(_dflash_has_sm60 ON)
        endif()
    endforeach()
    # Also honour the single-arch path (min_sm) so non-multi builds keep
    # their existing behaviour when only one tier matches.
    if(_dflash27b_cuda_min_sm GREATER_EQUAL 80)
        set(_dflash_has_sm80 ON)
    elseif(_dflash27b_cuda_min_sm GREATER_EQUAL 70)
        set(_dflash_has_sm70 ON)
    elseif(_dflash27b_cuda_min_sm GREATER_EQUAL 60)
        set(_dflash_has_sm60 ON)
    endif()

    if(_dflash_has_sm80)
        target_sources(dflash27b PRIVATE
            src/flashprefill_kernels.cu
            src/pflash_ggml_adapter.cpp)
        target_compile_definitions(dflash27b PRIVATE
            DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL=1
            DFLASH27B_HAVE_SM80_FLASHPREFILL=1)
        # BF16 WMMA kernels require sm_80+.  Restrict compilation to those
        # arches so nvcc never instantiates m16n16k16 BF16 WMMA on Volta/Pascal.
        set_source_files_properties(src/flashprefill_kernels.cu PROPERTIES
            CUDA_ARCHITECTURES "80;86;90;100;110;120;121")
    endif()
    if(_dflash_has_sm70)
        target_sources(dflash27b PRIVATE src/flashprefill_f16.cu)
        target_compile_definitions(dflash27b PRIVATE
            DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL=1
            DFLASH27B_HAVE_VOLTA_FLASHPREFILL=1)
        # F16 WMMA kernels require sm_70+.  Restrict compilation to Volta/Turing
        # so nvcuda::wmma is never instantiated for Pascal (sm_60-69).
        set_source_files_properties(src/flashprefill_f16.cu PROPERTIES
            CUDA_ARCHITECTURES "70;75")
    endif()
    if(_dflash_has_sm60)
        target_sources(dflash27b PRIVATE src/flashprefill_scalar.cu)
        target_compile_definitions(dflash27b PRIVATE
            DFLASH27B_HAVE_CUDA_SCALAR_FLASHPREFILL=1
            DFLASH27B_HAVE_PASCAL_FLASHPREFILL=1)
        # Scalar Pascal kernels only target sm_60-69.
        set_source_files_properties(src/flashprefill_scalar.cu PROPERTIES
            CUDA_ARCHITECTURES "60;61;62")
    endif()
endif()

# BSA needs sm_80+ (FA-2 derived kernel uses BF16 tensor cores). Auto-disable
# on legacy arches with a clear message instead of failing the link.
if(DFLASH27B_GPU_BACKEND STREQUAL "hip" AND DFLASH27B_ENABLE_BSA)
    message(WARNING
        "DFLASH27B_ENABLE_BSA=ON requested with DFLASH27B_GPU_BACKEND=hip; "
        "disabling BSA because the bundled implementation is CUDA/CUTLASS-only.")
    set(DFLASH27B_ENABLE_BSA OFF)
endif()
if(DFLASH27B_ENABLE_BSA)
    foreach(_arch IN LISTS _dflash27b_archs)
        if(_arch LESS 80)
            message(WARNING
                "DFLASH27B_ENABLE_BSA=ON requested but CUDA_ARCHITECTURES contains '${_arch}' (<80); "
                "disabling BSA (the spec-prefill path will fall back to the WMMA kernel).")
            set(DFLASH27B_ENABLE_BSA OFF)
            break()
        endif()
    endforeach()
endif()
# BSA on CUDA also requires the Block-Sparse-Attention submodule (with bundled cutlass).
# Existing users who `git pull` without `--recurse-submodules` would otherwise
# hit a missing-header compile error — auto-disable here with a clear message.
if(DFLASH27B_ENABLE_BSA AND DFLASH27B_GPU_BACKEND STREQUAL "cuda" AND NOT EXISTS
   "${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include/cutlass/numeric_types.h")
    message(WARNING
        "DFLASH27B_ENABLE_BSA=ON requested but the Block-Sparse-Attention submodule "
        "is missing or its bundled cutlass is not initialized. Run "
        "`git submodule update --init --recursive` to enable the BSA spec-prefill "
        "path. Disabling BSA for this build.")
    set(DFLASH27B_ENABLE_BSA OFF)
endif()
if(DFLASH27B_ENABLE_BSA)
    if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
        # HIP BSA: backed by our rocWMMA sparse-FA kernel; no CUTLASS needed.
        target_sources(dflash27b PRIVATE src/bsa_launcher_hip.cu)
        set_source_files_properties(src/bsa_launcher_hip.cu PROPERTIES LANGUAGE HIP)
    else()
        target_sources(dflash27b PRIVATE src/bsa_fwd_inst.cu src/bsa_launcher.cu)
    endif()
endif()

target_include_directories(dflash27b
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${DFLASH27B_SRC_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/src
)
if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    target_include_directories(dflash27b PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
if(DFLASH27B_ENABLE_BSA)
    if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
        # HIP path: bsa_launcher_hip.cu only needs DFLASH27B_HAVE_BSA + the
        # hip_compat shim (which the Phase 2 block above already adds).
        target_compile_definitions(dflash27b PRIVATE DFLASH27B_HAVE_BSA=1)
    else()
        target_include_directories(dflash27b PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/bsa_stubs
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/block_sparse_attn/src)
        target_compile_options(dflash27b PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        target_compile_definitions(dflash27b PRIVATE FLASHATTENTION_DISABLE_DROPOUT FLASH_NAMESPACE=flash DFLASH27B_HAVE_BSA=1)
        # MSVC's <cmath> hides POSIX M_* macros (M_LOG2E etc.) unless _USE_MATH_DEFINES
        # is set before any cmath include. BSA's softmax.h relies on M_LOG2E; define
        # globally on the target so it precedes every TU's first <cmath> include.
        if(WIN32)
            target_compile_definitions(dflash27b PRIVATE _USE_MATH_DEFINES)
        endif()
    endif()
endif()

target_link_libraries(dflash27b
    PUBLIC
        ggml
        ${DFLASH27B_GGML_BACKEND_TARGET}
        ggml-base
)
if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    target_link_libraries(dflash27b PRIVATE hip::host)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     target_compile_options(dflash27b PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-function>
     )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
     target_compile_options(dflash27b PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
     )
endif()

# PFlash phase-split harness

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/pflash_daemon.cpp")
    add_executable(pflash_daemon test/pflash_daemon.cpp)
    target_include_directories(pflash_daemon PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
    target_link_libraries(pflash_daemon PRIVATE dflash27b ggml ${DFLASH27B_GGML_BACKEND_TARGET})
endif()

# ─── Tests (numerics vs oracle) ────────────────────────────────────

option(DFLASH27B_TESTS "Build numerics tests" ON)
if(DFLASH27B_TESTS)
    if(DFLASH27B_GPU_BACKEND STREQUAL "cuda" AND _dflash27b_cuda_min_sm GREATER_EQUAL 80 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_flashprefill_kernels.cpp")
        add_executable(test_flashprefill_kernels test/test_flashprefill_kernels.cpp)
        set_target_properties(test_flashprefill_kernels PROPERTIES CUDA_ARCHITECTURES "${_dflash27b_archs}")
        target_include_directories(test_flashprefill_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_flashprefill_kernels PRIVATE dflash27b CUDA::cudart)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_kv_quant.cpp")
        add_executable(test_kv_quant test/test_kv_quant.cpp)
        target_include_directories(test_kv_quant PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_kv_quant PRIVATE dflash27b)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_draft_vs_reference.cpp")
        add_executable(test_draft_vs_reference test/test_draft_vs_reference.cpp)
        target_link_libraries(test_draft_vs_reference PRIVATE dflash27b)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_draft.cpp")
        add_executable(smoke_load_draft test/smoke_load_draft.cpp)
        target_include_directories(smoke_load_draft PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_load_draft PRIVATE dflash27b ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/spike_thin_copy.cpp")
        add_executable(spike_thin_copy test/spike_thin_copy.cpp)
        target_include_directories(spike_thin_copy PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(spike_thin_copy PRIVATE ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_draft_graph.cpp")
        add_executable(smoke_draft_graph test/smoke_draft_graph.cpp)
        target_include_directories(smoke_draft_graph PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_draft_graph PRIVATE dflash27b ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_qwen3_forward.cpp")
        add_executable(smoke_qwen3_forward test/smoke_qwen3_forward.cpp)
        target_include_directories(smoke_qwen3_forward PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_qwen3_forward PRIVATE dflash27b ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_vs_oracle.cpp")
        add_executable(test_vs_oracle test/test_vs_oracle.cpp)
        target_include_directories(test_vs_oracle PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_vs_oracle PRIVATE dflash27b ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_target.cpp")
        add_executable(smoke_load_target test/smoke_load_target.cpp)
        target_include_directories(smoke_load_target PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_load_target PRIVATE dflash27b ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_target_laguna.cpp")
        add_executable(smoke_load_target_laguna test/smoke_load_target_laguna.cpp)
        target_include_directories(smoke_load_target_laguna PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_load_target_laguna PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_laguna_forward.cpp")
        add_executable(smoke_laguna_forward test/smoke_laguna_forward.cpp)
        target_include_directories(smoke_laguna_forward PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_laguna_forward PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_ttft.cpp")
        add_executable(bench_laguna_ttft test/bench_laguna_ttft.cpp)
        target_include_directories(bench_laguna_ttft PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(bench_laguna_ttft PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_pflash.cpp")
        add_executable(bench_laguna_pflash test/bench_laguna_pflash.cpp)
        target_include_directories(bench_laguna_pflash PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(bench_laguna_pflash PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_generate.cpp")
        add_executable(bench_laguna_generate test/bench_laguna_generate.cpp)
        target_include_directories(bench_laguna_generate PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(bench_laguna_generate PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_laguna_daemon.cpp")
        add_executable(test_laguna_daemon test/test_laguna_daemon.cpp)
        target_include_directories(test_laguna_daemon PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_laguna_daemon PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_target_forward.cpp")
        add_executable(smoke_target_forward test/smoke_target_forward.cpp)
        target_include_directories(smoke_target_forward PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_target_forward PRIVATE dflash27b ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_generate.cpp")
        add_executable(test_generate test/test_generate.cpp)
        target_include_directories(test_generate PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_generate PRIVATE dflash27b ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_dflash.cpp")
        add_executable(test_dflash test/test_dflash.cpp)
        target_include_directories(test_dflash PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
            target_compile_definitions(test_dflash PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
        else()
            target_compile_definitions(test_dflash PRIVATE
                DFLASH27B_BACKEND_CUDA=1
                DFLASH27B_CUDA_MIN_SM=${_dflash27b_cuda_min_sm})
        endif()
        target_link_libraries(test_dflash PRIVATE dflash27b ggml ${DFLASH27B_GGML_BACKEND_TARGET})
        # test_dflash uses cuda*/hip* runtime calls directly for fast rollback,
        # peer access, and target-layer split copies. Link the selected runtime.
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(test_dflash PRIVATE CUDA::cudart)
        else()
            target_link_libraries(test_dflash PRIVATE hip::host)
        endif()
        # OpenMP for parallel CPU top-K extraction in the ddtree path.
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(test_dflash PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()

    if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    # internal.h includes <cuda_runtime.h> when GGML_USE_CUDA is set; link
    # CUDA::cudart so the toolkit headers are on the compile line (same as
    # test_dflash historically had alone).
    set(_dflash_internal_h_cuda_tests
        smoke_load_draft
        smoke_draft_graph
        test_vs_oracle
        smoke_load_target
        smoke_load_target_laguna
        smoke_laguna_forward
        bench_laguna_ttft
        bench_laguna_pflash
        bench_laguna_generate
        smoke_target_forward
        test_generate
        test_dflash
    )
    foreach(_t IN LISTS _dflash_internal_h_cuda_tests)
        if(TARGET ${_t})
            target_link_libraries(${_t} PRIVATE CUDA::cudart)
        endif()
    endforeach()

    # Gated on the same condition as src/pflash_ggml_adapter.cpp above:
    # the adapter is only compiled into dflash27b when the CUDA arch list
    # includes an sm_80+ target, so this test can only link there too.
    if(DFLASH27B_GPU_BACKEND STREQUAL "cuda" AND _dflash_has_sm80
       AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_flash_attn_sparse.cpp")
        add_executable(test_flash_attn_sparse test/test_flash_attn_sparse.cpp)
        target_link_libraries(test_flash_attn_sparse PRIVATE dflash27b ggml ggml-cuda ggml-base)
        target_include_directories(test_flash_attn_sparse PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/include
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/src
            ${DFLASH27B_SRC_INCLUDE_DIRS})
    endif()
    endif()
endif()
