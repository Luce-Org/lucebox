cmake_minimum_required(VERSION 3.21)  # 3.21 adds first-class HIP language support (project(LANGUAGES ... HIP))
set(DFLASH27B_GPU_BACKEND "cuda" CACHE STRING "GPU backend to build: cuda or hip")
set_property(CACHE DFLASH27B_GPU_BACKEND PROPERTY STRINGS cuda hip)
string(TOLOWER "${DFLASH27B_GPU_BACKEND}" DFLASH27B_GPU_BACKEND)
if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    set(DFLASH27B_USER_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    project(dflash LANGUAGES C CXX CUDA)
elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    project(dflash LANGUAGES C CXX HIP)
else()
    message(FATAL_ERROR "DFLASH27B_GPU_BACKEND must be 'cuda' or 'hip', got '${DFLASH27B_GPU_BACKEND}'")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── nlohmann/json (header-only, prefer system install, fallback to download) ──
find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    set(_dflash_json_fetch_args
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        list(APPEND _dflash_json_fetch_args DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()
    FetchContent_Declare(json ${_dflash_json_fetch_args})
    unset(_dflash_json_fetch_args)
    FetchContent_MakeAvailable(json)
endif()

# If we do not set this, ggml will output to bin and DLLs will not load
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# ROCm root for HIP builds (rpath + rocwmma header discovery).
if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    if(DEFINED ROCM_PATH)
        set(_dflash_rocm_root "${ROCM_PATH}")
    elseif(DEFINED ENV{ROCM_PATH})
        set(_dflash_rocm_root "$ENV{ROCM_PATH}")
    elseif(EXISTS "/opt/rocm")
        set(_dflash_rocm_root "/opt/rocm")
    else()
        set(_dflash_rocm_root "")
    endif()
endif()
# Bake portable rpath into all executables so bundled ggml backend libs / libggml-base
# are found regardless of LD_LIBRARY_PATH or stale /usr/local/lib (closes #31).
set(CMAKE_INSTALL_RPATH "$ORIGIN/deps/llama.cpp/ggml/src;$ORIGIN/deps/llama.cpp/ggml/src/ggml-cuda;$ORIGIN/deps/llama.cpp/ggml/src/ggml-hip;$ORIGIN/../deps/llama.cpp/ggml/src;$ORIGIN/../deps/llama.cpp/ggml/src/ggml-cuda;$ORIGIN/../deps/llama.cpp/ggml/src/ggml-hip")
if(DFLASH27B_GPU_BACKEND STREQUAL "hip" AND _dflash_rocm_root)
    list(APPEND CMAKE_BUILD_RPATH "${_dflash_rocm_root}/lib" "${_dflash_rocm_root}/lib64")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${_dflash_rocm_root}/lib;${_dflash_rocm_root}/lib64")
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ─── ggml (vendored from llama.cpp) ──────────────────────────────────
#
# We use only ggml from the vendored llama.cpp snapshot. The drafter is
# loaded via our own custom Qwen3-0.6B forward
# (src/qwen3/qwen3_loader.cpp + src/qwen3/qwen3_graph.cpp)
# rather than libllama, so libllama is not built.
#
# No BLAS, no Metal, no Vulkan, no examples/tests/tools.

if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_HIP  OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Enable CUDA graphs for AR-decode replay (lucebox)" FORCE)
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
        "Vendored ggml sources missing at deps/llama.cpp/ggml. "
        "Restore them from the repository checkout.")
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
        set(_dflash_archs "${DFLASH27B_USER_CUDA_ARCHITECTURES}")
    else()
        set(_dflash_archs "60;61;62;70;75;86")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND _dflash_archs "120")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
            list(APPEND _dflash_archs "110")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
            list(APPEND _dflash_archs "121")
        endif()
    endif()
    # Keep the vendored ggml-cuda build aligned with the resolved dflash arch list.
    # Otherwise ggml may inherit CMake's compiler-default arch (for example sm_52),
    # which triggers massive first-request PTX JIT on newer GPUs even though
    # dflash_common itself is compiled for the intended arch set.
    set(CMAKE_CUDA_ARCHITECTURES "${_dflash_archs}" CACHE STRING "" FORCE)
elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    # User override precedence: -DDFLASH27B_HIP_ARCHITECTURES → -DAMDGPU_TARGETS
    # → gfx1151 default (Strix Halo).
    if(DFLASH27B_HIP_ARCHITECTURES)
        set(_dflash_archs "${DFLASH27B_HIP_ARCHITECTURES}")
    elseif(AMDGPU_TARGETS)
        set(_dflash_archs "${AMDGPU_TARGETS}")
    else()
        set(_dflash_archs "gfx1151")
    endif()
    # Make sure the HIP language picks up the resolved arch list.
    set(CMAKE_HIP_ARCHITECTURES "${_dflash_archs}" CACHE STRING "" FORCE)
else()
    set(_dflash_archs "")
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
    # Iterate the resolved dflash_common arch list, not raw CMAKE_CUDA_ARCHITECTURES,
    # which is empty on the default path (the project supplies its own list above).
    foreach(_arch IN LISTS _dflash_archs)
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

if(WIN32)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Static ggml on Windows (internal CUDA symbols not dllexported)" FORCE)
endif()

if(WIN32 AND NOT CMAKE_ASM_COMPILER)
    set(CMAKE_ASM_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "" FORCE)
endif()

# Use only the ggml subtree of llama.cpp (skip libllama).
add_subdirectory(deps/llama.cpp/ggml EXCLUDE_FROM_ALL)

if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    # The vendored ggml HIP shim still uses a few CUDA spellings that are not
    # mapped in this upstream snapshot. Keep the compatibility layer in this
    # repo so the build stays reproducible from a clean checkout.
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

# ─── dflash_common static library ──────────────────────────────────────

set(DFLASH27B_SRC_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/draft
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen35
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen35moe
    ${CMAKE_CURRENT_SOURCE_DIR}/src/laguna
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen3
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gemma4
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deepseek4
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server
)

add_library(dflash_common STATIC
    src/errors.cpp
    src/qwen35/gguf_target_loader.cpp
    src/qwen35/qwen35_target_graph.cpp
    src/draft/draft_gguf_loader.cpp
    src/draft/draft_safetensors_loader.cpp
    src/draft/draft_graph.cpp
    src/qwen3/anchor_scan.cpp
    src/qwen3/qwen3_drafter.cpp
    src/qwen3/qwen3_kvflash_scorer.cpp
    src/qwen3/qwen3_loader.cpp
    src/qwen3/qwen3_graph.cpp
    src/qwen3/qwen3_backend.cpp
    src/qwen3/qwen3_daemon.cpp
    src/gemma4/gemma4_loader.cpp
    src/gemma4/gemma4_graph.cpp
    src/gemma4/gemma4_backend.cpp
    src/gemma4/gemma4_daemon.cpp
    src/gemma4/gemma4_dflash_target.cpp
    src/gemma4/gemma4_layer_split_adapter.cpp
    # DeepSeek V4 Flash target arch
    src/deepseek4/deepseek4_loader.cpp
    src/deepseek4/deepseek4_graph.cpp
    src/deepseek4/deepseek4_backend.cpp
    src/deepseek4/deepseek4_daemon.cpp
    src/deepseek4/deepseek4_layer_split_adapter.cpp
    src/deepseek4/deepseek4_target_shard_ipc_daemon.cpp
    src/deepseek4/deepseek4_dspark.cpp
    src/deepseek4/deepseek4_dspark_spec.cpp
    src/flashprefill_q8.cpp
    src/kv_cache.cpp
    src/kv_quant.cpp
    src/delta_net_chunked.cpp
    # Laguna-XS.2 (Poolside) target arch
    src/laguna/laguna_target_loader.cpp
    src/laguna/laguna_target_graph.cpp
    src/laguna/laguna_daemon.cpp
    src/laguna/laguna_backend.cpp
    src/laguna/laguna_layer_split_adapter.cpp
    src/laguna/laguna_dflash_target.cpp
    src/common/backend_ipc.cpp
    src/common/domino_head.cpp
    src/common/dspark_head.cpp
    src/common/target_shard_ipc.cpp
    src/common/target_shard_ipc_daemon.cpp
    src/common/dflash_feature_ring.cpp
    src/common/dflash_capture.cpp
    src/common/dflash_draft_ipc.cpp
    src/common/dflash_draft_ipc_daemon.cpp
    src/common/pflash_drafter_ipc.cpp
    src/common/dflash_draft_graph.cpp
    src/common/dflash_spec_decode.cpp
    src/common/layer_split_backend.cpp
    src/common/layer_split_runtime.cpp
    src/qwen35/graph_builders.cpp
    src/qwen35moe/qwen35moe_ffn.cpp
    src/qwen35moe/qwen35moe_backend.cpp
    src/qwen35moe/qwen35moe_daemon.cpp
    src/qwen35moe/qwen35moe_pipelined_decode.cpp
    # ── Common MoE hybrid infrastructure ──
    src/common/moe_hybrid_placement.cpp
    src/common/moe_hybrid_routing_stats.cpp
    src/common/moe_hybrid_storage.cpp
    src/common/spark_corpus.cpp
    src/common/moe_hybrid_ffn_eval.cpp
    src/common/moe_hybrid_stream.cpp
    src/common/moe_expert_compute.cpp
    src/common/moe_expert_compute_cpu.cpp
    src/common/moe_expert_compute_ipc.cpp
    src/common/moe_hybrid_swap_manager.cpp
    src/common/moe_routing_collector.cpp
    src/qwen35/layer_split_forward.cpp
    src/qwen35/qwen35_target_shard_ipc.cpp
    src/qwen35/qwen35_target_shard_ipc_daemon.cpp
    src/qwen35/layer_split_daemon.cpp
    src/qwen35/qwen35_backend.cpp
    src/qwen35/qwen35_layer_split_adapter.cpp
    src/qwen35/qwen35_dflash_target.cpp
    src/qwen35/qwen35_layer_split_dflash_target.cpp
    src/qwen35/layer_split_daemon_loop.cpp
    src/qwen35/qwen35_daemon.cpp
    src/common/sampler.cpp
    src/common/backend_precision.cpp
    src/common/daemon_loop.cpp
    src/common/gguf_inspect.cpp
    src/common/backend_factory.cpp
    src/placement/placement_config.cpp
    src/common/layer_split_utils.cpp
    src/common/ddtree.cpp
    src/common/peer_access.cpp
    # ── Server components (tokenizer, chat template) ──
    src/server/tokenizer.cpp
    src/server/chat_template.cpp
    src/server/tool_parser.cpp
    src/server/tool_hint.cpp
    src/server/reasoning.cpp
    src/server/tool_memory.cpp
    src/server/sse_emitter.cpp
    src/server/prefix_cache.cpp
    src/server/disk_prefix_cache.cpp
    src/server/freeze_history.cpp
    # ── Jinja chat-template engine (vendored under deps/llama.cpp/common/) ──
    # Used by render_chat_template_jinja() to support --chat-template-file
    # in dflash_server. Mirrors llama.cpp's common_chat_template plumbing.
    # unicode.cpp supplies common_parse_utf8_codepoint() used by jinja's
    # value.cpp tojson() and is otherwise self-contained.
    deps/llama.cpp/common/jinja/lexer.cpp
    deps/llama.cpp/common/jinja/parser.cpp
    deps/llama.cpp/common/jinja/runtime.cpp
    deps/llama.cpp/common/jinja/value.cpp
    deps/llama.cpp/common/jinja/string.cpp
    deps/llama.cpp/common/jinja/caps.cpp
    deps/llama.cpp/common/unicode.cpp
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
    target_sources(dflash_common PRIVATE
        src/cuda_cross_device_copy.cpp
        src/deepseek4/deepseek4_hc_cuda.cu)
    set_target_properties(dflash_common PROPERTIES CUDA_ARCHITECTURES "${_dflash_archs}")
    list(GET _dflash_archs 0 _dflash_cuda_min_sm)
    # Strip any trailing 'a' suffix (e.g. "121a" -> "121")
    string(REGEX REPLACE "[^0-9]" "" _dflash_cuda_min_sm "${_dflash_cuda_min_sm}")
    target_compile_definitions(dflash_common PRIVATE
        DFLASH27B_BACKEND_CUDA=1
        DFLASH27B_CUDA_MIN_SM=${_dflash_cuda_min_sm}
        DFLASH27B_MIN_SM=${_dflash_cuda_min_sm})
elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    target_sources(dflash_common PRIVATE src/deepseek4/deepseek4_hc_cuda.cu)
    set_source_files_properties(src/deepseek4/deepseek4_hc_cuda.cu PROPERTIES LANGUAGE HIP)
    set_target_properties(dflash_common PROPERTIES HIP_ARCHITECTURES "${_dflash_archs}")
    target_compile_definitions(dflash_common PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
    # hip_compat shim is needed by ALL dflash_common sources (peer_access.cpp,
    # dflash_feature_ring.cpp, flashprefill.cpp), not just the SM80_EQUIV path.
    target_include_directories(dflash_common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
endif()

if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    target_link_libraries(dflash_common PUBLIC CUDA::cudart)
elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip")
    target_link_libraries(dflash_common PUBLIC hip::host)
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
    # rms_norm_hip.cu is needed by the HIP chunk-B graph path regardless of SM80_EQUIV.
    target_sources(dflash_common PRIVATE src/rms_norm_hip.cu)
    set_source_files_properties(src/rms_norm_hip.cu PROPERTIES LANGUAGE HIP)
    if(DFLASH27B_HIP_SM80_EQUIV)
        find_path(DFLASH27B_ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp
            HINTS "${_dflash_rocm_root}/include" /opt/rocm/include
            NO_DEFAULT_PATH)
        if(NOT DFLASH27B_ROCWMMA_INCLUDE_DIR)
            message(FATAL_ERROR
                "DFLASH27B_HIP_SM80_EQUIV=ON but rocwmma/rocwmma.hpp not found. "
                "Install rocwmma (e.g. `sudo apt install rocwmma` on Debian/Ubuntu, "
                "or fetch headers from https://github.com/ROCm/rocWMMA), or rebuild "
                "with -DDFLASH27B_HIP_SM80_EQUIV=OFF (uses the slower q8 fallback).")
        endif()
        target_sources(dflash_common PRIVATE
            src/flashprefill_kernels.hip.cu
            src/flashprefill_select.cpp
            src/flashprefill.cpp)
        set_source_files_properties(src/flashprefill_kernels.hip.cu
            PROPERTIES LANGUAGE HIP)
        target_include_directories(dflash_common PRIVATE
            ${DFLASH27B_ROCWMMA_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
        target_compile_definitions(dflash_common PRIVATE
            DFLASH27B_HAVE_FLASHPREFILL=1
            DFLASH27B_CUDA_MIN_SM=80)
        message(STATUS "dflash: HIP Phase 2 - rocWMMA flashprefill kernels enabled")
    else()
        message(STATUS "dflash: HIP Phase 1 - ggml q8 fallback for flashprefill")
    endif()
elseif(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    target_sources(dflash_common PRIVATE
        src/flashprefill_select.cpp
        src/flashprefill.cpp
        src/common/geometric_draft_topk_cuda.cu)
    # PUBLIC so consumers (e.g. the test_dflash executable) also see the macro
    # and take the GPU draft top-K path instead of the CPU fallback.
    target_compile_definitions(dflash_common PUBLIC DFLASH27B_HAVE_DRAFT_TOPK_CUDA=1)
    # GPU port of the sample_logits chain. Compiled in by default; the path is
    # then opted into at runtime via the DFLASH_GPU_SAMPLE env var. Turn the
    # whole thing off at configure time with -DDFLASH_GPU_SAMPLER=OFF.
    option(DFLASH_GPU_SAMPLER "Build the CUDA sample_logits path" ON)
    if(DFLASH_GPU_SAMPLER)
        target_sources(dflash_common PRIVATE src/common/geometric_sampler_cuda.cu)
        # PUBLIC so the backends and test executables that call sample_logits()
        # also compile their GPU dispatch branch.
        target_compile_definitions(dflash_common PUBLIC DFLASH27B_HAVE_GPU_SAMPLER=1)
    endif()
    # Multi-arch: scan all resolved arches and compile every applicable
    # flashprefill kernel variant.  This lets a single binary run on mixed
    # GPUs (e.g. Volta sm_70 + Pascal sm_61) without "no kernel image" errors.
    set(_dflash_has_sm80 OFF)
    set(_dflash_has_sm70 OFF)
    set(_dflash_has_sm60 OFF)
    foreach(_arch IN LISTS _dflash_archs)
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
    if(_dflash_cuda_min_sm GREATER_EQUAL 80)
        set(_dflash_has_sm80 ON)
    elseif(_dflash_cuda_min_sm GREATER_EQUAL 70)
        set(_dflash_has_sm70 ON)
    elseif(_dflash_cuda_min_sm GREATER_EQUAL 60)
        set(_dflash_has_sm60 ON)
    endif()

    if(_dflash_has_sm80)
        target_sources(dflash_common PRIVATE
            src/flashprefill_kernels.cu
            src/pflash_ggml_adapter.cpp)
        target_compile_definitions(dflash_common PRIVATE
            DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL=1
            DFLASH27B_HAVE_SM80_FLASHPREFILL=1)
        # BF16 WMMA kernels require sm_80+.  Restrict compilation to those
        # arches so nvcc never instantiates m16n16k16 BF16 WMMA on Volta/Pascal.
        set_source_files_properties(src/flashprefill_kernels.cu PROPERTIES
            CUDA_ARCHITECTURES "80;86;90;100;110;120;121")
    endif()
    if(_dflash_has_sm70)
        target_sources(dflash_common PRIVATE src/flashprefill_f16.cu)
        target_compile_definitions(dflash_common PRIVATE
            DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL=1
            DFLASH27B_HAVE_VOLTA_FLASHPREFILL=1)
        # F16 WMMA kernels require sm_70+.  Restrict compilation to Volta/Turing
        # so nvcuda::wmma is never instantiated for Pascal (sm_60-69).
        set_source_files_properties(src/flashprefill_f16.cu PROPERTIES
            CUDA_ARCHITECTURES "70;75")
    endif()
    if(_dflash_has_sm60)
        target_sources(dflash_common PRIVATE src/flashprefill_scalar.cu)
        target_compile_definitions(dflash_common PRIVATE
            DFLASH27B_HAVE_CUDA_SCALAR_FLASHPREFILL=1
            DFLASH27B_HAVE_PASCAL_FLASHPREFILL=1)
        # Best-effort narrowing for CMake versions/toolchains that honor
        # per-source CUDA_ARCHITECTURES; flashprefill_scalar.cu also keeps
        # its kernels visible in newer arch passes so default fatbinary builds
        # do not end up with host launchers but missing device symbols.
        set_source_files_properties(src/flashprefill_scalar.cu PROPERTIES
            CUDA_ARCHITECTURES "60;61;62")
    endif()
endif()

# BSA needs sm_80+ (FA-2 derived kernel uses BF16 tensor cores). Auto-disable
# on legacy arches with a clear message instead of failing the link.
if(DFLASH27B_GPU_BACKEND STREQUAL "hip" AND DFLASH27B_ENABLE_BSA AND NOT DFLASH27B_HIP_SM80_EQUIV)
    message(WARNING
        "DFLASH27B_ENABLE_BSA=ON requested with DFLASH27B_GPU_BACKEND=hip; "
        "disabling BSA because the bundled implementation is CUDA/CUTLASS-only. "
        "Build with -DDFLASH27B_HIP_SM80_EQUIV=ON to enable the rocWMMA HIP BSA path.")
    set(DFLASH27B_ENABLE_BSA OFF)
endif()
if(DFLASH27B_ENABLE_BSA)
    foreach(_arch IN LISTS _dflash_archs)
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
        target_sources(dflash_common PRIVATE src/bsa_launcher_hip.cu)
        set_source_files_properties(src/bsa_launcher_hip.cu PROPERTIES LANGUAGE HIP)
    else()
        target_sources(dflash_common PRIVATE src/bsa_fwd_inst.cu src/bsa_launcher.cu)
    endif()
endif()

target_include_directories(dflash_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${DFLASH27B_SRC_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/src
        # Jinja chat-template engine (lexer/parser/runtime/value/string/caps)
        # vendored under llama.cpp/common for --chat-template-file support.
        # nlohmann_json is already linked PUBLIC (used by jinja/value.cpp).
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/common
)
if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
    target_include_directories(dflash_common PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
if(DFLASH27B_ENABLE_BSA)
    if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
        # HIP path: bsa_launcher_hip.cu only needs DFLASH27B_HAVE_BSA + the
        # hip_compat shim (which the Phase 2 block above already adds).
        target_compile_definitions(dflash_common PRIVATE DFLASH27B_HAVE_BSA=1)
    else()
        target_include_directories(dflash_common PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/bsa_stubs
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/block_sparse_attn/src)
        target_compile_options(dflash_common PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        target_compile_definitions(dflash_common PRIVATE FLASHATTENTION_DISABLE_DROPOUT FLASH_NAMESPACE=flash DFLASH27B_HAVE_BSA=1)
        # MSVC's <cmath> hides POSIX M_* macros (M_LOG2E etc.) unless _USE_MATH_DEFINES
        # is set before any cmath include. BSA's softmax.h relies on M_LOG2E; define
        # globally on the target so it precedes every TU's first <cmath> include.
        if(WIN32)
            target_compile_definitions(dflash_common PRIVATE _USE_MATH_DEFINES)
        endif()
    endif()
endif()

target_link_libraries(dflash_common
    PUBLIC
        ggml
        ${DFLASH27B_GGML_BACKEND_TARGET}
        ggml-base
        nlohmann_json::nlohmann_json
)
# OpenMP for parallel MoE expert compute kernel (saturate memory bandwidth).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dflash_common PRIVATE OpenMP::OpenMP_CXX)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     target_compile_options(dflash_common PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-function>
     )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
     target_compile_options(dflash_common PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
     )
endif()

# PFlash phase-split harness

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/pflash_daemon.cpp")
    add_executable(pflash_daemon test/pflash_daemon.cpp)
    target_include_directories(pflash_daemon PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
    target_link_libraries(pflash_daemon PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
        target_link_libraries(pflash_daemon PRIVATE CUDA::cudart)
    else()
        target_link_libraries(pflash_daemon PRIVATE hip::host)
    endif()
endif()

# ─── Tests (numerics vs oracle) ────────────────────────────────────

option(DFLASH27B_TESTS "Build numerics tests" ON)
option(DFLASH27B_SERVER "Build dflash_server + backend_ipc_daemon" ON)
if(DFLASH27B_TESTS)
    if(DFLASH27B_GPU_BACKEND STREQUAL "cuda" AND _dflash_cuda_min_sm GREATER_EQUAL 80 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_flashprefill_kernels.cpp")
        add_executable(test_flashprefill_kernels test/test_flashprefill_kernels.cpp)
        set_target_properties(test_flashprefill_kernels PROPERTIES CUDA_ARCHITECTURES "${_dflash_archs}")
        target_include_directories(test_flashprefill_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_flashprefill_kernels PRIVATE dflash_common CUDA::cudart)
    elseif(DFLASH27B_GPU_BACKEND STREQUAL "hip" AND DFLASH27B_HIP_SM80_EQUIV AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_flashprefill_kernels.cpp")
        # HIP build of the rocWMMA flashprefill numerics test. The test source is
        # written in CUDA spellings; the hip_compat/ shim maps cuda_runtime.h /
        # cuda_bf16.h and the cuda* APIs onto HIP, exactly as the kernel sources
        # do. Only meaningful when Phase 2 (rocWMMA) is enabled, since that is
        # what provides the launch_*_bf16 symbols in dflash_common. Lets the
        # rocWMMA path be validated on AMD (gfx1100 / gfx1151 / gfx12xx incl.
        # the R9700) instead of CUDA-only.
        add_executable(test_flashprefill_kernels test/test_flashprefill_kernels.cpp)
        set_source_files_properties(test/test_flashprefill_kernels.cpp PROPERTIES LANGUAGE HIP)
        set_target_properties(test_flashprefill_kernels PROPERTIES HIP_ARCHITECTURES "${_dflash_archs}")
        target_include_directories(test_flashprefill_kernels PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
        target_link_libraries(test_flashprefill_kernels PRIVATE dflash_common ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    # RMSNorm HIP kernel vs CPU reference. HIP only, and on ANY HIP build:
    # rms_norm_hip.cu is compiled into dflash_common regardless of SM80_EQUIV
    # (it feeds the HIP chunk-B graph path). Validates the wavefront-generalized
    # reduction against the device's actual wave width.
    if(DFLASH27B_GPU_BACKEND STREQUAL "hip" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_rms_norm_hip.cpp")
        add_executable(test_rms_norm_hip test/test_rms_norm_hip.cpp)
        set_source_files_properties(test/test_rms_norm_hip.cpp PROPERTIES LANGUAGE HIP)
        set_target_properties(test_rms_norm_hip PROPERTIES HIP_ARCHITECTURES "${_dflash_archs}")
        target_include_directories(test_rms_norm_hip PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
        target_link_libraries(test_rms_norm_hip PRIVATE dflash_common ${DFLASH27B_GGML_BACKEND_TARGET})
        add_test(NAME rms_norm_hip COMMAND test_rms_norm_hip)
    endif()
    # GPU draft top-K kernel vs CPU reference (extract_draft_topk). CUDA only:
    # geometric_draft_topk_cuda.cu is compiled into dflash_common solely on the cuda backend.
    if(DFLASH27B_GPU_BACKEND STREQUAL "cuda" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_draft_topk_cuda.cpp")
        add_executable(test_draft_topk_cuda test/test_draft_topk_cuda.cpp)
        target_include_directories(test_draft_topk_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_draft_topk_cuda PRIVATE dflash_common CUDA::cudart)
        add_test(NAME draft_topk_cuda COMMAND test_draft_topk_cuda)
    endif()
    # GPU port of the sample_logits chain vs the CPU reference. CUDA only:
    # geometric_sampler_cuda.cu is compiled into dflash_common solely on the cuda backend.
    if(DFLASH27B_GPU_BACKEND STREQUAL "cuda" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_gpu_sampler_cuda.cpp")
        add_executable(test_gpu_sampler_cuda test/test_gpu_sampler_cuda.cpp)
        target_include_directories(test_gpu_sampler_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_gpu_sampler_cuda PRIVATE dflash_common CUDA::cudart)
        add_test(NAME gpu_sampler_cuda COMMAND test_gpu_sampler_cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_kv_quant.cpp")
        add_executable(test_kv_quant test/test_kv_quant.cpp)
        target_include_directories(test_kv_quant PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_kv_quant PRIVATE dflash_common)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_gguf_mmap.cpp")
        add_executable(test_gguf_mmap test/test_gguf_mmap.cpp)
        target_include_directories(test_gguf_mmap PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_gguf_mmap PRIVATE dflash_common)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_adaptive_keep_ratio.cpp")
        add_executable(test_adaptive_keep_ratio test/test_adaptive_keep_ratio.cpp)
        target_include_directories(test_adaptive_keep_ratio PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_adaptive_keep_ratio PRIVATE dflash_common)
        add_test(NAME adaptive_keep COMMAND test_adaptive_keep_ratio)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_derived_scalars.cpp")
        add_executable(test_derived_scalars test/test_derived_scalars.cpp)
        target_include_directories(test_derived_scalars PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        add_test(NAME derived_scalars COMMAND test_derived_scalars)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_kvflash_placement.cpp")
        add_executable(test_kvflash_placement test/test_kvflash_placement.cpp)
        target_include_directories(test_kvflash_placement PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        add_test(NAME kvflash_placement COMMAND test_kvflash_placement)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_bandit_integration.cpp")
        add_executable(test_bandit_integration test/test_bandit_integration.cpp)
        target_include_directories(test_bandit_integration PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_bandit_integration PRIVATE dflash_common)
        add_test(NAME bandit_integration COMMAND test_bandit_integration)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_anchor_params.cpp")
        add_executable(test_anchor_params test/test_anchor_params.cpp)
        target_include_directories(test_anchor_params PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_anchor_params PRIVATE dflash_common)
        add_test(NAME anchor_params COMMAND test_anchor_params)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_drafter_early_exit_score_range.cpp")
        add_executable(test_drafter_early_exit_score_range
            test/test_drafter_early_exit_score_range.cpp)
        target_include_directories(test_drafter_early_exit_score_range PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
        add_test(NAME test_drafter_early_exit_score_range
            COMMAND test_drafter_early_exit_score_range)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_anchor_transitive.cpp")
        add_executable(test_anchor_transitive
            test/test_anchor_transitive.cpp
            src/qwen3/anchor_scan.cpp)
        target_include_directories(test_anchor_transitive PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen3)
        add_test(NAME test_anchor_transitive
            COMMAND test_anchor_transitive)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_drafter_warm_path_regression.cpp")
        add_executable(test_drafter_warm_path_regression
            test/test_drafter_warm_path_regression.cpp)
        target_include_directories(test_drafter_warm_path_regression PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
        add_test(NAME test_drafter_warm_path_regression
            COMMAND test_drafter_warm_path_regression)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_drafter_tail_capture_guard.cpp")
        # GREEN phase: built with TAIL_GUARD_USE_NEW_FORMULA — must pass after Bug #42 fix.
        add_executable(test_drafter_tail_capture_guard
            test/test_drafter_tail_capture_guard.cpp)
        target_compile_definitions(test_drafter_tail_capture_guard PRIVATE
            TAIL_GUARD_USE_NEW_FORMULA)
        add_test(NAME test_drafter_tail_capture_guard
            COMMAND test_drafter_tail_capture_guard)
        # RED phase binary: same source WITHOUT the fix flag — documents the bug.
        add_executable(test_drafter_tail_capture_guard_red
            test/test_drafter_tail_capture_guard.cpp)
        # No TAIL_GUARD_USE_NEW_FORMULA — uses old (buggy) guard, expected to FAIL.
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_draft_vs_reference.cpp")
        add_executable(test_draft_vs_reference test/test_draft_vs_reference.cpp)
        target_link_libraries(test_draft_vs_reference PRIVATE dflash_common)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_qwen35moe_routing_stats.cpp")
        add_executable(test_qwen35moe_routing_stats test/test_qwen35moe_routing_stats.cpp)
        target_include_directories(test_qwen35moe_routing_stats PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            target_include_directories(test_qwen35moe_routing_stats PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        endif()
        target_link_libraries(test_qwen35moe_routing_stats PRIVATE dflash_common)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_qwen35moe_expert_placement.cpp")
        add_executable(test_qwen35moe_expert_placement test/test_qwen35moe_expert_placement.cpp)
        target_include_directories(test_qwen35moe_expert_placement PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            target_include_directories(test_qwen35moe_expert_placement PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        endif()
        target_link_libraries(test_qwen35moe_expert_placement PRIVATE dflash_common)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_qwen35moe_swap_manager.cpp")
        add_executable(test_qwen35moe_swap_manager test/test_qwen35moe_swap_manager.cpp)
        target_include_directories(test_qwen35moe_swap_manager PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            target_include_directories(test_qwen35moe_swap_manager PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        endif()
        target_link_libraries(test_qwen35moe_swap_manager PRIVATE dflash_common)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_deepseek4_unit.cpp")
        add_executable(test_deepseek4_unit test/test_deepseek4_unit.cpp)
        target_include_directories(test_deepseek4_unit PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
        if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
            target_compile_definitions(test_deepseek4_unit PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
        else()
            target_compile_definitions(test_deepseek4_unit PRIVATE
                DFLASH27B_BACKEND_CUDA=1
                DFLASH27B_CUDA_MIN_SM=${_dflash_cuda_min_sm})
        endif()
        target_link_libraries(test_deepseek4_unit PRIVATE dflash_common ggml ggml-cpu ${DFLASH27B_GGML_BACKEND_TARGET} ggml-base)
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(test_deepseek4_unit PRIVATE CUDA::cudart)
        else()
            target_link_libraries(test_deepseek4_unit PRIVATE hip::host)
        endif()
        add_test(NAME deepseek4_unit COMMAND test_deepseek4_unit)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_ds4_dspark_load.cpp")
        add_executable(test_ds4_dspark_load test/test_ds4_dspark_load.cpp)
        target_include_directories(test_ds4_dspark_load PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
        if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
            target_compile_definitions(test_ds4_dspark_load PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
        else()
            target_compile_definitions(test_ds4_dspark_load PRIVATE DFLASH27B_BACKEND_CUDA=1 GGML_USE_CUDA)
        endif()
        target_link_libraries(test_ds4_dspark_load PRIVATE dflash_common ggml ggml-cpu ${DFLASH27B_GGML_BACKEND_TARGET} ggml-base)
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(test_ds4_dspark_load PRIVATE CUDA::cudart)
        else()
            target_link_libraries(test_ds4_dspark_load PRIVATE hip::host)
        endif()
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_draft.cpp")
        add_executable(smoke_load_draft test/smoke_load_draft.cpp)
        target_include_directories(smoke_load_draft PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_load_draft PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/spike_thin_copy.cpp")
        add_executable(spike_thin_copy test/spike_thin_copy.cpp)
        target_include_directories(spike_thin_copy PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(spike_thin_copy PRIVATE ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_draft_graph.cpp")
        add_executable(smoke_draft_graph test/smoke_draft_graph.cpp)
        target_include_directories(smoke_draft_graph PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_draft_graph PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_qwen3_forward.cpp")
        add_executable(smoke_qwen3_forward test/smoke_qwen3_forward.cpp)
        target_include_directories(smoke_qwen3_forward PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_qwen3_forward PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_vs_oracle.cpp")
        add_executable(test_vs_oracle test/test_vs_oracle.cpp)
        target_include_directories(test_vs_oracle PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_vs_oracle PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_target.cpp")
        add_executable(smoke_load_target test/smoke_load_target.cpp)
        target_include_directories(smoke_load_target PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_load_target PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_target_laguna.cpp")
        add_executable(smoke_load_target_laguna test/smoke_load_target_laguna.cpp)
        target_include_directories(smoke_load_target_laguna PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_load_target_laguna PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_laguna_forward.cpp")
        add_executable(smoke_laguna_forward test/smoke_laguna_forward.cpp)
        target_include_directories(smoke_laguna_forward PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_laguna_forward PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_ttft.cpp")
        add_executable(bench_laguna_ttft test/bench_laguna_ttft.cpp)
        target_include_directories(bench_laguna_ttft PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(bench_laguna_ttft PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_pflash.cpp")
        add_executable(bench_laguna_pflash test/bench_laguna_pflash.cpp)
        target_include_directories(bench_laguna_pflash PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(bench_laguna_pflash PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_generate.cpp")
        add_executable(bench_laguna_generate test/bench_laguna_generate.cpp)
        target_include_directories(bench_laguna_generate PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(bench_laguna_generate PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_spark.cpp")
        add_executable(bench_laguna_spark test/bench_laguna_spark.cpp)
        target_include_directories(bench_laguna_spark PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(bench_laguna_spark PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_moe_stream.cpp" AND DFLASH27B_GPU_BACKEND STREQUAL "cuda")
        add_executable(bench_moe_stream test/bench_moe_stream.cpp)
        target_include_directories(bench_moe_stream PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(bench_moe_stream PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET} CUDA::cudart)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_laguna_daemon.cpp")
        add_executable(test_laguna_daemon test/test_laguna_daemon.cpp)
        target_include_directories(test_laguna_daemon PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_laguna_daemon PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_target_forward.cpp")
        add_executable(smoke_target_forward test/smoke_target_forward.cpp)
        target_include_directories(smoke_target_forward PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_target_forward PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_generate.cpp")
        add_executable(test_generate test/test_generate.cpp)
        target_include_directories(test_generate PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(test_generate PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_kvflash.cpp")
        add_executable(test_kvflash test/test_kvflash.cpp)
        target_include_directories(test_kvflash PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        # Default --qkbench filler: a bundled real-token stream, tiled to the
        # requested length. Keeps the qk gate valid out of the box (uniform-
        # random filler has no Q/K contrast and collapses qk recall to ~LRU).
        target_compile_definitions(test_kvflash PRIVATE
            QKBENCH_FILLER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/qkbench_filler.i32")
        target_link_libraries(test_kvflash PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            target_link_libraries(test_kvflash PRIVATE CUDA::cudart)
        endif()
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_kvflash_qk.cpp")
        # Pure-math unit test (KVFLASH_QK_PURE_ONLY): no ggml, no GPU.
        add_executable(test_kvflash_qk test/test_kvflash_qk.cpp)
        target_include_directories(test_kvflash_qk PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_kvflash_pool_sizing.cpp")
        # Pure unit test for kvflash_pool_from_env: ggml headers only, no ggml link, no GPU.
        add_executable(test_kvflash_pool_sizing test/test_kvflash_pool_sizing.cpp)
        target_include_directories(test_kvflash_pool_sizing PRIVATE
            ${DFLASH27B_SRC_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/include
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/src)
        add_test(NAME kvflash_pool_sizing COMMAND test_kvflash_pool_sizing)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_restore_delta.cpp")
        add_executable(test_restore_delta test/test_restore_delta.cpp)
        target_include_directories(test_restore_delta PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_dflash.cpp")
        add_executable(test_dflash test/test_dflash.cpp)
        target_include_directories(test_dflash PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
            target_compile_definitions(test_dflash PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
        else()
            target_compile_definitions(test_dflash PRIVATE
                DFLASH27B_BACKEND_CUDA=1
                DFLASH27B_CUDA_MIN_SM=${_dflash_cuda_min_sm})
        endif()
        target_link_libraries(test_dflash PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
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

    # Tokenizer test harness (no GPU needed — links static lib for tokenizer + GGUF reader)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_tokenizer_harness.cpp")
        add_executable(test_tokenizer_harness test/test_tokenizer_harness.cpp)
        target_include_directories(test_tokenizer_harness PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
            target_compile_definitions(test_tokenizer_harness PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
        else()
            target_compile_definitions(test_tokenizer_harness PRIVATE
                DFLASH27B_BACKEND_CUDA=1
                DFLASH27B_CUDA_MIN_SM=${_dflash_cuda_min_sm})
        endif()
        target_link_libraries(test_tokenizer_harness PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(test_tokenizer_harness PRIVATE CUDA::cudart)
        else()
            target_link_libraries(test_tokenizer_harness PRIVATE hip::host)
        endif()
    endif()

    # ─── Unit tests (no GPU, no model files) ────────────────────────────
    enable_testing()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_server_unit.cpp")
        add_executable(test_server_unit test/test_server_unit.cpp)
        target_sources(test_server_unit PRIVATE
            src/server/http_server.cpp
            src/server/model_card.cpp
            src/server/prompt_normalize.cpp)
        target_include_directories(test_server_unit PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
            target_compile_definitions(test_server_unit PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
        else()
            target_compile_definitions(test_server_unit PRIVATE
                DFLASH27B_BACKEND_CUDA=1
                DFLASH27B_CUDA_MIN_SM=${_dflash_cuda_min_sm})
        endif()
        if(CURL_FOUND)
            target_compile_definitions(test_server_unit PRIVATE DFLASH_HAS_CURL=1)
        endif()
        if(CURL_FOUND)
            target_link_libraries(test_server_unit PRIVATE CURL::libcurl)
        endif()
        target_link_libraries(test_server_unit PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(test_server_unit PRIVATE CUDA::cudart)
        else()
            target_link_libraries(test_server_unit PRIVATE hip::host)
        endif()
        add_test(NAME server_unit COMMAND test_server_unit)
    endif()

    # 'make check' — builds test targets then runs ctest
    if(TARGET test_server_unit)
        set(_check_deps test_server_unit)
        if(TARGET test_deepseek4_unit)
            list(APPEND _check_deps test_deepseek4_unit)
        endif()
        add_custom_target(check
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            DEPENDS ${_check_deps}
            COMMENT "Building and running unit tests"
        )
    else()
        add_custom_target(check
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMENT "Building and running unit tests (server unit tests skipped — CURL not found)"
        )
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
    # the adapter is only compiled into dflash_common when the CUDA arch list
    # includes an sm_80+ target, so this test can only link there too.
    if(DFLASH27B_GPU_BACKEND STREQUAL "cuda" AND _dflash_has_sm80
       AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_flash_attn_sparse.cpp")
        add_executable(test_flash_attn_sparse test/test_flash_attn_sparse.cpp)
        target_link_libraries(test_flash_attn_sparse PRIVATE dflash_common ggml ggml-cuda ggml-base)
        target_include_directories(test_flash_attn_sparse PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/include
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/src
            ${DFLASH27B_SRC_INCLUDE_DIRS})
    endif()
    endif()
endif()

if(DFLASH27B_SERVER)

    # ─── dflash_server: native C++ HTTP server ─────────────────────────
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/server/server_main.cpp")
        find_package(CURL QUIET)
        if(NOT CURL_FOUND)
            message(WARNING "CURL not found — building dflash_server without passthrough proxy")
        endif()
        add_executable(dflash_server
            src/server/server_main.cpp
            src/server/http_server.cpp
            src/server/model_card.cpp
            src/server/prompt_normalize.cpp
        )
        target_include_directories(dflash_server PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
            target_compile_definitions(dflash_server PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
        else()
            target_compile_definitions(dflash_server PRIVATE
                DFLASH27B_BACKEND_CUDA=1
                DFLASH27B_CUDA_MIN_SM=${_dflash_cuda_min_sm})
        endif()
        if(NOT WIN32)
            target_link_libraries(dflash_server PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET} pthread)
        else()
            target_link_libraries(dflash_server PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET} ws2_32)
        endif()
        if(CURL_FOUND)
            target_compile_definitions(dflash_server PRIVATE DFLASH_HAS_CURL=1)
            target_link_libraries(dflash_server PRIVATE CURL::libcurl)
        endif()
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(dflash_server PRIVATE CUDA::cudart)
        else()
            target_link_libraries(dflash_server PRIVATE hip::host)
        endif()

        # Copy share/status.html next to the binary so it can be found at runtime.
        if(TARGET dflash_server)
            add_custom_command(TARGET dflash_server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:dflash_server>/share"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/share/status.html"
                    "$<TARGET_FILE_DIR:dflash_server>/share/status.html"
                COMMENT "Copying status.html to build/share/"
            )
            install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/share/status.html"
                    DESTINATION share)
        endif()
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/backend_ipc_main.cpp")
        add_executable(backend_ipc_daemon
            src/ipc/backend_ipc_main.cpp
            src/common/pflash_drafter_ipc_daemon.cpp
        )
        target_include_directories(backend_ipc_daemon PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        if(DFLASH27B_GPU_BACKEND STREQUAL "hip")
            target_compile_definitions(backend_ipc_daemon PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP)
        else()
            target_compile_definitions(backend_ipc_daemon PRIVATE
                DFLASH27B_BACKEND_CUDA=1
                DFLASH27B_CUDA_MIN_SM=${_dflash_cuda_min_sm})
        endif()
        if(NOT WIN32)
            target_link_libraries(backend_ipc_daemon PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET} pthread)
        else()
            target_link_libraries(backend_ipc_daemon PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET} ws2_32)
        endif()
        if(DFLASH27B_GPU_BACKEND STREQUAL "cuda")
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(backend_ipc_daemon PRIVATE CUDA::cudart)
        else()
            target_link_libraries(backend_ipc_daemon PRIVATE hip::host)
        endif()
    endif()
endif()
