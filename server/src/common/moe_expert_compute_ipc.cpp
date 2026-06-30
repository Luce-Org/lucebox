// Mixed-backend MoE expert compute IPC.
//
// The parent backend computes router + hot experts. This client sends the
// selected non-local expert work to a backend-local daemon, which computes only
// that routed expert partial and returns one F32 hidden vector.

#include "moe_expert_compute.h"

#include "backend_ipc.h"
#include "io_utils.h"
#include "moe_hybrid_ffn_eval.h"
#include "moe_hybrid_placement.h"
#include "moe_hybrid_storage.h"
#include "ggml_graph_precision.h"
#include "internal.h"
#include "laguna_internal.h"
#include "moe_hybrid_types_impl.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace dflash::common {

namespace {

using MoeExpertClock = std::chrono::steady_clock;

uint64_t moe_expert_compute_elapsed_us(MoeExpertClock::time_point start,
                                 MoeExpertClock::time_point end) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
}

bool moe_expert_compute_profile_enabled() {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_PROFILE");
    return raw && *raw && std::strcmp(raw, "0") != 0 &&
        std::strcmp(raw, "false") != 0 && std::strcmp(raw, "off") != 0;
}

#if !defined(_WIN32)
bool moe_expert_compute_drain_exact_fd(int fd, size_t bytes) {
    std::array<char, 8192> tmp{};
    while (bytes > 0) {
        const size_t chunk = std::min(bytes, tmp.size());
        if (!read_exact_fd(fd, tmp.data(), chunk)) return false;
        bytes -= chunk;
    }
    return true;
}
#endif

bool moe_expert_compute_checked_mul_size(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

struct MoeExpertIpcClientStats {
    uint64_t calls = 0;
    uint64_t payload_bytes = 0;
    uint64_t pack_us = 0;
    uint64_t send_us = 0;
    uint64_t wait_us = 0;
    uint64_t recv_us = 0;

    void maybe_print(bool force) const {
        if (!force && calls > 0 && calls % 128 != 0) return;
        if (calls == 0) return;
        std::fprintf(stderr,
            "[moe-expert-compute-ipc] profile calls=%" PRIu64
            " payload_mib=%.3f pack_ms=%.3f send_ms=%.3f wait_ms=%.3f recv_ms=%.3f\n",
            calls, payload_bytes / 1048576.0,
            pack_us / 1000.0, send_us / 1000.0,
            wait_us / 1000.0, recv_us / 1000.0);
    }
};

struct MoeExpertDaemonStats {
    uint64_t calls = 0;
    uint64_t payload_bytes = 0;
    uint64_t read_us = 0;
    uint64_t unpack_us = 0;
    uint64_t graph_builds = 0;
    uint64_t graph_build_us = 0;
    uint64_t tensor_set_us = 0;
    uint64_t compute_us = 0;
    uint64_t tensor_get_us = 0;
    uint64_t write_us = 0;

    void maybe_print(bool force, const char * scope = "") const {
        if (!force && calls > 0 && calls % 128 != 0) return;
        if (calls == 0) return;
        std::fprintf(stderr,
            "[moe-expert-compute-daemon] profile%s calls=%" PRIu64
            " payload_mib=%.3f builds=%" PRIu64
            " read_ms=%.3f unpack_ms=%.3f build_ms=%.3f set_ms=%.3f"
            " compute_ms=%.3f get_ms=%.3f write_ms=%.3f\n",
            scope, calls, payload_bytes / 1048576.0, graph_builds,
            read_us / 1000.0, unpack_us / 1000.0,
            graph_build_us / 1000.0, tensor_set_us / 1000.0,
            compute_us / 1000.0, tensor_get_us / 1000.0,
            write_us / 1000.0);
    }
};

inline ggml_tensor * moe_expert_apply_scale2(ggml_context * ctx,
                                           ggml_tensor * mm_result,
                                           float scale) {
    return scale == 1.0f ? mm_result : ggml_scale(ctx, mm_result, scale);
}

bool moe_expert_ipc_is_supported_input_type(ggml_type type);

struct CachedBatchedMoeExpertGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor * inp = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * output = nullptr;
    int n_tokens = 0;
    int n_selected = 0;
    ggml_type input_type = GGML_TYPE_F32;

    bool valid() const { return ctx && gf && alloc && output; }

    void free() {
        if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        gf = nullptr;
        inp = nullptr;
        ids = nullptr;
        weights = nullptr;
        output = nullptr;
        n_tokens = 0;
        n_selected = 0;
        input_type = GGML_TYPE_F32;
    }
};

struct RemoteMoeLayerRuntime {
    float gate_scale = 1.0f;
    float up_scale = 1.0f;
    float down_scale = 1.0f;
    float gate_up_scale = 1.0f;
};

struct RemoteMoeRuntime {
    std::string arch;
    int n_layer = 0;
    int n_embd = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff_exp = 0;
    std::vector<RemoteMoeLayerRuntime> layers;
    MoeHybridStorage hybrid;
};

bool build_cached_batched_cold_graph(
    CachedBatchedMoeExpertGraph & out,
    ggml_backend_t backend,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    int n_embd,
    int n_ff_exp,
    int n_selected,
    int n_tokens,
    ggml_type input_type = GGML_TYPE_F32) {

    out.free();
    if (n_embd <= 0 || n_ff_exp <= 0 || n_selected <= 0 || n_tokens <= 0 ||
        !down_tensor || (!gate_up_tensor && (!gate_tensor || !up_tensor)) ||
        !moe_expert_ipc_is_supported_input_type(input_type)) {
        return false;
    }

    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.inp = ggml_new_tensor_2d(out.ctx, input_type, n_embd, n_tokens);
    ggml_set_input(out.inp);
    out.ids = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_selected, n_tokens);
    ggml_set_input(out.ids);
    out.weights = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_selected, n_tokens);
    ggml_set_input(out.weights);

    ggml_tensor * cur_f32 = graph_tensor_f32(out.ctx, out.inp);
    ggml_tensor * cur_3d = ggml_reshape_3d(out.ctx, cur_f32, n_embd, 1, n_tokens);
    ggml_tensor * gu = nullptr;
    if (gate_up_tensor) {
        ggml_tensor * gate_up_e = moe_expert_apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, gate_up_tensor, cur_3d, out.ids),
            gate_up_scale);
        ggml_tensor * gate_e = ggml_view_3d(out.ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2], 0);
        ggml_tensor * up_e = ggml_view_3d(out.ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2],
            (size_t)n_ff_exp * ggml_element_size(gate_up_e));
        gate_e = ggml_cont(out.ctx, gate_e);
        up_e = ggml_cont(out.ctx, up_e);
        gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
    } else {
        ggml_tensor * gate_e = moe_expert_apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, gate_tensor, cur_3d, out.ids),
            gate_scale);
        ggml_tensor * up_e = moe_expert_apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, up_tensor, cur_3d, out.ids),
            up_scale);
        gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
    }

    ggml_tensor * experts = moe_expert_apply_scale2(out.ctx,
        ggml_mul_mat_id(out.ctx, down_tensor, gu, out.ids), down_scale);
    ggml_tensor * w_view = ggml_reshape_3d(out.ctx, out.weights,
                                           1, n_selected, n_tokens);
    experts = ggml_mul(out.ctx, experts, w_view);

    ggml_tensor * sum_shape =
        ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * moe_sum = ggml_repeat_back(out.ctx, experts, sum_shape);
    out.output = ggml_reshape_2d(out.ctx, moe_sum, n_embd, n_tokens);
    out.gf = ggml_new_graph_custom(out.ctx, 4096, false);
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.gf, out.output);
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    out.n_tokens = n_tokens;
    out.n_selected = n_selected;
    out.input_type = input_type;
    return true;
}

bool emit_status(int stream_fd, int32_t status) {
    if (stream_fd < 0) return false;
#if defined(_WIN32)
    stream_emit_fd(stream_fd, status);
    return true;
#else
    return write_exact_fd(stream_fd, &status, sizeof(status));
#endif
}

uint64_t moe_expert_compute_batch_graph_key(int n_tokens, int n_selected) {
    return ((uint64_t)(uint32_t)n_tokens << 32) | (uint32_t)n_selected;
}

uint64_t moe_expert_compute_batch_graph_key(int n_tokens,
                                            int n_selected,
                                            ggml_type input_type) {
    uint64_t key = ((uint64_t)(uint32_t)n_tokens << 32) |
                   ((uint64_t)(uint32_t)n_selected << 8) |
                   (uint64_t)(uint8_t)input_type;
    return key;
}

MoeHybridPlacement invert_moe_hybrid_placement(const MoeHybridPlacement & main) {
    MoeHybridPlacement remote;
    remote.n_layer = main.n_layer;
    remote.n_expert = main.n_expert;
    remote.n_expert_used = main.n_expert_used;
    remote.hot_counts.assign((size_t)main.n_layer, 0);
    remote.hot_expert_ids.assign((size_t)main.n_layer, {});
    remote.total_hot = 0;

    for (int il = 0; il < main.n_layer; ++il) {
        std::vector<uint8_t> is_main_hot((size_t)main.n_expert, 0);
        if ((size_t)il < main.hot_expert_ids.size()) {
            for (int32_t expert : main.hot_expert_ids[(size_t)il]) {
                if (expert >= 0 && expert < main.n_expert) {
                    is_main_hot[(size_t)expert] = 1;
                }
            }
        }
        auto & remote_hot = remote.hot_expert_ids[(size_t)il];
        for (int expert = 0; expert < main.n_expert; ++expert) {
            if (!is_main_hot[(size_t)expert]) {
                remote_hot.push_back((int32_t)expert);
            }
        }
        remote.hot_counts[(size_t)il] = (int)remote_hot.size();
        remote.total_hot += (int)remote_hot.size();
    }
    return remote;
}

bool write_temp_remote_placement(const MoeHybridPlacement & main,
                                 const std::string & work_dir,
                                 std::string & path_out,
                                 std::string * err) {
#if defined(_WIN32)
    (void)main; (void)work_dir; (void)path_out;
    if (err) *err = "remote MoE expert compute IPC is POSIX-only";
    return false;
#else
    std::string dir = work_dir;
    if (dir.empty()) {
        const char * tmp = std::getenv("TMPDIR");
        dir = (tmp && *tmp) ? tmp : "/tmp";
    }
    if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
        if (err) *err = std::string("mkdir failed for remote placement dir: ") +
                        dir + ": " + std::strerror(errno);
        return false;
    }
    struct stat st {};
    if (::stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        if (err) *err = "remote placement dir is not a directory: " + dir;
        return false;
    }
    std::string templ = dir + "/lucebox_moe_cold_placement_XXXXXX";
    std::vector<char> templ_buf(templ.begin(), templ.end());
    templ_buf.push_back('\0');
    int fd = ::mkstemp(templ_buf.data());
    if (fd < 0) {
        if (err) *err = std::string("mkstemp failed: ") + std::strerror(errno);
        return false;
    }
    ::close(fd);
    path_out = templ_buf.data();
    const MoeHybridPlacement remote = invert_moe_hybrid_placement(main);
    if (!remote.save_json(path_out, "moe_remote_expert_compute", err)) {
        ::unlink(path_out.c_str());
        path_out.clear();
        return false;
    }
    return true;
#endif
}

BackendIpcPayloadTransport moe_expert_transport_from_env() {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_TRANSPORT");
    if (!raw || !*raw) {
        return BackendIpcPayloadTransport::Auto;
    }
    BackendIpcPayloadTransport transport = BackendIpcPayloadTransport::Stream;
    if (!parse_backend_ipc_payload_transport(raw, transport)) {
        return BackendIpcPayloadTransport::Auto;
    }
    return transport;
}

int moe_expert_ipc_shared_batch_capacity_from_env() {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY");
    if (!raw || !*raw) return 1024;
    char * end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || value <= 0) return 1024;
    if (value > 4096) return 4096;
    return (int)value;
}

ggml_type moe_expert_ipc_input_type_from_env() {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_DTYPE");
    if (!raw || !*raw || std::strcmp(raw, "f32") == 0 || std::strcmp(raw, "F32") == 0) {
        return GGML_TYPE_F32;
    }
    if (std::strcmp(raw, "f16") == 0 || std::strcmp(raw, "F16") == 0) {
        return GGML_TYPE_F16;
    }
    if (std::strcmp(raw, "bf16") == 0 || std::strcmp(raw, "BF16") == 0) {
        return GGML_TYPE_BF16;
    }
    std::fprintf(stderr,
                 "[moe-expert-compute-ipc] ignoring unsupported "
                 "DFLASH_MOE_EXPERT_COMPUTE_IPC_DTYPE=%s\n",
                 raw);
    return GGML_TYPE_F32;
}

bool moe_expert_ipc_is_supported_input_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
           type == GGML_TYPE_BF16;
}

size_t moe_expert_ipc_input_row_size(ggml_type type, size_t n_embd) {
    if (!moe_expert_ipc_is_supported_input_type(type) || n_embd == 0) {
        return 0;
    }
    return ggml_row_size(type, (int64_t)n_embd);
}

bool moe_expert_convert_input_to_ipc_type(ggml_type type,
                                          const float * src,
                                          int n_tokens,
                                          int n_embd,
                                          std::vector<uint8_t> & scratch,
                                          const void *& data,
                                          size_t & bytes) {
    data = src;
    bytes = 0;
    if (!src || n_tokens <= 0 || n_embd <= 0 ||
        !moe_expert_ipc_is_supported_input_type(type)) {
        return false;
    }
    const size_t elems = (size_t)n_tokens * (size_t)n_embd;
    if (type == GGML_TYPE_F32) {
        bytes = elems * sizeof(float);
        return true;
    }
    bytes = ggml_row_size(type, (int64_t)n_embd) * (size_t)n_tokens;
    scratch.resize(bytes);
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, reinterpret_cast<ggml_fp16_t *>(scratch.data()),
                              (int64_t)elems);
    } else {
        ggml_fp32_to_bf16_row(src, reinterpret_cast<ggml_bf16_t *>(scratch.data()),
                              (int64_t)elems);
    }
    data = scratch.data();
    return true;
}

size_t moe_expert_required_shared_bytes(int n_embd,
                                      int n_expert_used,
                                      int batch_limit,
                                      ggml_type input_type) {
    if (n_embd <= 0 || n_expert_used <= 0 || batch_limit <= 0) return 0;
    size_t input_bytes = 0;
    size_t ids_bytes = 0;
    size_t weights_bytes = 0;
    size_t total = 0;
    size_t input_elems = 0;
    size_t selected_elems = 0;
    if (!moe_expert_compute_checked_mul_size((size_t)batch_limit, (size_t)n_embd, input_elems) ||
        !moe_expert_compute_checked_mul_size((size_t)batch_limit, (size_t)n_expert_used, selected_elems) ||
        !moe_expert_compute_checked_mul_size(
            (size_t)batch_limit,
            moe_expert_ipc_input_row_size(input_type, (size_t)n_embd),
            input_bytes) ||
        !moe_expert_compute_checked_mul_size(selected_elems, sizeof(int32_t), ids_bytes) ||
        !moe_expert_compute_checked_mul_size(selected_elems, sizeof(float), weights_bytes) ||
        !backend_ipc_checked_add_size(input_bytes, ids_bytes, total) ||
        !backend_ipc_checked_add_size(total, weights_bytes, total)) {
        return 0;
    }
    return total;
}

size_t moe_expert_shared_bytes_from_env(size_t required_bytes) {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_SHARED_BYTES");
    if (!raw || !*raw) return required_bytes;
    if (raw[0] == '-') return required_bytes;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' ||
        parsed > (unsigned long long)std::numeric_limits<size_t>::max()) {
        return required_bytes;
    }
    return std::max((size_t)parsed, required_bytes);
}

bool open_expert_mmap(const char * target_path,
                      gguf_context *& gctx,
                      void *& mmap_addr,
                      size_t & file_size,
                      const uint8_t *& file_bytes,
                      size_t & data_start,
                      std::string * err) {
#if defined(_WIN32)
    (void)target_path; (void)gctx; (void)mmap_addr; (void)file_size;
    (void)file_bytes; (void)data_start;
    if (err) *err = "mmap expert loading is POSIX-only";
    return false;
#else
    gguf_init_params gip{};
    gctx = gguf_init_from_file(target_path, gip);
    if (!gctx) {
        if (err) *err = "failed to open GGUF for expert mmap";
        return false;
    }
    int fd = ::open(target_path, O_RDONLY);
    if (fd < 0) {
        if (err) *err = "failed to open target GGUF";
        gguf_free(gctx);
        gctx = nullptr;
        return false;
    }
    struct stat st;
    if (::fstat(fd, &st) < 0) {
        ::close(fd);
        if (err) *err = "fstat failed on target GGUF";
        gguf_free(gctx);
        gctx = nullptr;
        return false;
    }
    file_size = (size_t)st.st_size;
    mmap_addr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mmap_addr == MAP_FAILED) {
        if (err) *err = "mmap failed on target GGUF";
        gguf_free(gctx);
        gctx = nullptr;
        mmap_addr = nullptr;
        return false;
    }
    data_start = gguf_get_data_offset(gctx);
    file_bytes = static_cast<const uint8_t *>(mmap_addr);
    return true;
#endif
}

void close_expert_mmap(gguf_context * gctx, void * mmap_addr, size_t file_size) {
#if !defined(_WIN32)
    if (mmap_addr && mmap_addr != MAP_FAILED) {
        ::munmap(mmap_addr, file_size);
    }
#else
    (void)mmap_addr; (void)file_size;
#endif
    if (gctx) gguf_free(gctx);
}

std::vector<LayerExpertFileData> make_layer_expert_file_data(
        gguf_context * gctx,
        const uint8_t * file_bytes,
        size_t file_size,
        size_t data_start,
        int n_layer) {
    std::vector<LayerExpertFileData> layer_file_data((size_t)n_layer);
    for (int il = 0; il < n_layer; ++il) {
        char name[128];
        auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
            std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
            int64_t tid = gguf_find_tensor(gctx, name);
            if (tid < 0) return {};
            const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
            const size_t sz = gguf_get_tensor_size(gctx, tid);
            if (off + sz > file_size) return {};
            return { file_bytes + off, sz };
        };
        layer_file_data[(size_t)il].gate_exps = find_tensor_data("ffn_gate_exps");
        layer_file_data[(size_t)il].up_exps = find_tensor_data("ffn_up_exps");
        layer_file_data[(size_t)il].down_exps = find_tensor_data("ffn_down_exps");
        layer_file_data[(size_t)il].gate_up_exps = find_tensor_data("ffn_gate_up_exps");
    }
    return layer_file_data;
}

std::string read_gguf_arch(const char * path, std::string * err) {
    gguf_init_params gip{};
    gguf_context * gctx = gguf_init_from_file(path, gip);
    if (!gctx) {
        if (err) *err = "failed to open GGUF for arch detection";
        return {};
    }
    int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id < 0) {
        if (err) *err = "missing general.architecture";
        gguf_free(gctx);
        return {};
    }
    const char * arch = gguf_get_val_str(gctx, arch_id);
    std::string out = arch ? arch : "";
    gguf_free(gctx);
    return out;
}

template <typename WeightsT, typename FreeFn>
bool build_remote_moe_runtime_from_weights(
        ggml_backend_t backend,
        const MoeHybridPlacement & placement,
        gguf_context * gctx,
        const uint8_t * file_bytes,
        size_t file_size,
        size_t data_start,
        const std::string & arch,
        WeightsT & weights,
        FreeFn free_weights,
        RemoteMoeRuntime & out,
        std::string * err) {
    out.arch = arch;
    out.n_layer = weights.n_layer;
    out.n_embd = weights.n_embd;
    out.n_expert = weights.n_expert;
    out.n_expert_used = weights.n_expert_used;
    out.n_ff_exp = weights.n_ff_exp;

    MoeHybridConfig cfg = make_moe_hybrid_config(weights);
    std::vector<MoeLayerDesc> layer_descs((size_t)weights.n_layer);
    out.layers.resize((size_t)weights.n_layer);
    for (int il = 0; il < weights.n_layer; ++il) {
        layer_descs[(size_t)il] = make_moe_layer_desc(weights.layers[(size_t)il]);
        out.layers[(size_t)il].gate_scale = layer_descs[(size_t)il].ffn_gate_exps_s;
        out.layers[(size_t)il].up_scale = layer_descs[(size_t)il].ffn_up_exps_s;
        out.layers[(size_t)il].down_scale = layer_descs[(size_t)il].ffn_down_exps_s;
        out.layers[(size_t)il].gate_up_scale = layer_descs[(size_t)il].ffn_gate_up_exps_s;
    }

    const auto layer_file_data = make_layer_expert_file_data(
        gctx, file_bytes, file_size, data_start, weights.n_layer);
    std::fprintf(stderr,
                 "[moe-expert-compute-daemon] building remote hot storage arch=%s\n",
                 arch.c_str());
    if (!build_moe_hybrid_storage_from_file(cfg, backend, placement, layer_descs,
                                            layer_file_data, out.hybrid, err,
                                            /*cache_slots=*/0,
                                            /*allocate_cold=*/false)) {
        free_weights(weights);
        return false;
    }
    free_weights(weights);
    return true;
}

bool load_remote_moe_runtime(const char * target_path,
                             ggml_backend_t backend,
                             const MoeHybridPlacement & placement,
                             gguf_context * gctx,
                             const uint8_t * file_bytes,
                             size_t file_size,
                             size_t data_start,
                             RemoteMoeRuntime & out,
                             std::string * err) {
    std::string arch = read_gguf_arch(target_path, err);
    if (arch.empty()) return false;

    TargetLoadPlan plan;
    plan.skip_expert_tensors = true;
    plan.load_output = false;
    plan.metadata_only = true;

    if (arch == "qwen35moe") {
        TargetWeights weights;
        if (!load_target_gguf_partial(target_path, backend, plan, weights)) {
            if (err) *err = dflash27b_last_error();
            free_target_weights(weights);
            return false;
        }
        return build_remote_moe_runtime_from_weights(
            backend, placement, gctx, file_bytes, file_size,
            data_start, arch, weights, free_target_weights, out, err);
    }
    if (arch == "laguna") {
        LagunaTargetWeights weights;
        if (!load_target_gguf_laguna_partial(target_path, backend, plan, weights)) {
            if (err) *err = dflash27b_last_error();
            free_laguna_target_weights(weights);
            return false;
        }
        return build_remote_moe_runtime_from_weights(
            backend, placement, gctx, file_bytes, file_size,
            data_start, arch, weights, free_laguna_target_weights, out, err);
    }

    if (err) *err = "unsupported MoE expert compute arch: " + arch;
    return false;
}

bool validate_shared_payload_request(const void * shared_payload,
                                     const void * shared_payload_data,
                                     size_t shared_payload_capacity,
                                     size_t bytes,
                                     uint64_t seq) {
    const auto * header =
        static_cast<const BackendIpcSharedPayloadHeader *>(shared_payload);
    if (!shared_payload || !shared_payload_data || seq == 0 ||
        !backend_ipc_payload_in_bounds(0, bytes, shared_payload_capacity) ||
        header->sequence != seq || header->bytes != (uint64_t)bytes) {
        return false;
    }
    return true;
}

bool commit_shared_payload_response(void * shared_payload,
                                    void * shared_payload_data,
                                    size_t shared_payload_capacity,
                                    size_t bytes,
                                    uint64_t seq) {
    if (!shared_payload || !shared_payload_data || seq == 0 ||
        !backend_ipc_payload_in_bounds(0, bytes, shared_payload_capacity)) {
        return false;
    }
    auto * header = static_cast<BackendIpcSharedPayloadHeader *>(shared_payload);
    header->bytes = (uint64_t)bytes;
    header->sequence = seq;
    return true;
}

int remote_hot_expert_count(const MoeHybridLayerStorage & storage) {
    if (storage.gate_up_hot) return (int)storage.gate_up_hot->ne[2];
    if (storage.gate_hot) return (int)storage.gate_hot->ne[2];
    if (storage.down_hot) return (int)storage.down_hot->ne[2];
    return (int)storage.hot_expert_ids.size();
}

class MoeExpertComputeIpc : public MoeExpertCompute {
public:
    explicit MoeExpertComputeIpc(int n_ff_max)
        : fallback_(make_cpu_moe_expert_compute(n_ff_max)) {}

    ~MoeExpertComputeIpc() override {
        if (profile_) {
            stats_.maybe_print(true);
        }
        if (!placement_path_.empty()) {
#if !defined(_WIN32)
            ::unlink(placement_path_.c_str());
#endif
        }
    }

    bool start(const std::string & bin,
               const std::string & target_path,
               int target_gpu,
               const MoeHybridPlacement & main_placement,
               int n_embd,
               int n_expert_used,
               const std::string & work_dir,
               bool required) {
#if defined(_WIN32)
        (void)bin; (void)target_path; (void)target_gpu; (void)main_placement;
        (void)n_embd; (void)n_expert_used; (void)work_dir; (void)required;
        return false;
#else
        std::string err;
        if (!write_temp_remote_placement(main_placement, work_dir, placement_path_, &err)) {
            std::fprintf(stderr, "[moe-expert-compute-ipc] placement write failed: %s\n",
                         err.c_str());
            return false;
        }

        BackendIpcLaunchConfig launch;
        launch.bin = bin;
        launch.mode = BackendIpcMode::MoeExpertCompute;
        launch.payload_path = target_path;
        launch.work_dir = work_dir;
        launch.payload_transport = moe_expert_transport_from_env();
        input_type_ = moe_expert_ipc_input_type_from_env();
        const int shared_batch_capacity =
            std::max(moe_hybrid_expert_compute_batch_limit(),
                     moe_expert_ipc_shared_batch_capacity_from_env());
        launch.shared_payload_bytes = moe_expert_shared_bytes_from_env(
            moe_expert_required_shared_bytes(
                n_embd, n_expert_used, shared_batch_capacity, GGML_TYPE_F32));
        launch.args.push_back("--target-gpu=" + std::to_string(std::max(0, target_gpu)));
        launch.args.push_back("--placement=" + placement_path_);
        if (!process_.start(launch)) {
            std::fprintf(stderr, "[moe-expert-compute-ipc] backend process start failed%s\n",
                         required ? " (required)" : " (falling back to CPU)");
            return false;
        }
        active_ = true;
        required_ = required;
        n_embd_ = n_embd;
        n_expert_used_ = n_expert_used;
        profile_ = moe_expert_compute_profile_enabled();
        std::printf("[moe-expert-compute-ipc] ready bin=%s gpu=%d input_type=%s work_dir=%s\n",
                    bin.c_str(), target_gpu,
                    ggml_type_name(input_type_), process_.work_dir().c_str());
        return true;
#endif
    }

    bool compute(const MoeExpertLayer & layer,
                 const float * input,
                 const int32_t * ids,
                 const float * weights,
                 int n_cold,
        int n_embd,
        int n_ff,
        float * output) override {
        if (!output || n_embd <= 0) return false;
        if (n_cold <= 0) {
            std::fill(output, output + n_embd, 0.0f);
            return true;
        }
        if (!active_ || layer.layer_idx < 0 || n_embd != n_embd_) {
            if (required_) return false;
            return fallback_->compute(layer, input, ids, weights, n_cold, n_embd, n_ff, output);
        }

        for (int i = 0; i < n_cold; ++i) {
            const int32_t local = ids[i];
            if (local < 0 || (size_t)local >= layer.cold_global_by_local.size()) {
                if (required_) return false;
                return fallback_->compute(layer, input, ids, weights, n_cold, n_embd, n_ff, output);
            }
        }

        if (!compute_remote(layer.layer_idx, input, ids, weights,
                            n_cold, n_embd, output)) {
            active_ = false;
            if (required_) {
                std::fprintf(stderr,
                             "[moe-expert-compute-ipc] required remote compute failed\n");
                return false;
            }
            return fallback_->compute(layer, input, ids, weights, n_cold, n_embd, n_ff, output);
        }
        return true;
    }

    bool compute_batch(const MoeExpertLayer & layer,
                       const float * input,
                       const int32_t * ids,
                       const float * weights,
                       int n_tokens,
                       int n_selected,
                       int n_embd,
                       int n_ff,
                       float * output) override {
        if (!output || n_tokens < 0 || n_selected < 0 || n_embd <= 0) {
            return false;
        }
        if (n_tokens == 0 || n_selected == 0) {
            std::fill(output, output + (size_t)n_tokens * (size_t)n_embd, 0.0f);
            return true;
        }
        if (!active_ || layer.layer_idx < 0 || n_embd != n_embd_) {
            if (required_) return false;
            return fallback_->compute_batch(layer, input, ids, weights,
                                            n_tokens, n_selected,
                                            n_embd, n_ff, output);
        }

        for (int t = 0; t < n_tokens; ++t) {
            for (int i = 0; i < n_selected; ++i) {
                const size_t idx = (size_t)t * (size_t)n_selected + (size_t)i;
                const int32_t local = ids[idx];
                if (local < 0 || (size_t)local >= layer.cold_global_by_local.size()) {
                    if (required_) return false;
                    return fallback_->compute_batch(layer, input, ids, weights,
                                                    n_tokens, n_selected,
                                                    n_embd, n_ff, output);
                }
            }
        }

        if (!compute_remote_batch(layer.layer_idx, input, ids,
                                  weights, n_tokens,
                                  n_selected, n_embd, output)) {
            active_ = false;
            if (required_) {
                std::fprintf(stderr,
                             "[moe-expert-compute-ipc] required remote batch compute failed\n");
                return false;
            }
            return fallback_->compute_batch(layer, input, ids, weights,
                                            n_tokens, n_selected,
                                            n_embd, n_ff, output);
        }
        return true;
    }

    bool healthy() const override {
        return active_;
    }

private:
    bool compute_remote(int layer_idx,
                        const float * input,
                        const int32_t * local_ids,
                        const float * weights,
                        int n_selected,
                        int n_embd,
                        float * output) {
#if defined(_WIN32)
        (void)layer_idx; (void)input; (void)local_ids; (void)weights;
        (void)n_selected; (void)n_embd; (void)output;
        return false;
#else
        FILE * cmd = process_.command_stream();
        const int stream_fd = process_.stream_fd();
        const int payload_fd = process_.payload_fd();
        if (!cmd || stream_fd < 0 || layer_idx < 0 || !input ||
            !local_ids || !weights || n_selected <= 0 ||
            n_selected > n_expert_used_) {
            return false;
        }

        const ggml_type request_input_type = GGML_TYPE_F32;
        const void * ipc_input = nullptr;
        size_t input_bytes = 0;
        if (!moe_expert_convert_input_to_ipc_type(
                request_input_type, input, 1, n_embd, input_scratch_,
                ipc_input, input_bytes)) {
            return false;
        }
        const size_t ids_bytes = (size_t)n_selected * sizeof(int32_t);
        const size_t weights_bytes = (size_t)n_selected * sizeof(float);
        size_t payload_bytes = 0;
        if (!backend_ipc_checked_add_size(input_bytes, ids_bytes, payload_bytes) ||
            !backend_ipc_checked_add_size(payload_bytes, weights_bytes, payload_bytes)) {
            return false;
        }
        const bool use_shared =
            process_.resolved_payload_transport() == BackendIpcPayloadTransport::Shared;
        const auto pack_t0 = MoeExpertClock::now();
        uint64_t seq = 0;
        const BackendIpcPayloadSegment segments[] = {
            {ipc_input, input_bytes},
            {local_ids, ids_bytes},
            {weights, weights_bytes},
        };
        const auto pack_t1 = MoeExpertClock::now();

        const auto send_t0 = MoeExpertClock::now();
        if (use_shared) {
            if (!process_.write_shared_payload_segments(segments, 3, seq)) {
                return false;
            }
            if (request_input_type == GGML_TYPE_F32) {
                std::fprintf(cmd, "compute_local_shared %d %d %zu %" PRIu64 "\n",
                             layer_idx, n_selected, payload_bytes, seq);
            } else {
                std::fprintf(cmd, "compute_local_shared_typed %d %d %d %zu %" PRIu64 "\n",
                             layer_idx, n_selected, (int)request_input_type,
                             payload_bytes, seq);
            }
            std::fflush(cmd);
        } else if (payload_fd >= 0) {
            if (request_input_type == GGML_TYPE_F32) {
                std::fprintf(cmd, "compute_local_pipe %d %d %zu\n",
                             layer_idx, n_selected, payload_bytes);
            } else {
                std::fprintf(cmd, "compute_local_pipe_typed %d %d %d %zu\n",
                             layer_idx, n_selected, (int)request_input_type,
                             payload_bytes);
            }
            std::fflush(cmd);
            if (!write_exact_fd(payload_fd, ipc_input, input_bytes) ||
                !write_exact_fd(payload_fd, local_ids, ids_bytes) ||
                !write_exact_fd(payload_fd, weights, weights_bytes)) {
                return false;
            }
        } else {
            return false;
        }
        const auto send_t1 = MoeExpertClock::now();

        int32_t status = -1;
        const auto wait_t0 = MoeExpertClock::now();
        if (!read_exact_fd(stream_fd, &status, sizeof(status)) || status != 0) {
            std::fprintf(stderr, "[moe-expert-compute-ipc] compute failed status=%d\n", status);
            return false;
        }
        const auto wait_t1 = MoeExpertClock::now();
        const size_t output_bytes = sizeof(float) * (size_t)n_embd;
        const bool ok = use_shared
            ? process_.read_shared_payload(output, output_bytes, seq)
            : read_exact_fd(stream_fd, output, output_bytes);
        const auto recv_t1 = MoeExpertClock::now();
        if (profile_) {
            stats_.calls++;
            stats_.payload_bytes += payload_bytes;
            stats_.pack_us += moe_expert_compute_elapsed_us(pack_t0, pack_t1);
            stats_.send_us += moe_expert_compute_elapsed_us(send_t0, send_t1);
            stats_.wait_us += moe_expert_compute_elapsed_us(wait_t0, wait_t1);
            stats_.recv_us += moe_expert_compute_elapsed_us(wait_t1, recv_t1);
            stats_.maybe_print(false);
        }
        return ok;
#endif
    }

    bool compute_remote_batch(int layer_idx,
                              const float * input,
                              const int32_t * local_ids,
                              const float * weights,
                              int n_tokens,
                              int n_selected,
                              int n_embd,
                              float * output) {
#if defined(_WIN32)
        (void)layer_idx; (void)input; (void)local_ids; (void)weights;
        (void)n_tokens; (void)n_selected; (void)n_embd; (void)output;
        return false;
#else
        FILE * cmd = process_.command_stream();
        const int stream_fd = process_.stream_fd();
        const int payload_fd = process_.payload_fd();
        if (!cmd || stream_fd < 0 || layer_idx < 0 || !input ||
            !local_ids || !weights || n_tokens <= 0 || n_selected <= 0 ||
            n_selected > n_expert_used_) {
            return false;
        }

        const ggml_type request_input_type = input_type_;
        const void * ipc_input = nullptr;
        size_t input_bytes = 0;
        if (!moe_expert_convert_input_to_ipc_type(
                request_input_type, input, n_tokens, n_embd, input_scratch_,
                ipc_input, input_bytes)) {
            return false;
        }
        const size_t ids_bytes =
            (size_t)n_tokens * (size_t)n_selected * sizeof(int32_t);
        const size_t weights_bytes =
            (size_t)n_tokens * (size_t)n_selected * sizeof(float);
        size_t payload_bytes = 0;
        if (!backend_ipc_checked_add_size(input_bytes, ids_bytes, payload_bytes) ||
            !backend_ipc_checked_add_size(payload_bytes, weights_bytes, payload_bytes)) {
            return false;
        }
        const bool use_shared =
            process_.resolved_payload_transport() == BackendIpcPayloadTransport::Shared;
        const auto pack_t0 = MoeExpertClock::now();
        uint64_t seq = 0;
        const BackendIpcPayloadSegment segments[] = {
            {ipc_input, input_bytes},
            {local_ids, ids_bytes},
            {weights, weights_bytes},
        };
        const auto pack_t1 = MoeExpertClock::now();

        const auto send_t0 = MoeExpertClock::now();
        if (use_shared) {
            if (!process_.write_shared_payload_segments(segments, 3, seq)) {
                return false;
            }
            if (request_input_type == GGML_TYPE_F32) {
                std::fprintf(cmd, "compute_batch_local_shared %d %d %d %zu %" PRIu64 "\n",
                             layer_idx, n_tokens, n_selected, payload_bytes, seq);
            } else {
                std::fprintf(cmd, "compute_batch_local_shared_typed %d %d %d %d %zu %" PRIu64 "\n",
                             layer_idx, n_tokens, n_selected, (int)request_input_type,
                             payload_bytes, seq);
            }
            std::fflush(cmd);
        } else if (payload_fd >= 0) {
            if (request_input_type == GGML_TYPE_F32) {
                std::fprintf(cmd, "compute_batch_local_pipe %d %d %d %zu\n",
                             layer_idx, n_tokens, n_selected, payload_bytes);
            } else {
                std::fprintf(cmd, "compute_batch_local_pipe_typed %d %d %d %d %zu\n",
                             layer_idx, n_tokens, n_selected, (int)request_input_type,
                             payload_bytes);
            }
            std::fflush(cmd);
            if (!write_exact_fd(payload_fd, ipc_input, input_bytes) ||
                !write_exact_fd(payload_fd, local_ids, ids_bytes) ||
                !write_exact_fd(payload_fd, weights, weights_bytes)) {
                return false;
            }
        } else {
            return false;
        }
        const auto send_t1 = MoeExpertClock::now();

        int32_t status = -1;
        const auto wait_t0 = MoeExpertClock::now();
        if (!read_exact_fd(stream_fd, &status, sizeof(status)) || status != 0) {
            std::fprintf(stderr, "[moe-expert-compute-ipc] batch compute failed status=%d\n", status);
            return false;
        }
        const auto wait_t1 = MoeExpertClock::now();
        const size_t output_bytes =
            (size_t)n_tokens * (size_t)n_embd * sizeof(float);
        const bool ok = use_shared
            ? process_.read_shared_payload(output, output_bytes, seq)
            : read_exact_fd(stream_fd, output, output_bytes);
        const auto recv_t1 = MoeExpertClock::now();
        if (profile_) {
            stats_.calls++;
            stats_.payload_bytes += payload_bytes;
            stats_.pack_us += moe_expert_compute_elapsed_us(pack_t0, pack_t1);
            stats_.send_us += moe_expert_compute_elapsed_us(send_t0, send_t1);
            stats_.wait_us += moe_expert_compute_elapsed_us(wait_t0, wait_t1);
            stats_.recv_us += moe_expert_compute_elapsed_us(wait_t1, recv_t1);
            stats_.maybe_print(false);
        }
        return ok;
#endif
    }

    BackendIpcProcess process_;
    std::unique_ptr<MoeExpertCompute> fallback_;
    std::string placement_path_;
    std::vector<uint8_t> input_scratch_;
    MoeExpertIpcClientStats stats_;
    int n_embd_ = 0;
    int n_expert_used_ = 0;
    ggml_type input_type_ = GGML_TYPE_F32;
    bool active_ = false;
    bool required_ = false;
    bool profile_ = false;
};

}  // namespace

MoeExpertComputeIpcStartResult make_moe_expert_compute_ipc(
    const std::string & bin,
    const std::string & target_path,
    int target_gpu,
    const MoeHybridPlacement & main_placement,
    int n_embd,
    int n_ff_exp,
    int n_expert_used,
    const std::string & work_dir,
    bool required) {
    MoeExpertComputeIpcStartResult result;
    auto compute = std::make_unique<MoeExpertComputeIpc>(n_ff_exp);
    if (!compute->start(bin, target_path, target_gpu, main_placement,
                        n_embd, n_expert_used, work_dir, required)) {
        if (required) return result;
        result.compute = make_cpu_moe_expert_compute(n_ff_exp);
        return result;
    }
    result.started_remote = true;
    result.compute = std::move(compute);
    return result;
}

int run_moe_expert_compute_ipc_daemon(const char * target_path,
                                const char * placement_path,
                                int target_gpu,
                                int stream_fd,
                                int payload_fd,
                                int shared_payload_fd,
                                size_t shared_payload_bytes) {
#if defined(_WIN32)
    (void)target_path; (void)placement_path; (void)target_gpu;
    (void)stream_fd; (void)payload_fd; (void)shared_payload_fd;
    (void)shared_payload_bytes;
    std::fprintf(stderr, "MoE expert compute IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    setvbuf(stderr, nullptr, _IONBF, 0);
    if (!target_path || !placement_path || stream_fd < 0) {
        std::fprintf(stderr,
            "usage: backend_ipc_daemon --backend-ipc-mode=moe-expert-compute "
            "<target.gguf> --target-gpu=N --placement=PATH --stream-fd=FD\n");
        return 2;
    }

    void * shared_payload = nullptr;
    void * shared_payload_data = nullptr;
    size_t shared_payload_capacity = 0;
    size_t shared_payload_map_bytes = 0;
    if (shared_payload_fd >= 0 || shared_payload_bytes > 0) {
        if (shared_payload_fd < 0 || shared_payload_bytes == 0 ||
            !backend_ipc_shared_payload_map_bytes(shared_payload_bytes,
                                                  shared_payload_map_bytes)) {
            emit_status(stream_fd, -1);
            return 1;
        }
        shared_payload = ::mmap(nullptr, shared_payload_map_bytes,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                shared_payload_fd, 0);
        if (shared_payload == MAP_FAILED) {
            std::fprintf(stderr, "[moe-expert-compute-daemon] shared payload mmap failed\n");
            emit_status(stream_fd, -1);
            return 1;
        }
        shared_payload_data =
            static_cast<char *>(shared_payload) + backend_ipc_shared_payload_header_bytes();
        shared_payload_capacity = shared_payload_bytes;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(std::max(0, target_gpu));
    if (!backend) {
        std::fprintf(stderr, "[moe-expert-compute-daemon] backend init failed gpu=%d\n",
                     target_gpu);
        emit_status(stream_fd, -1);
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
        }
        return 1;
    }

    std::fprintf(stderr,
                 "[moe-expert-compute-daemon] starting target metadata load target=%s gpu=%d\n",
                 target_path, target_gpu);

    MoeHybridPlacement placement;
    std::string err;
    std::fprintf(stderr, "[moe-expert-compute-daemon] loading placement=%s\n", placement_path);
    if (!MoeHybridPlacement::load_json(placement_path, placement, &err)) {
        std::fprintf(stderr, "[moe-expert-compute-daemon] placement load failed: %s\n",
                     err.c_str());
        emit_status(stream_fd, -1);
        ggml_backend_free(backend);
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
        }
        return 1;
    }
    std::fprintf(stderr,
                 "[moe-expert-compute-daemon] placement loaded total_hot=%d\n",
                 placement.total_hot);

    gguf_context * gctx = nullptr;
    void * mmap_addr = nullptr;
    size_t file_size = 0;
    const uint8_t * file_bytes = nullptr;
    size_t data_start = 0;
    std::fprintf(stderr, "[moe-expert-compute-daemon] opening expert mmap\n");
    if (!open_expert_mmap(target_path, gctx, mmap_addr, file_size,
                          file_bytes, data_start, &err)) {
        std::fprintf(stderr, "[moe-expert-compute-daemon] expert mmap failed: %s\n",
                     err.c_str());
        emit_status(stream_fd, -1);
        ggml_backend_free(backend);
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
        }
        return 1;
    }
    std::fprintf(stderr,
                 "[moe-expert-compute-daemon] expert mmap ready file_size=%zu data_start=%zu\n",
                 file_size, data_start);

    RemoteMoeRuntime runtime;
    if (!load_remote_moe_runtime(target_path, backend, placement, gctx, file_bytes,
                                 file_size, data_start, runtime, &err)) {
        std::fprintf(stderr, "[moe-expert-compute-daemon] target runtime load failed: %s\n",
                     err.c_str());
        emit_status(stream_fd, -1);
        close_expert_mmap(gctx, mmap_addr, file_size);
        ggml_backend_free(backend);
        if (shared_payload && shared_payload != MAP_FAILED) {
            ::munmap(shared_payload, shared_payload_map_bytes);
        }
        return 1;
    }
    std::fprintf(stderr,
                 "[moe-expert-compute-daemon] target runtime ready arch=%s layers=%d embd=%d experts=%d used=%d remote_hot=%d\n",
                 runtime.arch.c_str(), runtime.n_layer, runtime.n_embd,
                 runtime.n_expert, runtime.n_expert_used,
                 runtime.hybrid.placement.total_hot);

    close_expert_mmap(gctx, mmap_addr, file_size);

    std::vector<std::vector<CachedFfnGraph>> graphs(
        (size_t)runtime.n_layer,
        std::vector<CachedFfnGraph>((size_t)runtime.n_expert_used + 1));
    std::vector<std::unordered_map<uint64_t, CachedBatchedMoeExpertGraph>> batch_graphs(
        (size_t)runtime.n_layer);
    std::vector<uint8_t> input;
    std::vector<int32_t> local_ids((size_t)std::max(1, runtime.n_expert_used));
    std::vector<float> router_weights((size_t)std::max(1, runtime.n_expert_used));
    std::vector<float> output((size_t)runtime.n_embd);
    std::vector<int32_t> batch_local_ids;
    std::vector<int32_t> payload_ids;
    std::vector<float> payload_weights;
    MoeExpertDaemonStats profile_stats;
    MoeExpertDaemonStats profile_stats_prefill;
    MoeExpertDaemonStats profile_stats_decode;
    const bool profile = moe_expert_compute_profile_enabled();
    int warmup_builds = 0;
    std::fprintf(stderr,
                 "[moe-expert-compute-daemon] ready gpu=%d remote_hot=%d layers=%d warmup_graphs=%d warmup_ms=0.000\n",
                 target_gpu, runtime.hybrid.placement.total_hot, runtime.n_layer, warmup_builds);
    emit_status(stream_fd, 0);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") {
            break;
        }

        int layer_idx = -1;
        int n_selected = 0;
        int n_tokens = 1;
        size_t bytes = 0;
        uint64_t shared_seq = 0;
        bool payload_ok = false;
        bool pipe_payload = false;
        bool shared_payload_cmd = false;
        bool ids_are_local = false;
        ggml_type input_type = GGML_TYPE_F32;
        const auto read_t0 = MoeExpertClock::now();
        if (cmd == "compute_pipe") {
            iss >> layer_idx >> n_selected >> bytes;
            pipe_payload = true;
        } else if (cmd == "compute_shared") {
            iss >> layer_idx >> n_selected >> bytes >> shared_seq;
            shared_payload_cmd = true;
        } else if (cmd == "compute_local_pipe") {
            iss >> layer_idx >> n_selected >> bytes;
            pipe_payload = true;
            ids_are_local = true;
        } else if (cmd == "compute_local_pipe_typed") {
            int input_type_i = (int)GGML_TYPE_F32;
            iss >> layer_idx >> n_selected >> input_type_i >> bytes;
            pipe_payload = true;
            ids_are_local = true;
            input_type = (ggml_type)input_type_i;
        } else if (cmd == "compute_local_shared") {
            iss >> layer_idx >> n_selected >> bytes >> shared_seq;
            shared_payload_cmd = true;
            ids_are_local = true;
        } else if (cmd == "compute_local_shared_typed") {
            int input_type_i = (int)GGML_TYPE_F32;
            iss >> layer_idx >> n_selected >> input_type_i >> bytes >> shared_seq;
            shared_payload_cmd = true;
            ids_are_local = true;
            input_type = (ggml_type)input_type_i;
        } else if (cmd == "compute_batch_pipe") {
            iss >> layer_idx >> n_tokens >> n_selected >> bytes;
            pipe_payload = true;
        } else if (cmd == "compute_batch_shared") {
            iss >> layer_idx >> n_tokens >> n_selected >> bytes >> shared_seq;
            shared_payload_cmd = true;
        } else if (cmd == "compute_batch_local_pipe") {
            iss >> layer_idx >> n_tokens >> n_selected >> bytes;
            pipe_payload = true;
            ids_are_local = true;
        } else if (cmd == "compute_batch_local_pipe_typed") {
            int input_type_i = (int)GGML_TYPE_F32;
            iss >> layer_idx >> n_tokens >> n_selected >> input_type_i >> bytes;
            pipe_payload = true;
            ids_are_local = true;
            input_type = (ggml_type)input_type_i;
        } else if (cmd == "compute_batch_local_shared") {
            iss >> layer_idx >> n_tokens >> n_selected >> bytes >> shared_seq;
            shared_payload_cmd = true;
            ids_are_local = true;
        } else if (cmd == "compute_batch_local_shared_typed") {
            int input_type_i = (int)GGML_TYPE_F32;
            iss >> layer_idx >> n_tokens >> n_selected >> input_type_i >> bytes >> shared_seq;
            shared_payload_cmd = true;
            ids_are_local = true;
            input_type = (ggml_type)input_type_i;
        } else {
            emit_status(stream_fd, -1);
            continue;
        }

        size_t input_elems = 0;
        size_t selected_elems = 0;
        size_t input_bytes = 0;
        size_t ids_bytes = 0;
        size_t weights_bytes = 0;
        size_t expected = 0;
        bool header_ok = false;
        if (n_tokens > 0 && n_selected > 0 &&
            n_selected <= runtime.n_expert_used &&
            layer_idx >= 0 && layer_idx < runtime.n_layer &&
            moe_expert_ipc_is_supported_input_type(input_type)) {
            header_ok =
                moe_expert_compute_checked_mul_size((size_t)n_tokens,
                                              (size_t)runtime.n_embd,
                                              input_elems) &&
                moe_expert_compute_checked_mul_size((size_t)n_tokens,
                                              (size_t)n_selected,
                                              selected_elems) &&
                moe_expert_compute_checked_mul_size((size_t)n_tokens,
                                              moe_expert_ipc_input_row_size(
                                                  input_type,
                                                  (size_t)runtime.n_embd),
                                              input_bytes) &&
                moe_expert_compute_checked_mul_size(selected_elems, sizeof(int32_t),
                                              ids_bytes) &&
                moe_expert_compute_checked_mul_size(selected_elems, sizeof(float),
                                              weights_bytes) &&
                backend_ipc_checked_add_size(input_bytes, ids_bytes, expected) &&
                backend_ipc_checked_add_size(expected, weights_bytes, expected) &&
                bytes == expected;
        }

        if (pipe_payload && header_ok) {
            input.resize(input_bytes);
            payload_ids.resize((size_t)n_tokens * (size_t)n_selected);
            payload_weights.resize((size_t)n_tokens * (size_t)n_selected);
            payload_ok = payload_fd >= 0 &&
                read_exact_fd(payload_fd, input.data(), input_bytes) &&
                read_exact_fd(payload_fd, payload_ids.data(), ids_bytes) &&
                read_exact_fd(payload_fd, payload_weights.data(), weights_bytes);
        } else if (pipe_payload) {
            payload_ok = payload_fd >= 0 &&
                moe_expert_compute_drain_exact_fd(payload_fd, bytes);
        } else if (shared_payload_cmd) {
            payload_ok = header_ok &&
                validate_shared_payload_request(
                    shared_payload, shared_payload_data, shared_payload_capacity,
                    bytes, shared_seq);
        }
        const auto read_t1 = MoeExpertClock::now();

        if (!header_ok || !payload_ok) {
            emit_status(stream_fd, -1);
            continue;
        }

        size_t off = 0;
        if (!shared_payload_cmd) {
            output.resize((size_t)n_tokens * (size_t)runtime.n_embd);
        }
        const uint8_t * input_data = nullptr;
        const int32_t * payload_id_data = nullptr;
        const float * in_weights = nullptr;
        if (pipe_payload) {
            input_data = input.data();
            payload_id_data = payload_ids.data();
            in_weights = payload_weights.data();
        } else {
            const auto * payload_data =
                static_cast<const uint8_t *>(shared_payload_data);
            input_data = payload_data + off;
            off += input_bytes;
            payload_id_data = reinterpret_cast<const int32_t *>(payload_data + off);
            off += ids_bytes;
            in_weights = reinterpret_cast<const float *>(payload_data + off);
        }

        MoeHybridLayerStorage & storage = runtime.hybrid.layers[(size_t)layer_idx];
        const auto unpack_t1 = MoeExpertClock::now();
        const int remote_hot_count = remote_hot_expert_count(storage);

        auto & layer_graphs = graphs[(size_t)layer_idx];
        if ((size_t)n_selected >= layer_graphs.size()) {
            emit_status(stream_fd, -1);
            continue;
        }
        const RemoteMoeLayerRuntime & L = runtime.layers[(size_t)layer_idx];
        if (n_tokens > 1 || input_type != GGML_TYPE_F32) {
            auto & layer_batch_graphs = batch_graphs[(size_t)layer_idx];
            const uint64_t graph_key =
                moe_expert_compute_batch_graph_key(n_tokens, n_selected,
                                                   input_type);
            auto inserted = layer_batch_graphs.try_emplace(graph_key);
            CachedBatchedMoeExpertGraph & graph = inserted.first->second;
            if (!graph.valid() || graph.n_tokens != n_tokens ||
                graph.n_selected != n_selected) {
                const auto build_t0 = MoeExpertClock::now();
                if (!build_cached_batched_cold_graph(
                        graph, backend,
                        storage.gate_hot, storage.up_hot,
                        storage.down_hot, storage.gate_up_hot,
                        L.gate_scale, L.up_scale,
                        L.down_scale, L.gate_up_scale,
                        runtime.n_embd, runtime.n_ff_exp,
                        n_selected, n_tokens, input_type)) {
                    emit_status(stream_fd, -1);
                    continue;
                }
                const auto build_t1 = MoeExpertClock::now();
                if (profile) {
                    MoeExpertDaemonStats & scope_stats =
                        n_tokens > 1 ? profile_stats_prefill : profile_stats_decode;
                    scope_stats.graph_builds++;
                    scope_stats.graph_build_us +=
                        moe_expert_compute_elapsed_us(build_t0, build_t1);
                }
            }

            bool ids_ok = true;
            const int32_t * graph_ids_data = payload_id_data;
            if (!ids_are_local) {
                batch_local_ids.resize((size_t)n_tokens * (size_t)n_selected);
                graph_ids_data = batch_local_ids.data();
            }
            for (int t = 0; t < n_tokens && ids_ok; ++t) {
                for (int i = 0; i < n_selected; ++i) {
                    const size_t idx = (size_t)t * (size_t)n_selected +
                                       (size_t)i;
                    const int32_t id = payload_id_data[idx];
                    const int32_t local = ids_are_local
                        ? id
                        : ((id >= 0 && id < (int)storage.hot_local_by_global.size())
                            ? storage.hot_local_by_global[(size_t)id] : -1);
                    if (local < 0 || local >= remote_hot_count) {
                        ids_ok = false;
                        break;
                    }
                    if (!ids_are_local) {
                        batch_local_ids[idx] = local;
                    }
                }
            }
            if (!ids_ok) {
                emit_status(stream_fd, -1);
                continue;
            }

            const auto set_one_t0 = MoeExpertClock::now();
            ggml_backend_tensor_set(graph.inp, input_data, 0, input_bytes);
            ggml_backend_tensor_set(graph.ids, graph_ids_data, 0, ids_bytes);
            ggml_backend_tensor_set(graph.weights, in_weights, 0, weights_bytes);
            const auto set_one_t1 = MoeExpertClock::now();
            auto st = ggml_backend_graph_compute(backend, graph.gf);
            const auto compute_one_t1 = MoeExpertClock::now();
            if (st != GGML_STATUS_SUCCESS) {
                emit_status(stream_fd, -1);
                continue;
            }
            const size_t output_bytes =
                (size_t)n_tokens * (size_t)runtime.n_embd * sizeof(float);
            void * response_data = shared_payload_cmd
                ? shared_payload_data : static_cast<void *>(output.data());
            ggml_backend_tensor_get(graph.output, response_data, 0,
                                    output_bytes);
            const auto get_one_t1 = MoeExpertClock::now();

            const int32_t status = 0;
            if (shared_payload_cmd &&
                !commit_shared_payload_response(shared_payload, shared_payload_data,
                                                shared_payload_capacity,
                                                output_bytes, shared_seq)) {
                emit_status(stream_fd, -1);
                continue;
            }
            if (!write_exact_fd(stream_fd, &status, sizeof(status)) ||
                (!shared_payload_cmd &&
                 !write_exact_fd(stream_fd, output.data(), output_bytes))) {
                break;
            }
            const auto write_t1 = MoeExpertClock::now();
            if (profile) {
                MoeExpertDaemonStats & scope_stats =
                    n_tokens > 1 ? profile_stats_prefill : profile_stats_decode;
                const uint64_t read_us =
                    moe_expert_compute_elapsed_us(read_t0, read_t1);
                const uint64_t unpack_us =
                    moe_expert_compute_elapsed_us(read_t1, unpack_t1);
                const uint64_t set_us =
                    moe_expert_compute_elapsed_us(set_one_t0, set_one_t1);
                const uint64_t compute_us =
                    moe_expert_compute_elapsed_us(set_one_t1, compute_one_t1);
                const uint64_t get_us =
                    moe_expert_compute_elapsed_us(compute_one_t1, get_one_t1);
                const uint64_t write_us =
                    moe_expert_compute_elapsed_us(get_one_t1, write_t1);
                auto record_stats = [&](MoeExpertDaemonStats & stats) {
                    stats.calls++;
                    stats.payload_bytes += bytes;
                    stats.read_us += read_us;
                    stats.unpack_us += unpack_us;
                    stats.tensor_set_us += set_us;
                    stats.compute_us += compute_us;
                    stats.tensor_get_us += get_us;
                    stats.write_us += write_us;
                };
                record_stats(profile_stats);
                record_stats(scope_stats);
                profile_stats.maybe_print(false);
                scope_stats.maybe_print(n_tokens > 1,
                                        n_tokens > 1 ? "[prefill]" : "[decode]");
            }
            continue;
        }

        CachedFfnGraph & graph = layer_graphs[(size_t)n_selected];
        if (!graph.valid() || graph.n_hot != n_selected) {
            const auto build_t0 = MoeExpertClock::now();
            if (!build_cached_cold_graph(graph, backend,
                                         storage.gate_hot, storage.up_hot,
                                         storage.down_hot, storage.gate_up_hot,
                                         L.gate_scale, L.up_scale,
                                         L.down_scale, L.gate_up_scale,
                                         runtime.n_embd, runtime.n_ff_exp,
                                         n_selected)) {
                emit_status(stream_fd, -1);
                continue;
            }
            const auto build_t1 = MoeExpertClock::now();
            if (profile) {
                profile_stats_decode.graph_builds++;
                profile_stats_decode.graph_build_us +=
                    moe_expert_compute_elapsed_us(build_t0, build_t1);
            }
        }

        bool compute_ok = true;
        uint64_t set_us_accum = 0;
        uint64_t compute_us_accum = 0;
        uint64_t get_us_accum = 0;
        const size_t one_input_bytes = (size_t)runtime.n_embd * sizeof(float);
        const size_t one_ids_bytes = (size_t)n_selected * sizeof(int32_t);
        const size_t one_weights_bytes = (size_t)n_selected * sizeof(float);
        for (int t = 0; t < n_tokens; ++t) {
            bool ids_ok = true;
            const size_t token_off = (size_t)t * (size_t)n_selected;
            const int32_t * token_ids = payload_id_data + token_off;
            const float * token_weights = in_weights + token_off;
            const int32_t * graph_ids_data = token_ids;
            if (!ids_are_local) {
                graph_ids_data = local_ids.data();
            }
            for (int i = 0; i < n_selected; ++i) {
                const int32_t id = token_ids[i];
                const int32_t local = ids_are_local
                    ? id
                    : ((id >= 0 && id < (int)storage.hot_local_by_global.size())
                        ? storage.hot_local_by_global[(size_t)id] : -1);
                if (local < 0 || local >= remote_hot_count) {
                    ids_ok = false;
                    break;
                }
                if (!ids_are_local) {
                    local_ids[(size_t)i] = local;
                }
            }
            if (!ids_ok) {
                compute_ok = false;
                break;
            }
            const auto set_one_t0 = MoeExpertClock::now();
            ggml_backend_tensor_set(graph.inp,
                                    static_cast<const uint8_t *>(input_data) +
                                        (size_t)t * one_input_bytes,
                                    0, one_input_bytes);
            ggml_backend_tensor_set(graph.ids, graph_ids_data, 0,
                                    one_ids_bytes);
            ggml_backend_tensor_set(graph.weights, token_weights, 0,
                                    one_weights_bytes);
            const auto set_one_t1 = MoeExpertClock::now();
            auto st = ggml_backend_graph_compute(backend, graph.gf);
            const auto compute_one_t1 = MoeExpertClock::now();
            if (st != GGML_STATUS_SUCCESS) {
                compute_ok = false;
                break;
            }
            void * response_data = shared_payload_cmd
                ? static_cast<void *>(
                      static_cast<uint8_t *>(shared_payload_data) +
                      (size_t)t * one_input_bytes)
                : static_cast<void *>(
                      output.data() + (size_t)t * (size_t)runtime.n_embd);
            ggml_backend_tensor_get(graph.output, response_data, 0,
                                    one_input_bytes);
            const auto get_one_t1 = MoeExpertClock::now();
            set_us_accum += moe_expert_compute_elapsed_us(set_one_t0, set_one_t1);
            compute_us_accum += moe_expert_compute_elapsed_us(set_one_t1, compute_one_t1);
            get_us_accum += moe_expert_compute_elapsed_us(compute_one_t1, get_one_t1);
        }
        const auto set_t1 = MoeExpertClock::now();
        const auto get_t1 = set_t1;
        if (!compute_ok) {
            emit_status(stream_fd, -1);
            continue;
        }
        const int32_t status = 0;
        if (shared_payload_cmd &&
            !commit_shared_payload_response(shared_payload, shared_payload_data,
                                            shared_payload_capacity,
                                            input_bytes, shared_seq)) {
            emit_status(stream_fd, -1);
            continue;
        }
        if (!write_exact_fd(stream_fd, &status, sizeof(status)) ||
            (!shared_payload_cmd &&
             !write_exact_fd(stream_fd, output.data(), input_bytes))) {
            break;
        }
        const auto write_t1 = MoeExpertClock::now();
        if (profile) {
            const uint64_t read_us =
                moe_expert_compute_elapsed_us(read_t0, read_t1);
            const uint64_t unpack_us =
                moe_expert_compute_elapsed_us(read_t1, unpack_t1);
            const uint64_t write_us =
                moe_expert_compute_elapsed_us(get_t1, write_t1);
            auto record_stats = [&](MoeExpertDaemonStats & stats) {
                stats.calls++;
                stats.payload_bytes += bytes;
                stats.read_us += read_us;
                stats.unpack_us += unpack_us;
                stats.tensor_set_us += set_us_accum;
                stats.compute_us += compute_us_accum;
                stats.tensor_get_us += get_us_accum;
                stats.write_us += write_us;
            };
            record_stats(profile_stats);
            record_stats(profile_stats_decode);
            profile_stats.maybe_print(false);
            profile_stats_decode.maybe_print(false, "[decode]");
        }
    }

    if (profile) {
        profile_stats.maybe_print(true);
        profile_stats_prefill.maybe_print(true, "[prefill]");
        profile_stats_decode.maybe_print(true, "[decode]");
    }
    for (auto & per_layer : graphs) {
        for (auto & graph : per_layer) graph.free();
    }
    for (auto & per_layer : batch_graphs) {
        for (auto & graph : per_layer) graph.second.free();
    }
    ggml_backend_free(backend);
    if (shared_payload && shared_payload != MAP_FAILED) {
        ::munmap(shared_payload, shared_payload_map_bytes);
    }
    return 0;
#endif
}

}  // namespace dflash::common
