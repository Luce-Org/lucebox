#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

#if defined(GGML_USE_HIP)
#ifndef cudaPointerAttributes
#define cudaPointerAttributes    hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaMemoryTypeDevice     hipMemoryTypeDevice
#define cudaMemoryTypeManaged    hipMemoryTypeManaged
#endif
#endif

namespace {

// Commit work runs on a per-thread, per-device non-blocking stream. The legacy
// default stream, synchronous cudaMemcpy and cudaDeviceSynchronize() fail on
// HIP while another thread captures a graph (several models served by one
// process), and the failure also invalidates that capture.
struct CommitStreams {
    cudaStream_t streams[GGML_CUDA_MAX_DEVICES] = {};
    ~CommitStreams() {
        for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
            if (streams[device] && cudaSetDevice(device) == cudaSuccess) {
                (void) cudaStreamDestroy(streams[device]);
            }
        }
    }
};

// Selects `device` and returns its stream for this thread. Callers pass the
// device that owns the tensors, so a worker that last touched another GPU
// still copies and launches on the right one.
cudaStream_t commit_stream(int device) {
    thread_local CommitStreams local;
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES ||
        cudaSetDevice(device) != cudaSuccess) {
        return nullptr;
    }
    cudaStream_t & stream = local.streams[device];
    if (!stream && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        stream = nullptr;
    }
    return stream;
}

bool commit_sync(int device) {
    cudaStream_t stream = commit_stream(device);
    return stream && cudaStreamSynchronize(stream) == cudaSuccess;
}

// Validation reads may run before the commit selects `device`; restore the
// caller's device so a validate-only call has no side effect.
bool read_to_host(int device, void * dst, const void * src, size_t bytes) {
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess) return false;
    cudaStream_t stream = commit_stream(device);
    const bool ok = stream &&
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    return cudaSetDevice(previous) == cudaSuccess && ok;
}

// Per-element bodies. The per-layer kernels and the batched ones below both
// call them, so every element runs the same arithmetic whichever launch
// shape carries it.

__device__ __forceinline__ void gdn_replay_log_commit_element(
        const float * replay_log,
        float * state,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int state_size,
        int n_heads,
        int n_tokens,
        int n_seqs,
        int n_state_slots,
        int replay_log_width,
        int gate_values,
        int sequence,
        int head,
        int element) {
    const int state_elements = state_size * state_size;
    if (sequence >= n_seqs || head >= n_heads ||
        element >= state_elements) {
        return;
    }

    const int slot = active_slot_ids[sequence];
    const int accepted = accepted_prefixes[sequence];
    if (slot < 0 || slot >= n_state_slots ||
        accepted < 0 || accepted > n_tokens) {
        return;
    }

    const int row = element % state_size;
    const int col = element / state_size;
    const size_t state_offset =
        (((size_t) slot*n_heads + head)*state_size + col)*state_size + row;
    float current = state[state_offset];

    for (int token = 0; token < accepted; ++token) {
        const float * transition = replay_log +
            (((size_t) sequence*n_tokens + token)*n_heads + head) *
                replay_log_width;
        const float gate = gate_values == 1
            ? transition[0]
            : transition[row];
        const float key = transition[gate_values + row];
        const float delta =
            transition[gate_values + state_size + col];
        current = fmaf(key, delta, gate * current);
    }

    state[state_offset] = current;
}

// One 256-thread block per 32x8 tile of a 128x128 head state; the block's
// transitions are staged in shared memory first.
__device__ __forceinline__ void gdn_replay_log_commit_128_scalar_tile(
        const float * replay_log,
        float * state,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int n_heads,
        int n_tokens,
        int n_seqs,
        int n_state_slots,
        int sequence,
        int head) {
    constexpr int state_size = 128;
    constexpr int max_tokens = 8;
    constexpr int values_per_token = 1 + 32 + 8;
    if (sequence >= n_seqs || head >= n_heads) return;
    const int slot = active_slot_ids[sequence];
    const int accepted = accepted_prefixes[sequence];
    if (slot < 0 || slot >= n_state_slots ||
        accepted < 0 || accepted > n_tokens || accepted > max_tokens) {
        return;
    }
    const int row_base = (blockIdx.x & 3)*32;
    const int col_base = (blockIdx.x >> 2)*8;
    const int row = row_base + (threadIdx.x & 31);
    const int col = col_base + (threadIdx.x >> 5);
    __shared__ float staged[max_tokens][values_per_token];
    for (int item = threadIdx.x;
         item < accepted*values_per_token;
         item += blockDim.x) {
        const int token = item/values_per_token;
        const int field = item % values_per_token;
        const float * transition = replay_log +
            (((size_t) sequence*n_tokens + token)*n_heads + head) *
                (2*state_size + 1);
        if (field == 0) {
            staged[token][0] = transition[0];
        } else if (field <= 32) {
            staged[token][field] = transition[row_base + field];
        } else {
            staged[token][field] =
                transition[1 + state_size + col_base + field - 33];
        }
    }
    __syncthreads();
    const size_t state_offset =
        (((size_t) slot*n_heads + head)*state_size + col)*state_size + row;
    float current = state[state_offset];
    for (int token = 0; token < accepted; ++token) {
        current = fmaf(
            staged[token][1 + (threadIdx.x & 31)],
            staged[token][33 + (threadIdx.x >> 5)],
            staged[token][0]*current);
    }
    state[state_offset] = current;
}

__device__ __forceinline__ void gdn_conv_replay_log_commit_element(
        const float * conv_input,
        float * conv_state,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int window,
        int channels,
        int n_tokens,
        int n_seqs,
        int n_state_slots,
        int sequence,
        int element) {
    const int count = window * channels;
    if (sequence >= n_seqs || element >= count) return;
    const int slot = active_slot_ids[sequence];
    const int accepted = accepted_prefixes[sequence];
    if (slot < 0 || slot >= n_state_slots ||
        accepted < 0 || accepted > n_tokens) return;
    const int k = element % window;
    const int channel = element / window;
    const size_t source =
        ((size_t) sequence * channels + channel) * (window + n_tokens) +
        accepted + k;
    const size_t destination =
        ((size_t) slot * channels + channel) * window + k;
    conv_state[destination] = conv_input[source];
}

__device__ __forceinline__ void tree_cache_commit_byte(
        uint8_t * cache,
        const int64_t * commit_rows,
        const int32_t * active_slot_ids,
        size_t row_bytes,
        size_t head_stride,
        int n_heads,
        int tree_width,
        int n_seqs,
        int n_cache_rows,
        int scratch_base,
        int scratch_stride,
        int byte,
        int flat,
        int head) {
    if ((size_t) byte >= row_bytes || flat >= tree_width*n_seqs ||
        head >= n_heads) return;
    const int lane = flat / tree_width;
    const int node = flat % tree_width;
    const int slot = active_slot_ids[lane];
    const int64_t destination_row = commit_rows[flat];
    if (slot < 0 || destination_row < 0 ||
        destination_row >= n_cache_rows) return;
    const int64_t source_row =
        (int64_t) scratch_base + (int64_t) slot*scratch_stride + node;
    if (source_row < 0 || source_row >= n_cache_rows) return;
    const size_t source =
        (size_t) head*head_stride + (size_t) source_row*row_bytes + byte;
    const size_t destination =
        (size_t) head*head_stride + (size_t) destination_row*row_bytes + byte;
    cache[destination] = cache[source];
}

__global__ void gdn_replay_log_commit_kernel(
        const float * replay_log,
        float * state,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int state_size,
        int n_heads,
        int n_tokens,
        int n_seqs,
        int n_state_slots,
        int replay_log_width,
        int gate_values) {
    gdn_replay_log_commit_element(
        replay_log, state, accepted_prefixes, active_slot_ids,
        state_size, n_heads, n_tokens, n_seqs, n_state_slots,
        replay_log_width, gate_values,
        blockIdx.z, blockIdx.y, blockIdx.x * blockDim.x + threadIdx.x);
}

__global__ void gdn_replay_log_commit_128_scalar_tile_kernel(
        const float * replay_log,
        float * state,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int n_heads,
        int n_tokens,
        int n_seqs,
        int n_state_slots) {
    gdn_replay_log_commit_128_scalar_tile(
        replay_log, state, accepted_prefixes, active_slot_ids,
        n_heads, n_tokens, n_seqs, n_state_slots, blockIdx.z, blockIdx.y);
}

bool device_pointer(const void * pointer, int & device) {
    if (pointer == nullptr) return false;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice &&
        attributes.type != cudaMemoryTypeManaged) {
        return false;
    }
    device = attributes.device;
    return true;
}

} // namespace

namespace {

__global__ void gdn_conv_replay_log_commit_kernel(
        const float * conv_input,
        float * conv_state,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int window,
        int channels,
        int n_tokens,
        int n_seqs,
        int n_state_slots) {
    gdn_conv_replay_log_commit_element(
        conv_input, conv_state, accepted_prefixes, active_slot_ids,
        window, channels, n_tokens, n_seqs, n_state_slots,
        blockIdx.y, blockIdx.x * blockDim.x + threadIdx.x);
}

__global__ void tree_cache_commit_kernel(
        uint8_t * cache,
        const int64_t * commit_rows,
        const int32_t * active_slot_ids,
        size_t row_bytes,
        size_t head_stride,
        int n_heads,
        int tree_width,
        int n_seqs,
        int n_cache_rows,
        int scratch_base,
        int scratch_stride) {
    tree_cache_commit_byte(
        cache, commit_rows, active_slot_ids, row_bytes, head_stride,
        n_heads, tree_width, n_seqs, n_cache_rows, scratch_base,
        scratch_stride, blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y, blockIdx.z);
}

__global__ void tree_feature_commit_kernel(
        const uint8_t * source,
        uint8_t * destination,
        const int32_t * destination_rows,
        size_t row_bytes,
        int n_rows,
        int destination_capacity) {
    const int byte = blockIdx.x * blockDim.x + threadIdx.x;
    const int source_row = blockIdx.y;
    if ((size_t) byte >= row_bytes || source_row >= n_rows) return;
    const int destination_row = destination_rows[source_row];
    if (destination_row < 0 || destination_row >= destination_capacity) return;
    destination[(size_t) destination_row*row_bytes + byte] =
        source[(size_t) source_row*row_bytes + byte];
}

// Batched launches: one kernel for a run of layers or caches, their pointers
// passed by value in a table (no device-side table to upload). The layer or
// cache comes from blockIdx; the element body is the per-layer kernel's.

constexpr int commit_threads = 256;
constexpr int commit_layers_per_launch = 64;
constexpr int commit_caches_per_launch = 32;
constexpr unsigned commit_max_grid_yz = 65535;

struct gdn_commit_layer_table {
    const float * source[commit_layers_per_launch];  // replay log or conv input
    float * destination[commit_layers_per_launch];   // state or conv state
};

struct tree_cache_table {
    uint8_t * cache[commit_caches_per_launch];
    size_t row_bytes[commit_caches_per_launch];
    size_t head_stride[commit_caches_per_launch];
    int n_heads[commit_caches_per_launch];
};

// blockIdx.z is layer*n_seqs + sequence.
__global__ void gdn_replay_log_commit_layers_kernel(
        const gdn_commit_layer_table layers,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int state_size,
        int n_heads,
        int n_tokens,
        int n_seqs,
        int n_state_slots,
        int replay_log_width,
        int gate_values) {
    const int layer = blockIdx.z / n_seqs;
    gdn_replay_log_commit_element(
        layers.source[layer], layers.destination[layer],
        accepted_prefixes, active_slot_ids,
        state_size, n_heads, n_tokens, n_seqs, n_state_slots,
        replay_log_width, gate_values,
        blockIdx.z % n_seqs, blockIdx.y,
        blockIdx.x * blockDim.x + threadIdx.x);
}

// blockIdx.z is layer*n_seqs + sequence.
__global__ void gdn_replay_log_commit_128_scalar_tile_layers_kernel(
        const gdn_commit_layer_table layers,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int n_heads,
        int n_tokens,
        int n_seqs,
        int n_state_slots) {
    const int layer = blockIdx.z / n_seqs;
    gdn_replay_log_commit_128_scalar_tile(
        layers.source[layer], layers.destination[layer],
        accepted_prefixes, active_slot_ids,
        n_heads, n_tokens, n_seqs, n_state_slots,
        blockIdx.z % n_seqs, blockIdx.y);
}

// blockIdx.y is layer*n_seqs + sequence.
__global__ void gdn_conv_replay_log_commit_layers_kernel(
        const gdn_commit_layer_table layers,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int window,
        int channels,
        int n_tokens,
        int n_seqs,
        int n_state_slots) {
    const int layer = blockIdx.y / n_seqs;
    gdn_conv_replay_log_commit_element(
        layers.source[layer], layers.destination[layer],
        accepted_prefixes, active_slot_ids,
        window, channels, n_tokens, n_seqs, n_state_slots,
        blockIdx.y % n_seqs, blockIdx.x * blockDim.x + threadIdx.x);
}

// blockIdx.z is cache*max_heads + head; heads past a cache's own count exit.
__global__ void tree_cache_commit_caches_kernel(
        const tree_cache_table caches,
        int max_heads,
        const int64_t * commit_rows,
        const int32_t * active_slot_ids,
        int tree_width,
        int n_seqs,
        int n_cache_rows,
        int scratch_base,
        int scratch_stride) {
    const int index = blockIdx.z / max_heads;
    tree_cache_commit_byte(
        caches.cache[index], commit_rows, active_slot_ids,
        caches.row_bytes[index], caches.head_stride[index],
        caches.n_heads[index], tree_width, n_seqs, n_cache_rows,
        scratch_base, scratch_stride,
        blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y,
        blockIdx.z % max_heads);
}

bool same_device_pointer(const void * pointer, int expected_device) {
    int pointer_device = -1;
    return device_pointer(pointer, pointer_device) &&
        pointer_device == expected_device;
}

// Shape and value checks shared by the synchronous commits (which read the
// index tensors back) and the stream-ordered transaction (which gets the
// host's own copy). None of them touches the device.

// cache_rows: the row count every cache shares.
bool tree_cache_shapes_ok(
        ggml_tensor * const * caches,
        int n_caches,
        const ggml_tensor * commit_rows,
        const ggml_tensor * active_slot_ids,
        int tree_scratch_base,
        int tree_scratch_stride,
        int & cache_rows) {
    if (!caches || n_caches <= 0 || !commit_rows || !active_slot_ids ||
        commit_rows->type != GGML_TYPE_I64 ||
        active_slot_ids->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(commit_rows) ||
        !ggml_is_contiguous(active_slot_ids) ||
        commit_rows->ne[0] < 1 || commit_rows->ne[1] < 1 ||
        ggml_nelements(active_slot_ids) != commit_rows->ne[1] ||
        tree_scratch_base < 0 ||
        tree_scratch_stride < commit_rows->ne[0]) return false;
    cache_rows = -1;
    for (int index = 0; index < n_caches; ++index) {
        const ggml_tensor * cache = caches[index];
        if (!cache || !ggml_is_contiguous(cache) || cache->ne[0] < 1 ||
            cache->ne[1] < 1 || cache->ne[2] < 1 || cache->ne[3] != 1 ||
            cache->nb[1] < ggml_row_size(cache->type, cache->ne[0])) return false;
        if (cache_rows < 0) cache_rows = (int) cache->ne[1];
        else if (cache->ne[1] != cache_rows) return false;
    }
    return true;
}

bool tree_cache_values_ok(
        const int64_t * destinations,
        const int32_t * slots,
        int tree_width,
        int n_seqs,
        int cache_rows,
        int tree_scratch_base,
        int tree_scratch_stride) {
    for (int lane = 0; lane < n_seqs; ++lane) {
        const int slot = slots[(size_t) lane];
        if (slot == -1) continue;
        if (slot < 0) return false;
        const int64_t source_end = (int64_t) tree_scratch_base +
            (int64_t) slot*tree_scratch_stride + tree_width;
        if (source_end > cache_rows) return false;
        for (int node = 0; node < tree_width; ++node) {
            const int64_t destination =
                destinations[(size_t) lane*tree_width + node];
            if (destination < -1 || destination >= cache_rows) return false;
        }
    }
    return true;
}

bool feature_shapes_ok(
        const ggml_tensor * source,
        const ggml_tensor * destination,
        const ggml_tensor * destination_rows) {
    return source && destination && destination_rows &&
        source->type == destination->type &&
        source->type == GGML_TYPE_BF16 &&
        destination_rows->type == GGML_TYPE_I32 &&
        ggml_is_contiguous(source) &&
        ggml_is_contiguous(destination) &&
        ggml_is_contiguous(destination_rows) &&
        source->ne[0] == destination->ne[0] && source->ne[2] == 1 &&
        source->ne[3] == 1 && destination->ne[2] == 1 &&
        destination->ne[3] == 1 &&
        ggml_nelements(destination_rows) == source->ne[1] &&
        source->nb[1] == destination->nb[1];
}

bool feature_values_ok(const int32_t * rows, int n_rows, int64_t capacity) {
    for (int index = 0; index < n_rows; ++index) {
        const int row = rows[(size_t) index];
        if (row < -1 || row >= capacity) return false;
    }
    return true;
}

// common_tokens and common_state_slots: what every layer shares.
bool gdn_commit_shapes_ok(
        const ggml_tensor * const * replay_logs,
        ggml_tensor * const * states,
        const ggml_tensor * const * conv_inputs,
        ggml_tensor * const * conv_states,
        int n_layers,
        const ggml_tensor * accepted_prefixes,
        const ggml_tensor * active_slot_ids,
        int & common_tokens,
        int & common_state_slots) {
    if (!replay_logs || !states || !conv_inputs || !conv_states ||
        n_layers <= 0 || !accepted_prefixes || !active_slot_ids ||
        accepted_prefixes->type != GGML_TYPE_I32 ||
        active_slot_ids->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(accepted_prefixes) ||
        !ggml_is_contiguous(active_slot_ids)) return false;
    const int64_t n_seqs = ggml_nelements(accepted_prefixes);
    if (n_seqs < 1 || ggml_nelements(active_slot_ids) != n_seqs) return false;

    common_tokens = -1;
    common_state_slots = -1;
    for (int layer = 0; layer < n_layers; ++layer) {
        const ggml_tensor * replay_log = replay_logs[layer];
        const ggml_tensor * state = states[layer];
        const ggml_tensor * conv_input = conv_inputs[layer];
        const ggml_tensor * conv_state = conv_states[layer];
        if (!replay_log || !state || !conv_input || !conv_state ||
            replay_log->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
            conv_input->type != GGML_TYPE_F32 || conv_state->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(replay_log) || !ggml_is_contiguous(state) ||
            !ggml_is_contiguous(conv_input) || !ggml_is_contiguous(conv_state)) return false;
        const int64_t state_size = state->ne[0];
        const int64_t heads = state->ne[2];
        const int64_t tokens = replay_log->ne[2];
        const int64_t state_slots = state->ne[3];
        if ((state_size != 16 && state_size != 32 &&
             state_size != 64 && state_size != 128) ||
            state->ne[1] != state_size || heads < 1 ||
            replay_log->ne[1] != heads || replay_log->ne[3] != n_seqs ||
            (replay_log->ne[0] != 2*state_size + 1 &&
             replay_log->ne[0] != 3*state_size) || tokens < 1 ||
            conv_state->ne[0] < 1 || conv_state->ne[1] < 1 ||
            conv_state->ne[2] != state_slots || conv_state->ne[3] != 1 ||
            conv_input->ne[0] != conv_state->ne[0] + tokens ||
            conv_input->ne[1] != conv_state->ne[1] ||
            conv_input->ne[2] != n_seqs || conv_input->ne[3] != 1) return false;
        if (common_tokens < 0) {
            common_tokens = (int) tokens;
            common_state_slots = (int) state_slots;
        } else if (tokens != common_tokens || state_slots != common_state_slots) {
            return false;
        }
    }
    return true;
}

bool gdn_commit_values_ok(
        const int32_t * accepted,
        const int32_t * slots,
        int64_t n_seqs,
        int common_tokens,
        int common_state_slots) {
    std::vector<uint8_t> seen((size_t) common_state_slots, 0);
    for (int64_t lane = 0; lane < n_seqs; ++lane) {
        if (accepted[(size_t) lane] < 0 ||
            accepted[(size_t) lane] > common_tokens) return false;
        const int slot = slots[(size_t) lane];
        if (slot == -1) continue;
        if (slot < 0 || slot >= common_state_slots) return false;
        if (seen[(size_t) slot]) return false;
        seen[(size_t) slot] = 1;
    }
    return true;
}

// Launch helpers. `batched` puts a run of caches or same-shape layers into
// one launch; otherwise every cache, and every layer's state then its conv
// window, gets its own launch in argument order, as the per-layer commits do.

void launch_tree_cache_commit(
        ggml_tensor * const * caches,
        int n_caches,
        const ggml_tensor * commit_rows,
        const ggml_tensor * active_slot_ids,
        int cache_rows,
        int tree_scratch_base,
        int tree_scratch_stride,
        bool batched,
        cudaStream_t stream) {
    const int tree_width = (int) commit_rows->ne[0];
    const int n_seqs = (int) commit_rows->ne[1];
    const int n_rows = tree_width*n_seqs;
    int max_heads = 1;
    for (int index = 0; index < n_caches; ++index) {
        max_heads = std::max(max_heads, (int) caches[index]->ne[2]);
    }
    const int per_launch = batched
        ? std::max(1, std::min(commit_caches_per_launch,
                               (int) (commit_max_grid_yz / (unsigned) max_heads)))
        : 1;
    for (int first = 0; first < n_caches; first += per_launch) {
        const int count = std::min(per_launch, n_caches - first);
        tree_cache_table table{};
        size_t row_bytes = 0;
        int heads = 1;
        for (int index = 0; index < count; ++index) {
            const ggml_tensor * cache = caches[first + index];
            table.cache[index] = (uint8_t *) cache->data;
            table.row_bytes[index] = cache->nb[1];
            table.head_stride[index] = cache->nb[2];
            table.n_heads[index] = (int) cache->ne[2];
            row_bytes = std::max(row_bytes, (size_t) cache->nb[1]);
            heads = std::max(heads, (int) cache->ne[2]);
        }
        const dim3 grid(
            (unsigned int) ((row_bytes + commit_threads - 1)/commit_threads),
            (unsigned int) n_rows, (unsigned int) (count*heads));
        tree_cache_commit_caches_kernel<<<grid, commit_threads, 0, stream>>>(
            table, heads,
            (const int64_t *) commit_rows->data,
            (const int32_t *) active_slot_ids->data,
            tree_width, n_seqs, cache_rows,
            tree_scratch_base, tree_scratch_stride);
    }
}

void launch_tree_feature_commit(
        const ggml_tensor * source,
        ggml_tensor * destination,
        const ggml_tensor * destination_rows,
        cudaStream_t stream) {
    const int n_rows = (int) source->ne[1];
    const dim3 grid(
        (unsigned int) ((source->nb[1] + commit_threads - 1)/commit_threads),
        (unsigned int) n_rows, 1);
    tree_feature_commit_kernel<<<grid, commit_threads, 0, stream>>>(
        (const uint8_t *) source->data, (uint8_t *) destination->data,
        (const int32_t *) destination_rows->data,
        source->nb[1], n_rows, (int) destination->ne[1]);
}

// Layers a and b commit with the same kernel arguments apart from pointers.
bool gdn_same_launch_shape(
        const ggml_tensor * const * replay_logs,
        ggml_tensor * const * states,
        ggml_tensor * const * conv_states,
        int a,
        int b) {
    return states[a]->ne[0] == states[b]->ne[0] &&
        states[a]->ne[2] == states[b]->ne[2] &&
        states[a]->ne[3] == states[b]->ne[3] &&
        replay_logs[a]->ne[0] == replay_logs[b]->ne[0] &&
        replay_logs[a]->ne[2] == replay_logs[b]->ne[2] &&
        conv_states[a]->ne[0] == conv_states[b]->ne[0] &&
        conv_states[a]->ne[1] == conv_states[b]->ne[1] &&
        conv_states[a]->ne[2] == conv_states[b]->ne[2];
}

void launch_gdn_commit(
        const ggml_tensor * const * replay_logs,
        ggml_tensor * const * states,
        const ggml_tensor * const * conv_inputs,
        ggml_tensor * const * conv_states,
        int n_layers,
        const ggml_tensor * accepted_prefixes,
        const ggml_tensor * active_slot_ids,
        bool batched,
        cudaStream_t stream) {
    const int n_seqs = (int) ggml_nelements(accepted_prefixes);
    const int per_launch = batched
        ? std::max(1, std::min(commit_layers_per_launch,
                               (int) (commit_max_grid_yz / (unsigned) n_seqs)))
        : 1;
    const int32_t * accepted = (const int32_t *) accepted_prefixes->data;
    const int32_t * slots = (const int32_t *) active_slot_ids->data;
    for (int first = 0; first < n_layers;) {
        int count = 1;
        while (count < per_launch && first + count < n_layers &&
               gdn_same_launch_shape(
                   replay_logs, states, conv_states, first, first + count)) {
            ++count;
        }
        gdn_commit_layer_table state_table{};
        gdn_commit_layer_table conv_table{};
        for (int index = 0; index < count; ++index) {
            state_table.source[index] =
                (const float *) replay_logs[first + index]->data;
            state_table.destination[index] =
                (float *) states[first + index]->data;
            conv_table.source[index] =
                (const float *) conv_inputs[first + index]->data;
            conv_table.destination[index] =
                (float *) conv_states[first + index]->data;
        }

        const ggml_tensor * replay_log = replay_logs[first];
        const ggml_tensor * state = states[first];
        const int state_size = (int) state->ne[0];
        const int heads = (int) state->ne[2];
        const int tokens = (int) replay_log->ne[2];
        const int replay_log_width = (int) replay_log->ne[0];
        const int64_t state_elements = (int64_t) state_size*state_size;
        const dim3 state_grid(
            (unsigned int) ((state_elements + commit_threads - 1)/commit_threads),
            (unsigned int) heads, (unsigned int) (count*n_seqs));
        if (state_size == 128 && replay_log_width == 2*state_size + 1 &&
            tokens <= 8) {
            gdn_replay_log_commit_128_scalar_tile_layers_kernel<<<
                    state_grid, commit_threads, 0, stream>>>(
                state_table, accepted, slots,
                heads, tokens, n_seqs, (int) state->ne[3]);
        } else {
            gdn_replay_log_commit_layers_kernel<<<
                    state_grid, commit_threads, 0, stream>>>(
                state_table, accepted, slots,
                state_size, heads, tokens, n_seqs,
                (int) state->ne[3], replay_log_width,
                replay_log_width == 2*state_size + 1 ? 1 : state_size);
        }

        const ggml_tensor * conv_state = conv_states[first];
        const int conv_elements =
            (int) (conv_state->ne[0]*conv_state->ne[1]);
        const dim3 conv_grid(
            (unsigned int) ((conv_elements + commit_threads - 1)/commit_threads),
            (unsigned int) (count*n_seqs), 1);
        gdn_conv_replay_log_commit_layers_kernel<<<
                conv_grid, commit_threads, 0, stream>>>(
            conv_table, accepted, slots,
            (int) conv_state->ne[0], (int) conv_state->ne[1], tokens,
            n_seqs, (int) conv_state->ne[2]);
        first += count;
    }
}

// Device placement of a transaction's tensors, checked once per set of
// addresses and sizes: they change only when a graph or a cache is rebuilt,
// so a steady decode loop pays no pointer-attribute queries.
struct CommitRange {
    const void * data;
    size_t bytes;
    bool operator==(const CommitRange & other) const {
        return data == other.data && bytes == other.bytes;
    }
};

struct CommitPlacement {
    std::vector<CommitRange> destinations;
    std::vector<CommitRange> sources;
    int device = -1;
    // Every destination apart from the others and from every source, so the
    // batched launches (which run caches and layers concurrently, and every
    // state before every conv window) write what the per-layer order writes.
    bool batch_ok = false;
};

bool ranges_overlap(const CommitRange & a, const CommitRange & b) {
    const char * a_begin = (const char *) a.data;
    const char * b_begin = (const char *) b.data;
    return a_begin < b_begin + b.bytes && b_begin < a_begin + a.bytes;
}

bool destinations_disjoint(
        std::vector<CommitRange> destinations,
        const std::vector<CommitRange> & sources) {
    std::sort(destinations.begin(), destinations.end(),
              [](const CommitRange & a, const CommitRange & b) {
                  return (const char *) a.data < (const char *) b.data;
              });
    for (size_t index = 1; index < destinations.size(); ++index) {
        if (ranges_overlap(destinations[index - 1], destinations[index])) {
            return false;
        }
    }
    for (const CommitRange & source : sources) {
        for (const CommitRange & destination : destinations) {
            if (ranges_overlap(source, destination)) return false;
        }
    }
    return true;
}

// Returns the placement when every range is device memory of `device`, or
// nullptr. Good placements are kept per thread (one scheduler thread per
// engine), keyed by the exact ranges: a few, since rounds alternate between
// the verify graphs of a few widths and lane buckets.
const CommitPlacement * commit_placement(
        int device,
        const std::vector<CommitRange> & destinations,
        const std::vector<CommitRange> & sources) {
    constexpr size_t max_cached = 16;
    thread_local std::vector<CommitPlacement> cached;
    thread_local size_t next_slot = 0;
    for (const CommitPlacement & placement : cached) {
        if (placement.device == device &&
            placement.destinations == destinations &&
            placement.sources == sources) {
            return &placement;
        }
    }
    for (const auto * ranges : {&destinations, &sources}) {
        for (const CommitRange & range : *ranges) {
            if (!same_device_pointer(range.data, device)) return nullptr;
        }
    }
    CommitPlacement placement;
    placement.destinations = destinations;
    placement.sources = sources;
    placement.device = device;
    placement.batch_ok = destinations_disjoint(destinations, sources);
    // Replace the oldest entry once the cache is full.
    if (cached.size() < max_cached) {
        cached.push_back(std::move(placement));
        return &cached.back();
    }
    CommitPlacement & slot = cached[next_slot];
    next_slot = (next_slot + 1) % max_cached;
    slot = std::move(placement);
    return &slot;
}

} // namespace

enum class commit_mode {
    VALIDATE,
    COMMIT,
    COMMIT_PREVALIDATED,
};

static bool gdn_replay_log_commit_many_impl(
        const ggml_tensor * const * replay_logs,
        ggml_tensor * const * states,
        const ggml_tensor * const * conv_inputs,
        ggml_tensor * const * conv_states,
        int n_layers,
        const ggml_tensor * accepted_prefixes,
        const ggml_tensor * active_slot_ids,
        commit_mode mode) {
    const bool prevalidated = mode == commit_mode::COMMIT_PREVALIDATED;
    const bool commit = mode != commit_mode::VALIDATE;
    const bool synchronize = mode == commit_mode::COMMIT;
    int common_tokens = -1;
    int common_state_slots = -1;
    if (!prevalidated &&
        !gdn_commit_shapes_ok(
            replay_logs, states, conv_inputs, conv_states, n_layers,
            accepted_prefixes, active_slot_ids,
            common_tokens, common_state_slots)) return false;

    const int64_t n_seqs = ggml_nelements(accepted_prefixes);
    int device = -1;
    if (!device_pointer(accepted_prefixes->data, device)) return false;

    if (!prevalidated) {
        if (!same_device_pointer(active_slot_ids->data, device)) return false;
        for (int layer = 0; layer < n_layers; ++layer) {
            const void * pointers[] = {
                replay_logs[layer]->data, states[layer]->data,
                conv_inputs[layer]->data, conv_states[layer]->data,
            };
            for (const void * pointer : pointers) {
                if (!same_device_pointer(pointer, device)) return false;
            }
        }
        std::vector<int32_t> accepted((size_t) n_seqs);
        std::vector<int32_t> slots((size_t) n_seqs);
        const size_t map_bytes = (size_t) n_seqs*sizeof(int32_t);
        if (!read_to_host(device, accepted.data(), accepted_prefixes->data, map_bytes) ||
            !read_to_host(device, slots.data(), active_slot_ids->data, map_bytes)) return false;
        if (!gdn_commit_values_ok(
                accepted.data(), slots.data(), n_seqs,
                common_tokens, common_state_slots)) return false;
    }

    if (!commit) return true;
    ggml_cuda_set_device(device);
    // A null stream would be the legacy default stream; never launch on it.
    cudaStream_t stream = commit_stream(device);
    if (!stream) return false;
    constexpr int threads = 256;
    (void) cudaGetLastError();
    for (int layer = 0; layer < n_layers; ++layer) {
        const ggml_tensor * replay_log = replay_logs[layer];
        ggml_tensor * state = states[layer];
        const int state_size = (int) state->ne[0];
        const int heads = (int) state->ne[2];
        const int tokens = (int) replay_log->ne[2];
        const int replay_log_width = (int) replay_log->ne[0];
        const int64_t state_elements = (int64_t) state_size*state_size;
        const dim3 state_grid(
            (unsigned int) ((state_elements + threads - 1)/threads),
            (unsigned int) heads, (unsigned int) n_seqs);
        if (state_size == 128 && replay_log_width == 2*state_size + 1 &&
            tokens <= 8) {
            gdn_replay_log_commit_128_scalar_tile_kernel<<<state_grid, threads, 0, stream>>>(
                (const float *) replay_log->data, (float *) state->data,
                (const int32_t *) accepted_prefixes->data,
                (const int32_t *) active_slot_ids->data,
                heads, tokens, (int) n_seqs, (int) state->ne[3]);
        } else {
            gdn_replay_log_commit_kernel<<<state_grid, threads, 0, stream>>>(
                (const float *) replay_log->data, (float *) state->data,
                (const int32_t *) accepted_prefixes->data,
                (const int32_t *) active_slot_ids->data,
                state_size, heads, tokens, (int) n_seqs,
                (int) state->ne[3], replay_log_width,
                replay_log_width == 2*state_size + 1 ? 1 : state_size);
        }

        const ggml_tensor * conv_input = conv_inputs[layer];
        ggml_tensor * conv_state = conv_states[layer];
        const int conv_elements =
            (int) (conv_state->ne[0]*conv_state->ne[1]);
        const dim3 conv_grid(
            (unsigned int) ((conv_elements + threads - 1)/threads),
            (unsigned int) n_seqs, 1);
        gdn_conv_replay_log_commit_kernel<<<conv_grid, threads, 0, stream>>>(
            (const float *) conv_input->data, (float *) conv_state->data,
            (const int32_t *) accepted_prefixes->data,
            (const int32_t *) active_slot_ids->data,
            (int) conv_state->ne[0], (int) conv_state->ne[1], tokens,
            (int) n_seqs, (int) conv_state->ne[2]);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    return !synchronize || commit_sync(device);
}

extern "C" bool ggml_backend_cuda_gdn_replay_log_commit_many(
        const ggml_tensor * const * replay_logs,
        ggml_tensor * const * states,
        const ggml_tensor * const * conv_inputs,
        ggml_tensor * const * conv_states,
        int n_layers,
        const ggml_tensor * accepted_prefixes,
        const ggml_tensor * active_slot_ids) {
    if (!gdn_replay_log_commit_many_impl(
            replay_logs, states, conv_inputs, conv_states, n_layers,
            accepted_prefixes, active_slot_ids, commit_mode::VALIDATE)) {
        return false;
    }
    if (!gdn_replay_log_commit_many_impl(
            replay_logs, states, conv_inputs, conv_states, n_layers,
            accepted_prefixes, active_slot_ids, commit_mode::COMMIT)) {
        GGML_ABORT("recurrent commit failed after mutation began");
    }
    return true;
}

static bool tree_cache_commit_many_impl(
        ggml_tensor * const * caches,
        int n_caches,
        const ggml_tensor * commit_rows,
        const ggml_tensor * active_slot_ids,
        int tree_scratch_base,
        int tree_scratch_stride,
        commit_mode mode) {
    const bool prevalidated = mode == commit_mode::COMMIT_PREVALIDATED;
    const bool commit = mode != commit_mode::VALIDATE;
    const bool synchronize = mode == commit_mode::COMMIT;
    int cache_rows = prevalidated ? (int) caches[0]->ne[1] : -1;
    if (!prevalidated &&
        !tree_cache_shapes_ok(
            caches, n_caches, commit_rows, active_slot_ids,
            tree_scratch_base, tree_scratch_stride, cache_rows)) return false;
    const int tree_width = (int) commit_rows->ne[0];
    const int n_seqs = (int) commit_rows->ne[1];
    const int n_rows = tree_width*n_seqs;
    int device = -1;
    if (!device_pointer(commit_rows->data, device)) return false;

    if (!prevalidated) {
        if (!same_device_pointer(active_slot_ids->data, device)) return false;
        for (int index = 0; index < n_caches; ++index) {
            if (!same_device_pointer(caches[index]->data, device)) return false;
        }
        std::vector<int64_t> destinations((size_t) n_rows);
        std::vector<int32_t> slots((size_t) n_seqs);
        if (!read_to_host(device, destinations.data(), commit_rows->data,
                       destinations.size()*sizeof(int64_t)) ||
            !read_to_host(device, slots.data(), active_slot_ids->data,
                       slots.size()*sizeof(int32_t))) return false;
        if (!tree_cache_values_ok(
                destinations.data(), slots.data(), tree_width, n_seqs,
                cache_rows, tree_scratch_base, tree_scratch_stride)) return false;
    }

    if (!commit) return true;
    ggml_cuda_set_device(device);
    // A null stream would be the legacy default stream; never launch on it.
    cudaStream_t stream = commit_stream(device);
    if (!stream) return false;
    constexpr int threads = 256;
    (void) cudaGetLastError();
    for (int index = 0; index < n_caches; ++index) {
        ggml_tensor * cache = caches[index];
        const dim3 grid(
            (unsigned int) ((cache->nb[1] + threads - 1)/threads),
            (unsigned int) n_rows, (unsigned int) cache->ne[2]);
        tree_cache_commit_kernel<<<grid, threads, 0, stream>>>(
            (uint8_t *) cache->data,
            (const int64_t *) commit_rows->data,
            (const int32_t *) active_slot_ids->data,
            cache->nb[1], cache->nb[2], (int) cache->ne[2],
            tree_width, n_seqs, cache_rows,
            tree_scratch_base, tree_scratch_stride);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    return !synchronize || commit_sync(device);
}

extern "C" bool ggml_backend_cuda_tree_cache_commit_many(
        ggml_tensor * const * caches,
        int n_caches,
        const ggml_tensor * commit_rows,
        const ggml_tensor * active_slot_ids,
        int tree_scratch_base,
        int tree_scratch_stride) {
    if (!tree_cache_commit_many_impl(
            caches, n_caches, commit_rows, active_slot_ids,
            tree_scratch_base, tree_scratch_stride, commit_mode::VALIDATE)) {
        return false;
    }
    if (!tree_cache_commit_many_impl(
            caches, n_caches, commit_rows, active_slot_ids,
            tree_scratch_base, tree_scratch_stride, commit_mode::COMMIT)) {
        GGML_ABORT("K/V commit failed after mutation began");
    }
    return true;
}

static bool tree_feature_commit_impl(
        const ggml_tensor * source,
        ggml_tensor * destination,
        const ggml_tensor * destination_rows,
        commit_mode mode) {
    const bool prevalidated = mode == commit_mode::COMMIT_PREVALIDATED;
    const bool commit = mode != commit_mode::VALIDATE;
    const bool synchronize = mode == commit_mode::COMMIT;
    if (!prevalidated &&
        !feature_shapes_ok(source, destination, destination_rows)) return false;
    int device = -1;
    if (!device_pointer(source->data, device)) return false;
    const int n_rows = (int) source->ne[1];
    if (!prevalidated) {
        if (!same_device_pointer(destination->data, device) ||
            !same_device_pointer(destination_rows->data, device)) return false;
        std::vector<int32_t> rows((size_t) n_rows);
        if (!read_to_host(device, rows.data(), destination_rows->data,
                       rows.size()*sizeof(int32_t))) return false;
        if (!feature_values_ok(rows.data(), n_rows, destination->ne[1])) return false;
    }
    if (!commit) return true;
    ggml_cuda_set_device(device);
    // A null stream would be the legacy default stream; never launch on it.
    cudaStream_t stream = commit_stream(device);
    if (!stream) return false;
    (void) cudaGetLastError();
    launch_tree_feature_commit(source, destination, destination_rows, stream);
    if (cudaGetLastError() != cudaSuccess) return false;
    return !synchronize || commit_sync(device);
}

extern "C" bool ggml_backend_cuda_tree_feature_commit(
        const ggml_tensor * source,
        ggml_tensor * destination,
        const ggml_tensor * destination_rows) {
    if (!tree_feature_commit_impl(
            source, destination, destination_rows, commit_mode::VALIDATE)) {
        return false;
    }
    if (!tree_feature_commit_impl(
            source, destination, destination_rows, commit_mode::COMMIT)) {
        GGML_ABORT("feature commit failed after mutation began");
    }
    return true;
}

extern "C" bool ggml_backend_cuda_tree_commit_transaction(
        ggml_tensor * const * caches,
        int n_caches,
        const ggml_tensor * feature_source,
        ggml_tensor * feature_destination,
        const ggml_tensor * feature_destination_rows,
        const ggml_tensor * const * replay_logs,
        ggml_tensor * const * states,
        const ggml_tensor * const * conv_inputs,
        ggml_tensor * const * conv_states,
        int n_layers,
        const ggml_tensor * commit_rows,
        const ggml_tensor * accepted_prefixes,
        const ggml_tensor * active_slot_ids,
        int tree_scratch_base,
        int tree_scratch_stride) {
    if (!tree_cache_commit_many_impl(
            caches, n_caches, commit_rows, active_slot_ids,
            tree_scratch_base, tree_scratch_stride,
            commit_mode::VALIDATE)) {
        std::fprintf(stderr, "[gdn-transaction] K/V preflight failed\n");
        return false;
    }
    if (!tree_feature_commit_impl(
            feature_source, feature_destination,
            feature_destination_rows, commit_mode::VALIDATE)) {
        std::fprintf(stderr, "[gdn-transaction] feature preflight failed\n");
        return false;
    }
    if (!gdn_replay_log_commit_many_impl(
            replay_logs, states, conv_inputs, conv_states, n_layers,
            accepted_prefixes, active_slot_ids, commit_mode::VALIDATE)) {
        std::fprintf(stderr, "[gdn-transaction] recurrent preflight failed\n");
        return false;
    }

    int transaction_device = -1;
    if (!device_pointer(active_slot_ids->data, transaction_device) ||
        !same_device_pointer(feature_source->data, transaction_device)) {
        std::fprintf(stderr, "[gdn-transaction] mixed-device arguments\n");
        return false;
    }
    // No recoverable error may escape after the first destination changes.
    // Returning false here would let the caller run fallback from a mixed
    // pre-commit/post-commit state.
    if (!tree_cache_commit_many_impl(
            caches, n_caches, commit_rows, active_slot_ids,
            tree_scratch_base, tree_scratch_stride,
            commit_mode::COMMIT_PREVALIDATED)) {
        GGML_ABORT("tree commit failed after K/V mutation began");
    }
    if (!tree_feature_commit_impl(
            feature_source, feature_destination,
            feature_destination_rows,
            commit_mode::COMMIT_PREVALIDATED)) {
        GGML_ABORT("tree commit failed after feature mutation began");
    }
    if (!gdn_replay_log_commit_many_impl(
            replay_logs, states, conv_inputs, conv_states, n_layers,
            accepted_prefixes, active_slot_ids,
            commit_mode::COMMIT_PREVALIDATED)) {
        GGML_ABORT("tree commit failed after recurrent mutation began");
    }
    if (!commit_sync(transaction_device)) {
        GGML_ABORT("tree commit failed during final synchronization");
    }
    return true;
}

extern "C" bool ggml_backend_cuda_tree_commit_transaction_async(
        ggml_backend_t backend,
        ggml_tensor * const * caches,
        int n_caches,
        const ggml_tensor * feature_source,
        ggml_tensor * feature_destination,
        const ggml_tensor * feature_destination_rows,
        const ggml_tensor * const * replay_logs,
        ggml_tensor * const * states,
        const ggml_tensor * const * conv_inputs,
        ggml_tensor * const * conv_states,
        int n_layers,
        const ggml_tensor * commit_rows,
        const ggml_tensor * accepted_prefixes,
        const ggml_tensor * active_slot_ids,
        int tree_scratch_base,
        int tree_scratch_stride,
        const struct ggml_cuda_tree_commit_host_maps * host) {
    if (!backend || !ggml_backend_is_cuda(backend) || !host ||
        !host->commit_rows || !host->accepted_prefixes ||
        !host->active_slot_ids || !host->feature_rows) {
        std::fprintf(stderr, "[gdn-transaction] missing backend or host maps\n");
        return false;
    }

    // The same checks as the synchronous transaction, on the host's copy of
    // the index tensors instead of a readback of them.
    int cache_rows = -1;
    if (!tree_cache_shapes_ok(
            caches, n_caches, commit_rows, active_slot_ids,
            tree_scratch_base, tree_scratch_stride, cache_rows) ||
        !tree_cache_values_ok(
            host->commit_rows, host->active_slot_ids,
            (int) commit_rows->ne[0], (int) commit_rows->ne[1], cache_rows,
            tree_scratch_base, tree_scratch_stride)) {
        std::fprintf(stderr, "[gdn-transaction] K/V preflight failed\n");
        return false;
    }
    if (!feature_shapes_ok(
            feature_source, feature_destination, feature_destination_rows) ||
        !feature_values_ok(
            host->feature_rows, (int) feature_source->ne[1],
            feature_destination->ne[1])) {
        std::fprintf(stderr, "[gdn-transaction] feature preflight failed\n");
        return false;
    }
    int common_tokens = -1;
    int common_state_slots = -1;
    if (!gdn_commit_shapes_ok(
            replay_logs, states, conv_inputs, conv_states, n_layers,
            accepted_prefixes, active_slot_ids,
            common_tokens, common_state_slots) ||
        !gdn_commit_values_ok(
            host->accepted_prefixes, host->active_slot_ids,
            ggml_nelements(accepted_prefixes),
            common_tokens, common_state_slots)) {
        std::fprintf(stderr, "[gdn-transaction] recurrent preflight failed\n");
        return false;
    }

    ggml_backend_cuda_context * ctx =
        (ggml_backend_cuda_context *) backend->context;
    thread_local std::vector<CommitRange> destinations;
    thread_local std::vector<CommitRange> sources;
    destinations.clear();
    sources.clear();
    for (int index = 0; index < n_caches; ++index) {
        destinations.push_back({caches[index]->data, ggml_nbytes(caches[index])});
    }
    destinations.push_back(
        {feature_destination->data, ggml_nbytes(feature_destination)});
    for (int layer = 0; layer < n_layers; ++layer) {
        destinations.push_back({states[layer]->data, ggml_nbytes(states[layer])});
        destinations.push_back(
            {conv_states[layer]->data, ggml_nbytes(conv_states[layer])});
        sources.push_back(
            {replay_logs[layer]->data, ggml_nbytes(replay_logs[layer])});
        sources.push_back(
            {conv_inputs[layer]->data, ggml_nbytes(conv_inputs[layer])});
    }
    for (const ggml_tensor * index : {
             feature_source, feature_destination_rows, commit_rows,
             accepted_prefixes, active_slot_ids}) {
        sources.push_back({index->data, ggml_nbytes(index)});
    }
    const CommitPlacement * placement =
        commit_placement(ctx->device, destinations, sources);
    if (!placement) {
        std::fprintf(stderr, "[gdn-transaction] mixed-device arguments\n");
        return false;
    }

    // No recoverable error may escape after the first launch: the caller
    // could not run its fallback from a half-committed state.
    cudaStream_t stream = ggml_cuda_enter_stream(ctx);
    (void) cudaGetLastError();
    launch_tree_cache_commit(
        caches, n_caches, commit_rows, active_slot_ids, cache_rows,
        tree_scratch_base, tree_scratch_stride, placement->batch_ok, stream);
    if (cudaGetLastError() != cudaSuccess) {
        GGML_ABORT("tree commit failed after K/V mutation began");
    }
    launch_tree_feature_commit(
        feature_source, feature_destination, feature_destination_rows, stream);
    if (cudaGetLastError() != cudaSuccess) {
        GGML_ABORT("tree commit failed after feature mutation began");
    }
    launch_gdn_commit(
        replay_logs, states, conv_inputs, conv_states, n_layers,
        accepted_prefixes, active_slot_ids, placement->batch_ok, stream);
    if (cudaGetLastError() != cudaSuccess) {
        GGML_ABORT("tree commit failed after recurrent mutation began");
    }
    return true;
}
