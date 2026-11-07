#include "common.cuh"
#include "ggml-cuda.h"

#include <cstddef>
#include <cstdio>
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
    const int sequence = blockIdx.z;
    const int head = blockIdx.y;
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void gdn_replay_log_commit_128_scalar_tile_kernel(
        const float * replay_log,
        float * state,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int n_heads,
        int n_tokens,
        int n_seqs,
        int n_state_slots) {
    constexpr int state_size = 128;
    constexpr int max_tokens = 8;
    constexpr int values_per_token = 1 + 32 + 8;
    const int sequence = blockIdx.z;
    const int head = blockIdx.y;
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
    const int sequence = blockIdx.y;
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int byte = blockIdx.x * blockDim.x + threadIdx.x;
    const int flat = blockIdx.y;
    const int head = blockIdx.z;
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

bool same_device_pointer(const void * pointer, int expected_device) {
    int pointer_device = -1;
    return device_pointer(pointer, pointer_device) &&
        pointer_device == expected_device;
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
    if (!prevalidated &&
        (!replay_logs || !states || !conv_inputs || !conv_states ||
         n_layers <= 0 || !accepted_prefixes || !active_slot_ids ||
         accepted_prefixes->type != GGML_TYPE_I32 ||
         active_slot_ids->type != GGML_TYPE_I32 ||
         !ggml_is_contiguous(accepted_prefixes) ||
         !ggml_is_contiguous(active_slot_ids))) return false;

    const int64_t n_seqs = ggml_nelements(accepted_prefixes);
    if (!prevalidated &&
        (n_seqs < 1 || ggml_nelements(active_slot_ids) != n_seqs)) return false;
    int device = -1;
    if (!device_pointer(accepted_prefixes->data, device)) return false;

    if (!prevalidated) {
        if (!same_device_pointer(active_slot_ids->data, device)) return false;
        std::vector<int32_t> accepted((size_t) n_seqs);
        std::vector<int32_t> slots((size_t) n_seqs);
        const size_t map_bytes = (size_t) n_seqs*sizeof(int32_t);
        if (!read_to_host(device, accepted.data(), accepted_prefixes->data, map_bytes) ||
            !read_to_host(device, slots.data(), active_slot_ids->data, map_bytes)) return false;

        int common_tokens = -1;
        int common_state_slots = -1;
        std::vector<uint8_t> seen;
        for (int layer = 0; layer < n_layers; ++layer) {
            const ggml_tensor * replay_log = replay_logs[layer];
            ggml_tensor * state = states[layer];
            const ggml_tensor * conv_input = conv_inputs[layer];
            ggml_tensor * conv_state = conv_states[layer];
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
                seen.assign((size_t) state_slots, 0);
            } else if (tokens != common_tokens || state_slots != common_state_slots) {
                return false;
            }
            const void * pointers[] = {
                replay_log->data, state->data, conv_input->data, conv_state->data,
            };
            for (const void * pointer : pointers) {
                if (!same_device_pointer(pointer, device)) return false;
            }
        }
        for (int64_t lane = 0; lane < n_seqs; ++lane) {
            if (accepted[(size_t) lane] < 0 ||
                accepted[(size_t) lane] > common_tokens) return false;
            const int slot = slots[(size_t) lane];
            if (slot == -1) continue;
            if (slot < 0 || slot >= common_state_slots) return false;
            if (seen[(size_t) slot]) return false;
            seen[(size_t) slot] = 1;
        }
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
    if (!prevalidated &&
        (!caches || n_caches <= 0 || !commit_rows || !active_slot_ids ||
         commit_rows->type != GGML_TYPE_I64 ||
         active_slot_ids->type != GGML_TYPE_I32 ||
         !ggml_is_contiguous(commit_rows) ||
         !ggml_is_contiguous(active_slot_ids) ||
         commit_rows->ne[0] < 1 || commit_rows->ne[1] < 1 ||
         ggml_nelements(active_slot_ids) != commit_rows->ne[1] ||
         tree_scratch_base < 0 ||
         tree_scratch_stride < commit_rows->ne[0])) return false;
    const int tree_width = (int) commit_rows->ne[0];
    const int n_seqs = (int) commit_rows->ne[1];
    const int n_rows = tree_width*n_seqs;
    int device = -1;
    if (!device_pointer(commit_rows->data, device)) return false;

    int cache_rows = prevalidated ? (int) caches[0]->ne[1] : -1;
    if (!prevalidated) {
        if (!same_device_pointer(active_slot_ids->data, device)) return false;
        std::vector<int64_t> destinations((size_t) n_rows);
        std::vector<int32_t> slots((size_t) n_seqs);
        if (!read_to_host(device, destinations.data(), commit_rows->data,
                       destinations.size()*sizeof(int64_t)) ||
            !read_to_host(device, slots.data(), active_slot_ids->data,
                       slots.size()*sizeof(int32_t))) return false;
        for (int index = 0; index < n_caches; ++index) {
            ggml_tensor * cache = caches[index];
            if (!cache || !ggml_is_contiguous(cache) || cache->ne[0] < 1 ||
                cache->ne[1] < 1 || cache->ne[2] < 1 || cache->ne[3] != 1 ||
                cache->nb[1] < ggml_row_size(cache->type, cache->ne[0]) ||
                !same_device_pointer(cache->data, device)) return false;
            if (cache_rows < 0) cache_rows = (int) cache->ne[1];
            else if (cache->ne[1] != cache_rows) return false;
        }
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
        (!source || !destination || !destination_rows ||
         source->type != destination->type ||
         source->type != GGML_TYPE_BF16 ||
         destination_rows->type != GGML_TYPE_I32 ||
         !ggml_is_contiguous(source) ||
         !ggml_is_contiguous(destination) ||
         !ggml_is_contiguous(destination_rows) ||
         source->ne[0] != destination->ne[0] || source->ne[2] != 1 ||
         source->ne[3] != 1 || destination->ne[2] != 1 ||
         destination->ne[3] != 1 ||
         ggml_nelements(destination_rows) != source->ne[1] ||
         source->nb[1] != destination->nb[1])) return false;
    int device = -1;
    if (!device_pointer(source->data, device)) return false;
    const int n_rows = (int) source->ne[1];
    if (!prevalidated) {
        if (!same_device_pointer(destination->data, device) ||
            !same_device_pointer(destination_rows->data, device)) return false;
        std::vector<int32_t> rows((size_t) n_rows);
        if (!read_to_host(device, rows.data(), destination_rows->data,
                       rows.size()*sizeof(int32_t))) return false;
        for (int row : rows) {
            if (row < -1 || row >= destination->ne[1]) return false;
        }
    }
    if (!commit) return true;
    ggml_cuda_set_device(device);
    // A null stream would be the legacy default stream; never launch on it.
    cudaStream_t stream = commit_stream(device);
    if (!stream) return false;
    constexpr int threads = 256;
    const dim3 grid(
        (unsigned int) ((source->nb[1] + threads - 1)/threads),
        (unsigned int) n_rows, 1);
    (void) cudaGetLastError();
    tree_feature_commit_kernel<<<grid, threads, 0, stream>>>(
        (const uint8_t *) source->data, (uint8_t *) destination->data,
        (const int32_t *) destination_rows->data,
        source->nb[1], n_rows, (int) destination->ne[1]);
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

// ---------------------------------------------------------------------------
// The concurrent fixed-chain commit
// transaction validated from the caller's host copies of the commit maps (no
// read-back, no pointer-attribute queries: devices come from the tensors'
// buffers), then one launch per kind: every K/V cache row copy (16-byte
// vectors), the feature rows, every layer's recurrent state, every conv
// window. Each element gets the same operands in the same order as the
// per-layer kernels above, so the committed bits are the same.

int ggml_cuda_tensor_device(const ggml_tensor * tensor);  // ggml-cuda.cu

namespace {

constexpr int kLanesState      = 128;
constexpr int kLanesMaxTokens  = 16;
constexpr int kLanesThreads    = 256;
constexpr int kLanesBlockCols  = 32;
constexpr int kLanesColLanes   = kLanesThreads / kLanesState;       // 2
constexpr int kLanesThreadCols = kLanesBlockCols / kLanesColLanes;  // 16
constexpr int kLanesStage      = 1 + kLanesState + kLanesBlockCols;
constexpr int kLanesMaxLayers  = 64;
constexpr int kLanesMaxCaches  = 64;

struct lanes_state_layers {
    const float * replay_log[kLanesMaxLayers];
    float *       state[kLanesMaxLayers];
};

struct lanes_conv_layers {
    const float * input[kLanesMaxLayers];
    float *       state[kLanesMaxLayers];
};

struct lanes_cache_table {
    uint8_t * cache[kLanesMaxCaches];
};

// Grid (S / 32, H * n_seqs, layers). Thread (row, lane) owns row `row` of 16
// columns, as gdn_chain_commit_kernel: a wave reads 32 consecutive rows of one
// column. The update is gdn_replay_log_commit_128_scalar_tile_kernel's:
//     current = fmaf(key[row], delta[col], gate * current), token by token.
__global__ void __launch_bounds__(kLanesThreads)
gdn_lanes_state_commit_kernel(const lanes_state_layers layers,
        const int32_t * accepted_prefixes, const int32_t * active_slot_ids,
        int H, int n_tokens, int n_state_slots) {
    const int layer    = blockIdx.z;
    const int sequence = blockIdx.y / H;
    const int head     = blockIdx.y % H;
    const int col_base = blockIdx.x * kLanesBlockCols;
    const int row      = threadIdx.x % kLanesState;
    const int lane     = threadIdx.x / kLanesState;
    const int slot     = active_slot_ids[sequence];
    const int accepted = accepted_prefixes[sequence];
    if (slot < 0 || slot >= n_state_slots || accepted <= 0 ||
        accepted > n_tokens || accepted > kLanesMaxTokens) {
        return;
    }
    __shared__ float gate_s[kLanesMaxTokens];
    __shared__ float key_s[kLanesMaxTokens][kLanesState];
    __shared__ float delta_s[kLanesMaxTokens][kLanesBlockCols];
    constexpr int width = 2*kLanesState + 1;
    const float * log = layers.replay_log[layer] + (size_t) sequence*n_tokens*H*width;
    for (int i = threadIdx.x; i < accepted*kLanesStage; i += kLanesThreads) {
        const int t = i / kLanesStage;
        const int f = i % kLanesStage;
        const float * transition = log + ((size_t) t*H + head)*width;
        if (f == 0) {
            gate_s[t] = transition[0];
        } else if (f <= kLanesState) {
            key_s[t][f - 1] = transition[f];
        } else {
            const int c = f - 1 - kLanesState;
            delta_s[t][c] = transition[1 + kLanesState + col_base + c];
        }
    }
    __syncthreads();
    float * state = layers.state[layer] + ((size_t) slot*H + head)*kLanesState*kLanesState;
    const int col0 = col_base + lane*kLanesThreadCols;
    float s[kLanesThreadCols];
#pragma unroll
    for (int j = 0; j < kLanesThreadCols; ++j) {
        s[j] = state[(size_t) (col0 + j)*kLanesState + row];
    }
    for (int t = 0; t < accepted; ++t) {
        const float g = gate_s[t];
        const float k = key_s[t][row];
#pragma unroll
        for (int j = 0; j < kLanesThreadCols; ++j) {
            s[j] = fmaf(k, delta_s[t][lane*kLanesThreadCols + j], g * s[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < kLanesThreadCols; ++j) {
        state[(size_t) (col0 + j)*kLanesState + row] = s[j];
    }
}

// gdn_conv_replay_log_commit_kernel over every layer: grid (elements / 256,
// n_seqs, layers).
__global__ void gdn_lanes_conv_commit_kernel(const lanes_conv_layers layers,
        const int32_t * accepted_prefixes, const int32_t * active_slot_ids,
        int window, int channels, int n_tokens, int n_seqs, int n_state_slots) {
    const int layer = blockIdx.z;
    const int sequence = blockIdx.y;
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
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
    layers.state[layer][destination] = layers.input[layer][source];
}

// tree_cache_commit_kernel over every cache, 16 bytes per thread: grid
// (vectors / 256, n_caches). Same source and destination rows and guards.
__global__ void tree_cache_commit_vec_kernel(const lanes_cache_table caches,
        const int64_t * commit_rows, const int32_t * active_slot_ids,
        size_t row_bytes, size_t head_stride, int n_heads, int tree_width,
        int n_seqs, int n_cache_rows, int scratch_base, int scratch_stride) {
    const size_t row_vecs = row_bytes / 16;
    const size_t per_head = row_vecs * (size_t) tree_width * n_seqs;
    const size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= per_head * n_heads) return;
    const int head = (int) (index / per_head);
    const size_t rest = index % per_head;
    const int flat = (int) (rest / row_vecs);
    const size_t vec = rest % row_vecs;
    const int lane = flat / tree_width;
    const int node = flat % tree_width;
    const int slot = active_slot_ids[lane];
    const int64_t destination_row = commit_rows[flat];
    if (slot < 0 || destination_row < 0 ||
        destination_row >= n_cache_rows) return;
    const int64_t source_row =
        (int64_t) scratch_base + (int64_t) slot*scratch_stride + node;
    if (source_row < 0 || source_row >= n_cache_rows) return;
    uint8_t * base = caches.cache[blockIdx.y] + (size_t) head*head_stride;
    const uint4 * source = (const uint4 *) (base + (size_t) source_row*row_bytes) + vec;
    uint4 * destination = (uint4 *) (base + (size_t) destination_row*row_bytes) + vec;
    *destination = *source;
}

bool tensor_on(const ggml_tensor * tensor, int device) {
    return tensor != nullptr && tensor->data != nullptr &&
        ggml_cuda_tensor_device(tensor) == device;
}

thread_local size_t g_lanes_commit_count = 0;

} // namespace

extern "C" size_t ggml_backend_cuda_get_tree_commit_host_count(void) {
    return g_lanes_commit_count;
}

extern "C" bool ggml_backend_cuda_tree_commit_transaction_host(
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
        const int64_t * host_commit_rows,
        const int32_t * host_accepted,
        const int32_t * host_slots,
        const int32_t * host_feature_rows) {
    // ---- validation, before any launch (the checks of the *_impl VALIDATE
    // passes, on the host copies).
    if (!caches || n_caches <= 0 || n_caches > kLanesMaxCaches ||
        !replay_logs || !states || !conv_inputs || !conv_states ||
        n_layers <= 0 || n_layers > kLanesMaxLayers ||
        !commit_rows || !accepted_prefixes || !active_slot_ids ||
        !host_commit_rows || !host_accepted || !host_slots || !host_feature_rows ||
        !feature_source || !feature_destination || !feature_destination_rows) {
        std::fprintf(stderr, "[gdn-transaction] host preflight: missing argument\n");
        return false;
    }
    const int device = ggml_cuda_tensor_device(active_slot_ids);
    if (device < 0) return false;
    // K/V.
    if (commit_rows->type != GGML_TYPE_I64 || active_slot_ids->type != GGML_TYPE_I32 ||
        accepted_prefixes->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(commit_rows) || !ggml_is_contiguous(active_slot_ids) ||
        !ggml_is_contiguous(accepted_prefixes) ||
        commit_rows->ne[0] < 1 || commit_rows->ne[1] < 1 ||
        ggml_nelements(active_slot_ids) != commit_rows->ne[1] ||
        ggml_nelements(accepted_prefixes) != commit_rows->ne[1] ||
        tree_scratch_base < 0 || tree_scratch_stride < commit_rows->ne[0] ||
        !tensor_on(commit_rows, device) || !tensor_on(accepted_prefixes, device)) {
        std::fprintf(stderr, "[gdn-transaction] host K/V preflight failed\n");
        return false;
    }
    const int tree_width = (int) commit_rows->ne[0];
    const int n_seqs = (int) commit_rows->ne[1];
    const int n_rows = tree_width*n_seqs;
    int cache_rows = -1;
    bool caches_uniform = true;
    for (int index = 0; index < n_caches; ++index) {
        const ggml_tensor * cache = caches[index];
        if (!cache || !ggml_is_contiguous(cache) || cache->ne[0] < 1 ||
            cache->ne[1] < 1 || cache->ne[2] < 1 || cache->ne[3] != 1 ||
            cache->nb[1] < ggml_row_size(cache->type, cache->ne[0]) ||
            !tensor_on(cache, device)) {
            std::fprintf(stderr, "[gdn-transaction] host K/V preflight failed (cache)\n");
            return false;
        }
        if (cache_rows < 0) cache_rows = (int) cache->ne[1];
        else if (cache->ne[1] != cache_rows) return false;
        caches_uniform = caches_uniform &&
            cache->nb[1] == caches[0]->nb[1] && cache->nb[2] == caches[0]->nb[2] &&
            cache->ne[2] == caches[0]->ne[2] && cache->nb[1] % 16 == 0 &&
            cache->nb[2] % 16 == 0 && ((uintptr_t) cache->data) % 16 == 0;
    }
    for (int lane = 0; lane < n_seqs; ++lane) {
        const int slot = host_slots[lane];
        if (slot == -1) continue;
        if (slot < 0) return false;
        const int64_t source_end = (int64_t) tree_scratch_base +
            (int64_t) slot*tree_scratch_stride + tree_width;
        if (source_end > cache_rows) return false;
        for (int node = 0; node < tree_width; ++node) {
            const int64_t destination = host_commit_rows[(size_t) lane*tree_width + node];
            if (destination < -1 || destination >= cache_rows) return false;
        }
    }
    // Feature rows.
    if (feature_source->type != feature_destination->type ||
        feature_source->type != GGML_TYPE_BF16 ||
        feature_destination_rows->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(feature_source) || !ggml_is_contiguous(feature_destination) ||
        !ggml_is_contiguous(feature_destination_rows) ||
        feature_source->ne[0] != feature_destination->ne[0] || feature_source->ne[2] != 1 ||
        feature_source->ne[3] != 1 || feature_destination->ne[2] != 1 ||
        feature_destination->ne[3] != 1 ||
        ggml_nelements(feature_destination_rows) != feature_source->ne[1] ||
        feature_source->nb[1] != feature_destination->nb[1] ||
        !tensor_on(feature_source, device) || !tensor_on(feature_destination, device) ||
        !tensor_on(feature_destination_rows, device)) {
        std::fprintf(stderr, "[gdn-transaction] host feature preflight failed\n");
        return false;
    }
    const int feature_rows = (int) feature_source->ne[1];
    for (int row = 0; row < feature_rows; ++row) {
        const int destination = host_feature_rows[row];
        if (destination < -1 || destination >= feature_destination->ne[1]) return false;
    }
    // Recurrent state and conv windows.
    int common_tokens = -1;
    int common_state_slots = -1;
    int heads = -1;
    int conv_window = -1;
    int conv_channels = -1;
    bool lanes_state = true;
    for (int layer = 0; layer < n_layers; ++layer) {
        const ggml_tensor * replay_log = replay_logs[layer];
        const ggml_tensor * state = states[layer];
        const ggml_tensor * conv_input = conv_inputs[layer];
        const ggml_tensor * conv_state = conv_states[layer];
        if (!replay_log || !state || !conv_input || !conv_state ||
            replay_log->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
            conv_input->type != GGML_TYPE_F32 || conv_state->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(replay_log) || !ggml_is_contiguous(state) ||
            !ggml_is_contiguous(conv_input) || !ggml_is_contiguous(conv_state) ||
            !tensor_on(replay_log, device) || !tensor_on(state, device) ||
            !tensor_on(conv_input, device) || !tensor_on(conv_state, device)) {
            std::fprintf(stderr, "[gdn-transaction] host recurrent preflight failed\n");
            return false;
        }
        const int64_t state_size = state->ne[0];
        const int64_t layer_heads = state->ne[2];
        const int64_t tokens = replay_log->ne[2];
        const int64_t state_slots = state->ne[3];
        if ((state_size != 16 && state_size != 32 &&
             state_size != 64 && state_size != 128) ||
            state->ne[1] != state_size || layer_heads < 1 ||
            replay_log->ne[1] != layer_heads || replay_log->ne[3] != n_seqs ||
            (replay_log->ne[0] != 2*state_size + 1 &&
             replay_log->ne[0] != 3*state_size) || tokens < 1 ||
            conv_state->ne[0] < 1 || conv_state->ne[1] < 1 ||
            conv_state->ne[2] != state_slots || conv_state->ne[3] != 1 ||
            conv_input->ne[0] != conv_state->ne[0] + tokens ||
            conv_input->ne[1] != conv_state->ne[1] ||
            conv_input->ne[2] != n_seqs || conv_input->ne[3] != 1) {
            std::fprintf(stderr, "[gdn-transaction] host recurrent preflight failed (shape)\n");
            return false;
        }
        if (common_tokens < 0) {
            common_tokens = (int) tokens;
            common_state_slots = (int) state_slots;
            heads = (int) layer_heads;
            conv_window = (int) conv_state->ne[0];
            conv_channels = (int) conv_state->ne[1];
        } else if (tokens != common_tokens || state_slots != common_state_slots) {
            return false;
        }
        lanes_state = lanes_state && state_size == kLanesState &&
            replay_log->ne[0] == 2*state_size + 1 && layer_heads == heads &&
            conv_state->ne[0] == conv_window && conv_state->ne[1] == conv_channels;
    }
    lanes_state = lanes_state && common_tokens <= kLanesMaxTokens &&
        (int64_t) heads * n_seqs <= 65535;
    {
        std::vector<uint8_t> seen((size_t) common_state_slots, 0);
        for (int lane = 0; lane < n_seqs; ++lane) {
            if (host_accepted[lane] < 0 || host_accepted[lane] > common_tokens) return false;
            const int slot = host_slots[lane];
            if (slot == -1) continue;
            if (slot < 0 || slot >= common_state_slots) return false;
            if (seen[(size_t) slot]) return false;
            seen[(size_t) slot] = 1;
        }
    }

    // ---- commit. No recoverable error may escape after the first launch.
    ggml_cuda_set_device(device);
    cudaStream_t stream = commit_stream(device);
    if (!stream) return false;
    constexpr int threads = 256;
    (void) cudaGetLastError();
    if (caches_uniform) {
        lanes_cache_table table{};
        for (int index = 0; index < n_caches; ++index) {
            table.cache[index] = (uint8_t *) caches[index]->data;
        }
        const size_t row_bytes = caches[0]->nb[1];
        const int n_heads = (int) caches[0]->ne[2];
        const size_t vectors = row_bytes/16 * (size_t) n_rows * n_heads;
        const dim3 grid((unsigned int) ((vectors + threads - 1)/threads), (unsigned int) n_caches, 1);
        tree_cache_commit_vec_kernel<<<grid, threads, 0, stream>>>(
            table, (const int64_t *) commit_rows->data, (const int32_t *) active_slot_ids->data,
            row_bytes, caches[0]->nb[2], n_heads, tree_width, n_seqs, cache_rows,
            tree_scratch_base, tree_scratch_stride);
    } else {
        for (int index = 0; index < n_caches; ++index) {
            ggml_tensor * cache = caches[index];
            const dim3 grid(
                (unsigned int) ((cache->nb[1] + threads - 1)/threads),
                (unsigned int) n_rows, (unsigned int) cache->ne[2]);
            tree_cache_commit_kernel<<<grid, threads, 0, stream>>>(
                (uint8_t *) cache->data, (const int64_t *) commit_rows->data,
                (const int32_t *) active_slot_ids->data,
                cache->nb[1], cache->nb[2], (int) cache->ne[2],
                tree_width, n_seqs, cache_rows,
                tree_scratch_base, tree_scratch_stride);
        }
    }
    {
        const dim3 grid(
            (unsigned int) ((feature_source->nb[1] + threads - 1)/threads),
            (unsigned int) feature_rows, 1);
        tree_feature_commit_kernel<<<grid, threads, 0, stream>>>(
            (const uint8_t *) feature_source->data, (uint8_t *) feature_destination->data,
            (const int32_t *) feature_destination_rows->data,
            feature_source->nb[1], feature_rows, (int) feature_destination->ne[1]);
    }
    if (lanes_state) {
        lanes_state_layers state_args{};
        lanes_conv_layers conv_args{};
        for (int layer = 0; layer < n_layers; ++layer) {
            state_args.replay_log[layer] = (const float *) replay_logs[layer]->data;
            state_args.state[layer] = (float *) states[layer]->data;
            conv_args.input[layer] = (const float *) conv_inputs[layer]->data;
            conv_args.state[layer] = (float *) conv_states[layer]->data;
        }
        const dim3 state_grid(kLanesState / kLanesBlockCols, (unsigned int) (heads*n_seqs), (unsigned int) n_layers);
        gdn_lanes_state_commit_kernel<<<state_grid, kLanesThreads, 0, stream>>>(
            state_args, (const int32_t *) accepted_prefixes->data,
            (const int32_t *) active_slot_ids->data, heads, common_tokens, common_state_slots);
        const int conv_elements = conv_window*conv_channels;
        const dim3 conv_grid((unsigned int) ((conv_elements + threads - 1)/threads),
                             (unsigned int) n_seqs, (unsigned int) n_layers);
        gdn_lanes_conv_commit_kernel<<<conv_grid, threads, 0, stream>>>(
            conv_args, (const int32_t *) accepted_prefixes->data,
            (const int32_t *) active_slot_ids->data, conv_window, conv_channels,
            common_tokens, n_seqs, common_state_slots);
    } else if (!gdn_replay_log_commit_many_impl(
            replay_logs, states, conv_inputs, conv_states, n_layers,
            accepted_prefixes, active_slot_ids, commit_mode::COMMIT_PREVALIDATED)) {
        GGML_ABORT("tree commit failed after recurrent mutation began");
    }
    if (cudaGetLastError() != cudaSuccess) {
        GGML_ABORT("tree commit launch failed after mutation began");
    }
    if (!commit_sync(device)) {
        GGML_ABORT("tree commit failed during final synchronization");
    }
    ++g_lanes_commit_count;
    return true;
}
