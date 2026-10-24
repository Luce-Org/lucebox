#include "weights.h"

#include "common/platform_io.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace luce::moss_image {

namespace {

constexpr size_t kAlign = 256;

size_t align_up(const size_t n, const size_t a) {
    return (n + a - 1) / a * a;
}

enum class Convert {
    none,         // the stored bytes
    bf16_to_f32,  // a bf16 vector widened to f32
    conv_taps,    // f16 [oc][ic][kh][kw] to [oc][kh * kw][ic]
};

struct Upload {
    uint64_t file_offset = 0;
    size_t   file_bytes  = 0;
    Convert  convert     = Convert::none;
    size_t   dst_offset  = 0;
    size_t   dst_bytes   = 0;
    int64_t  oc = 0, ic = 0, taps = 0;  // conv_taps
};

void convert(const Upload & u, const uint8_t * in, uint8_t * out) {
    if (u.convert == Convert::bf16_to_f32) {
        const uint16_t * src = reinterpret_cast<const uint16_t *>(in);
        uint32_t *       dst = reinterpret_cast<uint32_t *>(out);
        for (size_t i = 0; i < u.file_bytes / 2; ++i) {
            dst[i] = uint32_t(src[i]) << 16;
        }
    } else if (u.convert == Convert::conv_taps) {
        const uint16_t * src = reinterpret_cast<const uint16_t *>(in);
        uint16_t *       dst = reinterpret_cast<uint16_t *>(out);
        for (int64_t oc = 0; oc < u.oc; ++oc) {
            for (int64_t tap = 0; tap < u.taps; ++tap) {
                for (int64_t ic = 0; ic < u.ic; ++ic) {
                    dst[(oc * u.taps + tap) * u.ic + ic] = src[(oc * u.ic + ic) * u.taps + tap];
                }
            }
        }
    }
}

// Lays the tensors out in one device allocation (checking each one's type and shape), then streams them from the
// file through two pinned buffers, converting on the host while the previous upload is in flight.
class Loader {
public:
    Loader(gguf_context * gguf, ggml_context * meta) : gguf_(gguf), meta_(meta) {}

    const std::string & error() const { return error_; }
    size_t              total() const { return total_; }

    // What bind() asked for, to point the fields at any later copy of the allocation.
    std::vector<std::function<void(uint8_t *)>> take_binds() { return std::move(fixups_); }

    size_t reserve(const size_t bytes) {
        const size_t offset = total_;
        total_              = align_up(total_ + bytes, kAlign);
        return offset;
    }

    template <typename T> void bind(const T *& field, const size_t offset) {
        fixups_.push_back([&field, offset](uint8_t * base) { field = reinterpret_cast<const T *>(base + offset); });
    }

    // A [n] vector, bf16 or f32 in the file, as f32 at `dst`.
    void vector_into(const std::string & name, const int64_t n, const size_t dst) {
        const ggml_tensor * t = find(name, { n });
        if (t == nullptr) {
            return;
        }
        if (t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_F32) {
            fail(name + ": expected bf16 or f32");
            return;
        }
        Upload u   = upload_of(name, t, dst);
        u.convert  = t->type == GGML_TYPE_BF16 ? Convert::bf16_to_f32 : Convert::none;
        u.dst_bytes = size_t(n) * sizeof(float);
        uploads_.push_back(u);
    }

    size_t vector(const std::string & name, const int64_t n) {
        const size_t dst = reserve(size_t(n) * sizeof(float));
        vector_into(name, n, dst);
        return dst;
    }

    // An [n][k] matrix of `type`, as stored, at `dst`.
    void matrix_into(const std::string & name, const ggml_type type, const int64_t n, const int64_t k,
                     const size_t dst) {
        const ggml_tensor * t = find(name, { k, n });
        if (t == nullptr) {
            return;
        }
        if (t->type != type) {
            fail(name + ": expected " + ggml_type_name(type) + ", found " + ggml_type_name(t->type));
            return;
        }
        uploads_.push_back(upload_of(name, t, dst));
    }

    size_t matrix(const std::string & name, const ggml_type type, const int64_t n, const int64_t k) {
        const size_t dst = reserve(size_t(n) * size_t(k) * ggml_type_size(type));
        matrix_into(name, type, n, k, dst);
        return dst;
    }

    // An f16 convolution weight [oc][ic][k][k] as [oc][k * k][ic] at `dst`.
    void conv_weight_into(const std::string & name, const int64_t oc, const int64_t ic, const int64_t k,
                          const size_t dst) {
        const ggml_tensor * t = find(name, { k, k, ic, oc });
        if (t == nullptr) {
            return;
        }
        if (t->type != GGML_TYPE_F16) {
            fail(name + ": expected f16");
            return;
        }
        Upload u = upload_of(name, t, dst);
        if (k > 1) {
            u.convert = Convert::conv_taps;
            u.oc      = oc;
            u.ic      = ic;
            u.taps    = k * k;
        }
        uploads_.push_back(u);
    }

    void linear(I8Linear & l, const std::string & name, const int64_t n, const int64_t k) {
        bind(l.w, matrix(name + ".weight", GGML_TYPE_I8, n, k));
        bind(l.scale, vector(name + ".weight_scale", n));
    }

    void linear(Bf16Linear & l, const std::string & name, const int64_t n, const int64_t k, const bool bias) {
        bind(l.w, matrix(name + ".weight", GGML_TYPE_BF16, n, k));
        if (bias) {
            bind(l.b, vector(name + ".bias", n));
        }
    }

    // Two [n][k] linears with biases as one [2n][k]: `first` rows, then `second` rows.
    void joined_linear(Bf16Linear & l, const std::string & first, const std::string & second, const int64_t n,
                       const int64_t k) {
        const size_t w_bytes = size_t(n) * size_t(k) * sizeof(uint16_t);
        const size_t w       = reserve(2 * w_bytes);
        matrix_into(first + ".weight", GGML_TYPE_BF16, n, k, w);
        matrix_into(second + ".weight", GGML_TYPE_BF16, n, k, w + w_bytes);
        const size_t b = reserve(2 * size_t(n) * sizeof(float));
        vector_into(first + ".bias", n, b);
        vector_into(second + ".bias", n, b + size_t(n) * sizeof(float));
        bind(l.w, w);
        bind(l.b, b);
    }

    void conv(Conv2d & c, const std::string & name, const int oc, const int ic, const int k) {
        const size_t w = reserve(size_t(oc) * size_t(ic) * size_t(k) * size_t(k) * sizeof(uint16_t));
        conv_weight_into(name + ".weight", oc, ic, k, w);
        bind(c.w, w);
        bind(c.b, vector(name + ".bias", oc));
        c.oc = oc;
        c.ic = ic;
        c.k  = k;
    }

    void group_norm(GroupNorm & g, const std::string & name, const int channels) {
        bind(g.w, vector(name + ".weight", channels));
        bind(g.b, vector(name + ".bias", channels));
        g.channels = channels;
    }

    void res_block(VaeResBlock & r, const std::string & name, const int ic, const int oc) {
        group_norm(r.norm1, name + ".norm1", ic);
        conv(r.conv1, name + ".conv1", oc, ic, 3);
        group_norm(r.norm2, name + ".norm2", oc);
        conv(r.conv2, name + ".conv2", oc, oc, 3);
        if (ic != oc) {
            conv(r.nin, name + ".nin_shortcut", oc, ic, 1);
        }
    }

    void attention(VaeAttention & a, const std::string & name, const int channels) {
        group_norm(a.norm, name + ".norm", channels);
        const size_t w_bytes = size_t(channels) * size_t(channels) * sizeof(uint16_t);
        const size_t w       = reserve(3 * w_bytes);
        const size_t b       = reserve(3 * size_t(channels) * sizeof(float));
        const char * parts[3] = { ".q", ".k", ".v" };
        for (int i = 0; i < 3; ++i) {
            conv_weight_into(name + parts[i] + ".weight", channels, channels, 1, w + i * w_bytes);
            vector_into(name + parts[i] + ".bias", channels, b + i * size_t(channels) * sizeof(float));
        }
        bind(a.qkv.w, w);
        bind(a.qkv.b, b);
        a.qkv.oc = 3 * channels;
        a.qkv.ic = channels;
        a.qkv.k  = 1;
        conv(a.proj, name + ".proj_out", channels, channels, 1);
    }

    // Every tensor into `host`, an allocation laid out as on the device, on a few threads.
    bool read_into(const std::string & path, uint8_t * host) {
        luce::common::ReadOnlyFile file;
        if (!file.open(path, false, &error_)) {
            return false;
        }
        constexpr size_t   kThreads = 4;
        std::atomic<bool>  failed{ false };
        std::vector<std::thread> threads;
        for (size_t t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                std::vector<uint8_t> staged;
                for (size_t i = t; i < uploads_.size() && !failed; i += kThreads) {
                    const Upload & u   = uploads_[i];
                    uint8_t *      dst = host + u.dst_offset;
                    if (u.convert != Convert::none) {
                        staged.resize(u.file_bytes);
                    }
                    uint8_t * buf = u.convert != Convert::none ? staged.data() : dst;
                    if (!file.read_at(u.file_offset, buf, u.file_bytes)) {
                        failed = true;
                        break;
                    }
                    if (u.convert != Convert::none) {
                        convert(u, buf, dst);
                    }
                }
            });
        }
        for (std::thread & t : threads) {
            t.join();
        }
        if (failed) {
            error_ = "cannot read " + path;
        }
        return !failed;
    }

    bool upload(const std::string & path, uint8_t * device, hipStream_t stream) {
        luce::common::ReadOnlyFile file;
        if (!file.open(path, false, &error_)) {
            return false;
        }
        size_t stage = 0;
        for (const Upload & u : uploads_) {
            stage = std::max(stage, align_up(u.file_bytes, 16) + (u.convert != Convert::none ? u.dst_bytes : 0));
        }
        void *     pinned[2] = {};
        hipEvent_t done[2]   = {};
        bool       ok        = true;
        for (int i = 0; i < 2; ++i) {
            ok = ok && hipHostMalloc(&pinned[i], stage, hipHostMallocDefault) == hipSuccess &&
                 hipEventCreateWithFlags(&done[i], hipEventDisableTiming) == hipSuccess;
        }
        for (size_t i = 0; ok && i < uploads_.size(); ++i) {
            const Upload & u   = uploads_[i];
            uint8_t *      buf = static_cast<uint8_t *>(pinned[i % 2]);
            if (hipEventSynchronize(done[i % 2]) != hipSuccess || !file.read_at(u.file_offset, buf, u.file_bytes)) {
                error_ = "cannot read " + path;
                ok     = false;
                break;
            }
            const uint8_t * src = buf;
            if (u.convert != Convert::none) {
                uint8_t * converted = buf + align_up(u.file_bytes, 16);
                convert(u, buf, converted);
                src = converted;
            }
            ok = hipMemcpyAsync(device + u.dst_offset, src, u.dst_bytes, hipMemcpyHostToDevice, stream) == hipSuccess &&
                 hipEventRecord(done[i % 2], stream) == hipSuccess;
        }
        ok = hipStreamSynchronize(stream) == hipSuccess && ok;
        if (!ok && error_.empty()) {
            error_ = "uploading the weights failed";
        }
        for (int i = 0; i < 2; ++i) {
            if (done[i] != nullptr) {
                (void) hipEventDestroy(done[i]);
            }
            if (pinned[i] != nullptr) {
                (void) hipHostFree(pinned[i]);
            }
        }
        if (ok) {
            for (const auto & fixup : fixups_) {
                fixup(device);
            }
        }
        return ok;
    }

private:
    void fail(const std::string & message) {
        if (error_.empty()) {
            error_ = message;
        }
    }

    // The tensor when it exists with exactly the shape `ne` (ggml order, innermost first).
    const ggml_tensor * find(const std::string & name, std::initializer_list<int64_t> ne) {
        const ggml_tensor * t = ggml_get_tensor(meta_, name.c_str());
        if (t == nullptr) {
            fail("missing tensor " + name);
            return nullptr;
        }
        int dim = 0;
        for (const int64_t n : ne) {
            if (t->ne[dim++] != n) {
                fail("unexpected shape of " + name);
                return nullptr;
            }
        }
        for (; dim < GGML_MAX_DIMS; ++dim) {
            if (t->ne[dim] != 1) {
                fail("unexpected shape of " + name);
                return nullptr;
            }
        }
        return t;
    }

    Upload upload_of(const std::string & name, const ggml_tensor * t, const size_t dst) const {
        const int64_t id = gguf_find_tensor(gguf_, name.c_str());
        Upload        u;
        u.file_offset = gguf_get_data_offset(gguf_) + gguf_get_tensor_offset(gguf_, id);
        u.file_bytes  = ggml_nbytes(t);
        u.dst_offset  = dst;
        u.dst_bytes   = u.file_bytes;
        return u;
    }

    gguf_context *                             gguf_;
    ggml_context *                             meta_;
    std::string                                error_;
    size_t                                     total_ = 0;
    std::vector<Upload>                        uploads_;
    std::vector<std::function<void(uint8_t *)>> fixups_;
};

void lay_out_dit(Loader & l, DitWeights & dit, size_t * part_end) {
    l.linear(dit.text_proj, "dit.text_embedder.proj", kHidden, kTextHidden, true);
    for (int b = 0; b < kTextBlocks; ++b) {
        TextBlock &       blk = dit.text_blocks[b];
        const std::string p   = "dit.text_embedder.blocks." + std::to_string(b) + ".";
        l.linear(blk.qkv, p + "qkv", 3 * kHidden, kHidden, true);
        l.linear(blk.proj, p + "proj", kHidden, kHidden, true);
        l.joined_linear(blk.up_gate, p + "ffn_up", p + "ffn_gate", kFfn, kHidden);
        l.linear(blk.down, p + "ffn_down", kHidden, kFfn, true);
        l.bind(blk.norm1, l.vector(p + "norm1.weight", kHidden));
        l.bind(blk.norm2, l.vector(p + "norm2.weight", kHidden));
        l.bind(blk.q_norm, l.vector(p + "q_norm.weight", kHeadDim));
        l.bind(blk.k_norm, l.vector(p + "k_norm.weight", kHeadDim));
    }
    l.linear(dit.video_embedder, "dit.video_embedder", kHidden, kLatentChannels, true);
    l.linear(dit.time_in, "dit.time_embedding.0", kHidden, kTimeFreqs, true);
    l.linear(dit.time_out, "dit.time_embedding.2", kHidden, kHidden, true);
    l.linear(dit.block_modulation, "dit.shared_block_modulation", 6 * kHidden, kHidden, true);
    l.linear(dit.head_modulation, "dit.video_head_modulation", 2 * kHidden, kHidden, true);
    l.bind(dit.final_norm, l.vector("dit.final_norm_video.weight", kHidden));
    l.linear(dit.final_linear, "dit.final_linear_video", kLatentChannels, kHidden, false);
    part_end[0] = l.total();
    for (int i = 0; i < kLayers; ++i) {
        DitLayer &        layer = dit.layers[i];
        const std::string p     = "dit.layers." + std::to_string(i) + ".";
        l.linear(layer.qkv, p + "proj.qkv", 3 * kHidden, kHidden);
        l.linear(layer.o, p + "proj.o", kHidden, kHidden);
        l.linear(layer.up_gate, p + "proj.ffn_up_gate", 2 * kFfn, kHidden);
        l.linear(layer.down, p + "proj.ffn_down", kHidden, kFfn);
        l.bind(layer.norm1, l.vector(p + "norm1.weight", kHidden));
        l.bind(layer.norm2, l.vector(p + "norm2.weight", kHidden));
        l.bind(layer.q_norm, l.vector(p + "q_norm.weight", kHeadDim));
        l.bind(layer.k_norm, l.vector(p + "k_norm.weight", kHeadDim));
        part_end[1 + i] = l.total();
    }
}

void lay_out_vae(Loader & l, VaeDecoder & vae, size_t * part_end) {
    l.bind(vae.latents_mean, l.vector("vae.latents_mean", kLatentChannels));
    l.bind(vae.latents_std, l.vector("vae.latents_std", kLatentChannels));
    l.conv(vae.post_quant, "vae.decoder.post_quant_conv", 32, 32, 1);
    l.conv(vae.conv_in, "vae.decoder.conv_in", 512, 32, 3);
    l.res_block(vae.mid_block1, "vae.decoder.mid.block_1", 512, 512);
    l.attention(vae.mid_attn, "vae.decoder.mid.attn_1", 512);
    l.res_block(vae.mid_block2, "vae.decoder.mid.block_2", 512, 512);
    const int channels[4] = { 128, 256, 512, 512 };
    for (int level = 3; level >= 0; --level) {
        const std::string p  = "vae.decoder.up." + std::to_string(level);
        const int         oc = channels[level];
        const int         ic = level == 3 ? 512 : channels[level + 1];
        for (int b = 0; b < 3; ++b) {
            l.res_block(vae.up_blocks[level][b], p + ".block." + std::to_string(b), b == 0 ? ic : oc, oc);
        }
        if (level > 0) {
            l.conv(vae.upsample[level], p + ".upsample.conv", oc, oc, 3);
        }
    }
    l.group_norm(vae.norm_out, "vae.decoder.norm_out", 128);
    l.conv(vae.conv_out, "vae.decoder.conv_out", 3, 128, 3);
    part_end[kLayers + 1] = l.total();
}

// The GGUF's metadata, freed with it; open() checks that it is a moss-image file.
struct GgufFile {
    ggml_context * meta = nullptr;
    gguf_context * gguf = nullptr;

    GgufFile() = default;
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;

    ~GgufFile() {
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        if (meta != nullptr) {
            ggml_free(meta);
        }
    }

    bool open(const std::string & path, std::string & error) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx      = &meta;
        gguf            = gguf_init_from_file(path.c_str(), params);
        if (gguf == nullptr) {
            error = "cannot read the GGUF " + path;
            return false;
        }
        const int64_t arch = gguf_find_key(gguf, "general.architecture");
        if (arch < 0 || gguf_get_kv_type(gguf, arch) != GGUF_TYPE_STRING ||
            std::strcmp(gguf_get_val_str(gguf, arch), "moss-image") != 0) {
            error = path + " is not a moss-image GGUF";
            return false;
        }
        return true;
    }
};

} // namespace

Weights::~Weights() {
    if (device_ != nullptr) {
        (void) hipFree(device_);
    }
    if (host_ != nullptr) {
        (void) hipHostFree(host_);
    }
}

bool Weights::load(const std::string & path, hipStream_t stream, std::string & error) {
    GgufFile file;
    if (!file.open(path, error)) {
        return false;
    }
    Loader loader(file.gguf, file.meta);
    lay_out_dit(loader, dit, part_end_);
    lay_out_vae(loader, vae, part_end_);
    if (!loader.error().empty()) {
        error = loader.error();
        return false;
    }
    bytes_ = loader.total();
    if (hipMalloc(&device_, bytes_) != hipSuccess) {
        device_ = nullptr;
        error   = "cannot allocate " + std::to_string(bytes_ >> 20) + " MiB for the weights";
        return false;
    }
    if (!loader.upload(path, static_cast<uint8_t *>(device_), stream)) {
        error = loader.error();
        return false;
    }
    bind_ = loader.take_binds();
    return true;
}

bool Weights::read(const std::string & path, std::string & error) {
    GgufFile file;
    if (!file.open(path, error)) {
        return false;
    }
    Loader loader(file.gguf, file.meta);
    lay_out_dit(loader, dit, part_end_);
    lay_out_vae(loader, vae, part_end_);
    if (!loader.error().empty()) {
        error = loader.error();
        return false;
    }
    bytes_ = loader.total();
    if (hipHostMalloc(&host_, bytes_, hipHostMallocDefault) != hipSuccess) {
        host_ = nullptr;
        error = "cannot pin " + std::to_string(bytes_ >> 20) + " MiB of host memory for the weights";
        return false;
    }
    if (!loader.read_into(path, static_cast<uint8_t *>(host_))) {
        error = loader.error();
        return false;
    }
    bind_ = loader.take_binds();
    return true;
}

bool Weights::upload(hipStream_t stream, hipEvent_t * ready, const std::function<bool()> * stopped,
                     std::string & error) {
    if (host_ == nullptr) {
        error = "the weights are not in host memory";
        return false;
    }
    if (device_ == nullptr) {
        if (hipMalloc(&device_, bytes_) != hipSuccess) {
            device_ = nullptr;
            error   = "out of GPU memory for the weights (" + std::to_string(bytes_ >> 20) + " MiB)";
            return false;
        }
        for (const auto & bind : bind_) {
            bind(static_cast<uint8_t *>(device_));
        }
    }
    // Pieces of at most 128 MiB: with `stopped`, each one is waited for, so that the copies stop soon.
    constexpr size_t kPiece = size_t(128) << 20;
    const uint8_t *  host   = static_cast<const uint8_t *>(host_);
    uint8_t *        device = static_cast<uint8_t *>(device_);
    size_t           begin  = 0;
    bool             ok     = true;
    for (int part = 0; ok && part < kParts; ++part) {
        while (ok && begin < part_end_[part]) {
            if (stopped != nullptr && (*stopped)()) {
                release();
                return false;
            }
            const size_t bytes = std::min(kPiece, part_end_[part] - begin);
            ok = hipMemcpyAsync(device + begin, host + begin, bytes, hipMemcpyHostToDevice, stream) == hipSuccess &&
                 (stopped == nullptr || hipStreamSynchronize(stream) == hipSuccess);
            begin += bytes;
        }
        ok = ok && hipEventRecord(ready[part], stream) == hipSuccess;
    }
    if (!ok) {
        (void) hipStreamSynchronize(stream);
        release();
        error = "copying the weights to the GPU failed";
    }
    return ok;
}

void Weights::release() {
    if (device_ != nullptr) {
        (void) hipFree(device_);
        device_ = nullptr;
    }
}

} // namespace luce::moss_image
