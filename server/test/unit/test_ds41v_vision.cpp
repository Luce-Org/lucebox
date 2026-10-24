// DeepSeek V4.1 vision encoder runtime: the blocked attention V4.1 images use
// against the whole-image attention V4 Vision qualified, the projector
// metadata of both families, and the projector-to-model check.
#include "deepseek4/deepseek4_vision.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace luce::vision;
namespace {
int g_failures = 0;
void check(bool value, const std::string & what) {
    if (!value) { ++g_failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}

// attention() and attention_blocked() on the same q, k, v ([64, 16, n]).
std::vector<float> run_attention(ggml_backend_t backend, int64_t n, int64_t block, const std::vector<float> & q,
                                 const std::vector<float> & k, const std::vector<float> & v) {
    const size_t nodes = 1024;
    ggml_init_params params{nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true};
    ggml_context * c = ggml_init(params);
    auto make = [&](const char * name) {
        ggml_tensor * t = ggml_new_tensor_3d(c, GGML_TYPE_F32, 64, 16, n);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };
    ggml_tensor * qt = make("q"), * kt = make("k"), * vt = make("v");
    ggml_tensor * out = block > 0 ? detail::attention_blocked(c, qt, kt, vt, block, backend)
                                  : detail::attention(c, qt, kt, vt, backend);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<float> result;
    if (ggml_gallocr_alloc_graph(alloc, gf)) {
        ggml_backend_tensor_set(qt, q.data(), 0, q.size() * sizeof(float));
        ggml_backend_tensor_set(kt, k.data(), 0, k.size() * sizeof(float));
        ggml_backend_tensor_set(vt, v.data(), 0, v.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
            result.resize(size_t(ggml_nelements(out)));
            ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(c);
    return result;
}

void test_blocked_attention(ggml_backend_t backend, const char * name, bool exact) {
    uint32_t state = 20261018u;
    auto next = [&] { state = state * 1664525u + 1013904223u; return float(int32_t(state >> 8) - (1 << 23)) / float(1 << 23); };
    for (int64_t n : {16, 70, 300}) {
        std::vector<float> q(size_t(64 * 16 * n)), k(q.size()), v(q.size());
        for (auto & x : q) x = 2.0f * next();
        for (auto & x : k) x = 2.0f * next();
        for (auto & x : v) x = next();
        const auto whole = run_attention(backend, n, 0, q, k, v);
        check(!whole.empty(), std::string(name) + ": whole attention ran");
        for (int64_t block : {int64_t(16), int64_t(48), n}) {
            const auto blocked = run_attention(backend, n, block, q, k, v);
            check(blocked.size() == whole.size(), std::string(name) + ": blocked attention ran");
            double worst = 0.0;
            for (size_t i = 0; i < blocked.size() && i < whole.size(); ++i)
                worst = std::max(worst, double(std::fabs(blocked[i] - whole[i])));
            const std::string what = std::string(name) + ": n=" + std::to_string(n) + " block=" + std::to_string(block);
            // CPU: every element is the same dot products in the same order.
            // GPU: the whole-image path has its own source-order AV kernel, so
            // the two agree to rounding (outputs are BF16-rounded).
            check(exact ? worst == 0.0 : worst <= 1e-2, what + " max diff " + std::to_string(worst));
        }
    }
}

void test_query_block_sizes() {
    check(detail::attention_query_block(1) == 16, "a tiny image is one block of 16");
    check(detail::attention_query_block(300) == 300, "a small image is one block");
    for (int64_t n : {4096, 8649, 9189}) {
        const int64_t block = detail::attention_query_block(n);
        check(block % 16 == 0 && block >= 16 && block <= n, "block rows for " + std::to_string(n));
        check(2 * 16 * 4 * n * block <= 512LL * 1024 * 1024, "block scores under 512 MiB for " + std::to_string(n));
    }
}

void test_image_block_tokens() {
    // V4.1: [start] + (columns + newline) * rows + [end].
    check(image_block_tokens(VisionVariant::V41, 1, 1) == 4, "V4.1 1x1");
    check(image_block_tokens(VisionVariant::V41, 31, 31) == 994, "V4.1 31x31");
    check(image_block_tokens(VisionVariant::V41, 1, 1021) == 1024, "V4.1 one row at the budget");
    // V4: the row-pair layout and the three compression pads its budget keeps.
    check(image_block_tokens(VisionVariant::V4, 1, 1) == 9, "V4 1x1");
}

void test_projector_mismatch() {
    VisionConfig v41;
    v41.variant = VisionVariant::V41;
    v41.source_revision = "1bae0e1088993b9b82de08f59de5be5a34e42170";
    VisionConfig v4;
    check(projector_mismatch(v41, true, "1bae0e1088993b9b82de08f59de5be5a34e42170").empty(), "matching pair");
    check(projector_mismatch(v41, true, "").empty(), "a model without a recorded revision");
    check(!projector_mismatch(v41, true, "df42c109f1defefcbfcedbe7d905718a12266e40").empty(), "another checkpoint");
    check(!projector_mismatch(v41, false, "").empty(), "V4.1 projector on V4");
    check(!projector_mismatch(v4, true, "").empty(), "V4 projector on V4.1");
    check(projector_mismatch(v4, false, "anything").empty(), "V4 Vision records no revision");
}

// A projector header without tensors: load() reads and checks its metadata
// first, so "wrong projector tensor count" means the metadata was accepted.
std::string load_header(const std::string & path, bool v41, int dimension, const char * layout = nullptr) {
    gguf_context * g = gguf_init_empty();
    const std::string p = "deepseek4.vision.";
    auto u32 = [&](const std::string & k, uint32_t v) { gguf_set_val_u32(g, k.c_str(), v); };
    auto f32 = [&](const std::string & k, float v) { gguf_set_val_f32(g, k.c_str(), v); };
    auto str = [&](const std::string & k, const char * v) { gguf_set_val_str(g, k.c_str(), v); };
    str("general.architecture", "deepseek4_vision");
    str("general.type", "mmproj");
    u32("general.alignment", 32);
    u32(p + "schema_version", v41 ? 2 : 1);
    if (v41) {
        str("general.source.revision", "1bae0e1088993b9b82de08f59de5be5a34e42170");
        str(p + "checkpoint_variant", "v4.1-flash");
    }
    u32(p + "block_count", 32); u32(p + "embedding_length", 1024); u32(p + "attention.head_count", 16);
    u32(p + "attention.head_dimension", 64); u32(p + "feed_forward_length", 2816); u32(p + "patch_size", 14);
    u32(p + "downsample_ratio", 3); u32(p + "aligner_input_length", 9216);
    u32(p + "language_embedding_length", v41 ? 5120 : 4096); u32(p + "vocabulary_size", 129280);
    u32(p + "image.max_tokens", v41 ? 1024 : 384); u32(p + "image.min_pixels", v41 ? 295936 : 147456);
    u32(p + "image.layout_version", v41 ? 2 : 1); u32(p + "image.compression_alignment", v41 ? 1 : 4);
    u32(p + "image.sentinel_type_count", v41 ? 4 : 5);
    if (v41) u32(p + "image.token_id", 129264);
    f32(p + "rope.freq_base", 10000.f); f32(p + "attention.layer_norm_rms_epsilon", 1e-6f);
    f32(p + "image.max_aspect_ratio", v41 ? 0.f : 8.f);
    str(p + "attention.rope_layout", "2d-half-split-height-width");
    str(p + "image.patch_layout", "channel-major");
    str(p + "image.layout", layout ? layout : (v41 ? "reading-order" : "n"));
    str(p + "image.layout_recipe", v41 ? "row-newline" : "row-pair-column-interleave");
    str(p + "image.sentinel_types", v41 ? "start,image,newline,end" : "start,pad,image,newline,end");
    str(p + "aligner.padding", "bottom-right"); str(p + "aligner.patch_layout", "channel-first-unfold");
    str(p + "aligner.activation", "gelu-exact");
    const float half[3] = {0.5f, 0.5f, 0.5f};
    gguf_set_arr_data(g, (p + "image.normalization_mean").c_str(), GGUF_TYPE_FLOAT32, half, 3);
    gguf_set_arr_data(g, (p + "image.normalization_std").c_str(), GGUF_TYPE_FLOAT32, half, 3);
    const bool written = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    if (!written) return "could not write the test header";
    ggml_backend_t cpu = ggml_backend_cpu_init();
    VisionRuntime runtime;
    std::string error;
    runtime.load(path, cpu, dimension, 129280, error);
    ggml_backend_free(cpu);
    std::remove(path.c_str());
    return error;
}

void test_projector_metadata() {
    const std::string dir = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    const std::string path = dir + "/ds41v_projector_" + std::to_string(getpid()) + ".gguf";
    const auto accepted = [](const std::string & e) { return e.find("wrong projector tensor count") != std::string::npos; };
    check(accepted(load_header(path, true, 5120)), "the V4.1 header is accepted on a 5120-wide model");
    check(accepted(load_header(path, false, 4096)), "the V4 header is accepted on a 4096-wide model");
    const std::string on_v4 = load_header(path, true, 4096);
    check(on_v4.find("V4.1's") != std::string::npos, "the V4.1 header refuses a 4096-wide model: " + on_v4);
    const std::string on_v41 = load_header(path, false, 5120);
    check(on_v41.find("V4 Vision's") != std::string::npos, "the V4 header refuses a 5120-wide model: " + on_v41);
    const std::string wrong = load_header(path, true, 5120, "n");
    check(wrong.find("unsupported metadata") != std::string::npos, "a V4.1 header with V4's layout is refused: " + wrong);
}
}  // namespace

int main() {
    ggml_backend_load_all();
    test_query_block_sizes();
    test_image_block_tokens();
    test_projector_mismatch();
    test_projector_metadata();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    test_blocked_attention(cpu, "cpu", true);
    ggml_backend_free(cpu);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
        if (!gpu) continue;
        test_blocked_attention(gpu, ggml_backend_dev_name(dev), false);
        ggml_backend_free(gpu);
        break;
    }
    if (g_failures) { std::fprintf(stderr, "%d failure(s)\n", g_failures); return 1; }
    std::printf("PASS V4.1 blocked attention = whole attention (CPU exact, GPU to rounding), "
                "query blocks, image block tokens, projector metadata and family/checkpoint checks\n");
    return 0;
}
