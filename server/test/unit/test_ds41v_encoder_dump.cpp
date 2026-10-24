// DeepSeek V4.1 image path on a real projector and image, dumped for a
// comparison with DeepSeek's reference code (scripts/ds41v_reference_check.py):
// the resize plan, the BF16 patches, the ViT features and the aligner rows.
//
//   LUCE_DS41V_MMPROJ=<V4.1 projector> LUCE_DS41V_IMAGE=<JPEG or PNG>
//   LUCE_DS41V_DUMP=<output directory> [LUCE_DS41V_DEVICE=cpu] test_ds41v_encoder_dump
//
// Without the variables it reports itself skipped and passes.
#include "common/vision/image_decode.h"
#include "deepseek4/deepseek4_vision.h"
#include "deepseek4/deepseek4_vision_preprocess.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace luce::vision;

static bool write_bytes(const std::string & path, const void * data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char *>(data), std::streamsize(size));
    return bool(out);
}

int main() {
    const char * mmproj = std::getenv("LUCE_DS41V_MMPROJ");
    const char * image_path = std::getenv("LUCE_DS41V_IMAGE");
    const char * dump = std::getenv("LUCE_DS41V_DUMP");
    if (!mmproj || !image_path || !dump) {
        std::printf("test_ds41v_encoder_dump skipped (LUCE_DS41V_MMPROJ, LUCE_DS41V_IMAGE, LUCE_DS41V_DUMP unset)\n");
        return 0;
    }
    std::ifstream in(image_path, std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto decoded = decode_image({bytes.data(), bytes.size()});
    if (!decoded) { std::fprintf(stderr, "decode failed: %s\n", decoded.status.message.c_str()); return 1; }
    const auto prepared = preprocess_rgb(decoded.image.view(), 0, v41_preprocess_config());
    if (!prepared) { std::fprintf(stderr, "preprocess failed: %s\n", prepared.status.message.c_str()); return 1; }
    const ResizePlan & plan = prepared.image.plan;

    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    const char * device = std::getenv("LUCE_DS41V_DEVICE");
    if (!device || std::strcmp(device, "cpu") != 0) {
        for (size_t i = 0; i < ggml_backend_dev_count() && !backend; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) backend = ggml_backend_dev_init(dev, nullptr);
        }
    }
    if (!backend) backend = ggml_backend_cpu_init();
    VisionRuntime runtime;
    std::string error;
    if (!runtime.load(mmproj, backend, 5120, 129280, error)) {
        std::fprintf(stderr, "projector load failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<float> patches(prepared.image.patches_bf16.size());
    for (size_t i = 0; i < patches.size(); ++i) {
        const std::uint32_t bits = std::uint32_t(prepared.image.patches_bf16[i]) << 16;
        std::memcpy(&patches[i], &bits, sizeof(float));
    }
    VisionOutput output;
    const auto t0 = std::chrono::steady_clock::now();
    if (!runtime.encode(patches, {int(plan.vit_rows), int(plan.vit_cols)}, output, error, /*retain_features=*/true)) {
        std::fprintf(stderr, "encode failed: %s\n", error.c_str());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const std::string dir = dump;
    bool ok = write_bytes(dir + "/patches.bf16", prepared.image.patches_bf16.data(), prepared.image.patches_bf16.size() * 2);
    ok = ok && write_bytes(dir + "/features.f32", output.features.data(), output.features.size() * sizeof(float));
    ok = ok && write_bytes(dir + "/embeddings.f32", output.embeddings.data(), output.embeddings.size() * sizeof(float));
    std::ofstream meta(dir + "/plan.json");
    meta << "{\"resized_width\": " << plan.resized_width << ", \"resized_height\": " << plan.resized_height
         << ", \"vit_rows\": " << plan.vit_rows << ", \"vit_cols\": " << plan.vit_cols
         << ", \"aligner_rows\": " << plan.aligner_rows << ", \"aligner_cols\": " << plan.aligner_cols
         << ", \"embedding_rows\": " << output.rows << ", \"embedding_columns\": " << output.columns
         << ", \"device\": \"" << ggml_backend_name(backend) << "\", \"encode_ms\": " << ms << "}\n";
    const std::string backend_name = ggml_backend_name(backend);
    ggml_backend_free(backend);
    if (!ok || !meta) { std::fprintf(stderr, "could not write the dump\n"); return 1; }
    std::printf("dumped %u x %u patches, %d x %d embeddings in %.0f ms on %s\n", plan.vit_rows, plan.vit_cols,
                output.rows, output.columns, ms, backend_name.c_str());
    return 0;
}
