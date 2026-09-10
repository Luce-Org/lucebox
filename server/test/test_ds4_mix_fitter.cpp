#define main ds4_mix_converter_main
#include "../tools/ds4_mix_converter/ds4_mix_converter.cpp"
#undef main

int main(int argc, char ** argv) {
    try {
        for (float value : {0.0f, -1.0f, 1.0f}) {
            HistogramFitter fitter;
            float values[16];
            std::fill_n(values, 16, value);
            fitter.add_half(values, nullptr);
            for (int levels : {4, 8}) {
                const auto books = fitter.fit(levels);
                if (books != fitter.fit(levels)) fail("fit is nondeterministic");
                for (int population = 0; population < 2; ++population) {
                    float previous = -std::numeric_limits<float>::infinity();
                    for (int i = 0; i < levels; ++i) {
                        float v = bf16_to_float(books[population*levels+i]);
                        if (!std::isfinite(v) || v < -1 || v > 1 || v <= previous)
                            fail("codebook is not finite, bounded, and strictly ordered");
                        previous = v;
                    }
                }
                float x[32] = {};
                block_rocmfp2 q2;
                block_rocmfp3 q3;
                if (levels == 4 && !rocmfpx_quantize_row_fp2_mix_ref(x, &q2, 32, books.data(), nullptr))
                    fail("repaired fp2 codebook rejected by encoder");
                if (levels == 8 && !rocmfpx_quantize_row_fp3_mix_ref(x, &q3, 32, books.data(), nullptr))
                    fail("repaired fp3 codebook rejected by encoder");
            }
        }
        bool repaired = false;
        const auto close = HistogramFitter::round_centers({-1.0f, 0.5f, 0.5001f, 1.0f}, repaired);
        if (!repaired || bf16_to_float(close[2]) <= bf16_to_float(close[1]))
            fail("BF16 rounding collision was not repaired");
        repaired = false;
        const auto distinct = HistogramFitter::round_centers({-1.0f, -0.5f, 0.5f, 1.0f}, repaired);
        if (repaired || distinct != std::vector<uint16_t>{0xbf80, 0xbf00, 0x3f00, 0x3f80})
            fail("nondegenerate centers changed");
        HistogramFitter stamped;
        float zero[16] = {};
        stamped.add_half(zero, nullptr);
        std::vector<std::string> stamps;
        stamped.fit(4, "layer=42 expert=164 down", &stamps);
        if (stamps.size() != 2 || stamps[0].find("layer=42 expert=164 down") == std::string::npos)
            fail("repair stamp missing expert identity");
        bool rejected = false;
        try { HistogramFitter().fit(4); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) fail("empty histogram accepted");
        if (argc != 1 && argc != 5) fail("usage: test_ds4_mix_fitter [SOURCE IMATRIX LAYER EXPERT]");
        if (argc == 5) {
            SafeTensorSet source(argv[1]);
            std::optional<Imatrix> imatrix = load_imatrix(argv[2]);
            const int layer = parse_nonnegative(argv[3], "layer");
            const int expert = parse_nonnegative(argv[4], "expert");
            HistogramFitter gate_up, down;
            for (const auto & recipe : kExpertRecipes) {
                const auto shape = validate_expert_source(source, layer, expert, recipe);
                const auto * importance = require_imatrix(imatrix, target_expert_name(layer, recipe), shape.in);
                add_expert_to_fitter(source, layer, expert, recipe, importance,
                    recipe.books == BookSource::GateUpJoint ? gate_up : down);
            }
            const std::string label = "layer=" + std::to_string(layer) + " expert=" + std::to_string(expert);
            gate_up.fit(kGuLevels, label + " gate_up");
            down.fit(kP4Levels, label + " down");
            std::cout << "PASS: source replay " << label << "\n";
        }
        std::cout << "PASS: degenerate fitter, BF16 ordering, determinism, codec acceptance, empty rejection\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
