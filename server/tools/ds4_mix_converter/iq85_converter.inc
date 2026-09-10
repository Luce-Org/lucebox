// Included inside the converter's anonymous namespace. Reuses only source decoding
// and metadata helpers; the existing MIX calibration/encoding path is unchanged.
bool iq85_dense(const TensorSpec & spec) {
    if (!spec.source || spec.type != GGML_TYPE_BF16 || spec.source->shape.size() != 2 ||
        spec.ne.size() < 2 || spec.ne[0] % 32) return false;
    if (spec.name == "token_embd.weight" || spec.name == "output.weight") return true;
    const auto parsed = parse_layer_name(spec.source->name);
    if (!parsed) return false;
    static const std::set<std::string> leaves = {
        "attn_kv.weight", "attn_output_a.weight", "attn_output_b.weight",
        "attn_q_a.weight", "attn_q_b.weight", "ffn_down_shexp.weight",
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight"};
    const size_t dot = spec.name.find('.', 4);
    return dot != std::string::npos && leaves.count(spec.name.substr(dot + 1));
}

std::vector<TensorSpec> iq85_plan(const SafeTensorSet & source,
        const std::vector<LayerCalibration> & layout, uint32_t experts, bool smoke) {
    auto plan = make_plan(source, layout, experts, smoke);
    for (auto & spec : plan) {
        if (spec.producer == Producer::Expert) {
            spec.type = spec.recipe->surface == Surface::Down ? GGML_TYPE_IQ2_XS : GGML_TYPE_IQ2_XXS;
        } else if (iq85_dense(spec)) {
            spec.type = GGML_TYPE_Q8_0;
        }
    }
    return plan;
}

const float * iq85_importance(const std::optional<Imatrix> & imatrix,
        const TensorSpec & spec, uint32_t expert, uint32_t source_experts) {
    if (!imatrix) fail("iq85 requires importance weights");
    const auto it = imatrix->find(spec.name);
    if (it == imatrix->end()) fail("imatrix missing " + spec.name);
    const size_t in = spec.ne[0];
    const size_t full = checked_mul(in, source_experts, "per-expert importance dimensions");
    const auto & values = it->second.values;
    if (values.size() != in && values.size() != full)
        fail("iq85 importance dimensions must be input width or input width * SOURCE expert count: " + spec.name);
    if (expert >= source_experts) fail("iq85 importance expert out of range");
    return values.data() + (values.size() == in ? 0 : size_t(expert)*in);
}

void iq85_validate_importance(const std::vector<TensorSpec> & plan,
        const std::optional<Imatrix> & imatrix, const std::string & provenance, uint32_t source_experts = 1) {
    bool any_nonuniform = false;
    for (const auto & spec : plan) {
        if (spec.producer != Producer::Expert) continue;
        for (uint32_t expert = 0; expert < uint32_t(spec.ne[2]); ++expert) {
            const float * values = iq85_importance(imatrix, spec, expert, source_experts);
            if (std::none_of(values, values + spec.ne[0], [](float x) { return x > 0; }))
                fail("iq85 requires nonzero importance for every selected expert: " + spec.name);
            any_nonuniform |= std::any_of(values, values + spec.ne[0], [&](float x) { return x != values[0]; });
        }
    }
    if (provenance != "uniform-unvalidated" && !any_nonuniform)
        fail("all importance rows are uniform; label these uniform-unvalidated, not activation-derived");
    std::cerr << "[iq85] imatrix provenance=" << provenance
              << "; this label is operator supplied, not a quality qualification\n";
}

void iq85_write_experts(FILE * out, const SafeTensorSet & source, const TensorSpec & spec,
        uint32_t experts, const std::optional<Imatrix> & imatrix, unsigned workers) {
    const size_t row_bytes = ggml_row_size(spec.type, spec.ne[0]);
    const size_t expert_bytes = ds4_mix_detail::checked_encoded_size(row_bytes, spec.ne[1], workers);
    const uint32_t source_experts = config_u32(source.config(), "n_routed_experts");
    // Initialize shared, immutable IQ lookup tables before any worker launches.
    ggml_quantize_init(spec.type);
    ds4_mix_detail::ordered_expert_batches(experts, workers, expert_bytes,
        [&](uint32_t expert) {
            const float * importance = iq85_importance(imatrix, spec, expert, source_experts);
            const auto shape = validate_expert_source(source, spec.layer, expert, *spec.recipe);
            if (shape.in != spec.ne[0] || shape.out != spec.ne[1]) fail("iq85 expert shape drift");
            const auto & w = source.at(source_expert_name(spec.layer, expert, *spec.recipe, "weight"));
            const auto & s = source.at(source_expert_name(spec.layer, expert, *spec.recipe, "scale"));
            OpenTensorPair input(w, s);
            std::vector<uint8_t> packed, scales;
            std::vector<float> values;
            ds4_mix_detail::EncodedExpert bytes(expert_bytes);
            for (uint32_t row = 0; row < shape.out; ++row) {
                decode_expert_row(input, row, shape.in, packed, scales, values);
                const size_t n = ggml_quantize_chunk(spec.type, values.data(), bytes.data() + row*row_bytes,
                    0, 1, shape.in, importance);
                if (n != row_bytes) fail("iq85 encoder returned wrong row size");
            }
            return bytes;
        }, [&](uint32_t expert, const ds4_mix_detail::EncodedExpert & bytes) {
            fwrite_exact(out, bytes.data(), bytes.size(), spec.name);
            std::cerr << "[iq85 encode] " << spec.name << " expert " << expert + 1 << '/' << experts << '\n';
        });
}

void iq85_write_dense(FILE * out, const TensorSpec & spec) {
    const auto & w = *spec.source;
    const size_t cols = spec.ne[0], rows = spec.ne[1];
    FileDescriptor wf(w.path);
    std::vector<float> values(cols);
    std::vector<uint8_t> input(cols * (spec.producer == Producer::DenseFp8 ? 1 : 2));
    std::vector<uint8_t> scales;
    if (spec.producer == Producer::DenseFp8) {
        FileDescriptor sf(spec.scale->path);
        scales.resize(spec.scale->size);
        pread_exact(sf.fd, scales.data(), scales.size(), spec.scale->offset, spec.scale->name);
    } else if (w.dtype != "BF16") fail("iq85 unsupported dense source " + w.name);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, cols);
    std::vector<uint8_t> encoded(row_bytes);
    for (size_t row = 0; row < rows; ++row) {
        pread_exact(wf.fd, input.data(), input.size(), w.offset + row*input.size(), w.name);
        for (size_t col = 0; col < cols; ++col) {
            if (spec.producer == Producer::DenseFp8) {
                values[col] = fp8_e4m3fn(input[col]) * fp8_e8m0(scales[(row/128)*spec.scale->shape[1] + col/128]);
            } else {
                uint16_t b;
                std::memcpy(&b, input.data() + col*2, 2);
                values[col] = bf16_to_float(b);
            }
            if (!std::isfinite(values[col])) fail("non-finite iq85 dense source " + w.name);
        }
        if (ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), encoded.data(), 0, 1, cols, nullptr) != row_bytes)
            fail("Q8 encoder returned wrong row size");
        fwrite_exact(out, encoded.data(), encoded.size(), spec.name);
    }
}

void run_iq85(const Options & options, const SafeTensorSet & source, uint32_t layers, uint32_t experts) {
    const auto layout = validate_input_layout(source, layers, experts);
    const auto plan = iq85_plan(source, layout, experts, options.experts_only);
    const std::optional<Imatrix> imatrix = load_imatrix(*options.imatrix);
    iq85_validate_importance(plan, imatrix, options.imatrix_provenance, config_u32(source.config(), "n_routed_experts"));
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(gguf_init_empty(), gguf_free);
    if (!ctx) fail("gguf_init_empty failed");
    set_model_metadata(ctx.get(), source, layers, experts, false, options.experts_only, {});
    for (const char * key : {"deepseek4.p4mix.sidecar", "deepseek4.mix.calibration",
            "deepseek4.mix.lower_quality_absmax_only", "deepseek4.mix.experts_only_smoke_artifact"})
        gguf_remove_key(ctx.get(), key);
    gguf_set_val_str(ctx.get(), "general.name", "DeepSeek-V4-Flash-Vision iq85 experimental");
    gguf_set_val_u32(ctx.get(), "general.file_type", GGML_FTYPE_MOSTLY_IQ2_XXS);
    gguf_set_val_str(ctx.get(), "deepseek4.quant.recipe", "iq85-v1: gate/up IQ2_XXS; down IQ2_XS; selected dense Q8_0");
    gguf_set_val_str(ctx.get(), "deepseek4.quant.imatrix_provenance", options.imatrix_provenance.c_str());
    gguf_set_val_str(ctx.get(), "deepseek4.quant.imatrix_source", options.imatrix->filename().c_str());
    gguf_set_val_bool(ctx.get(), "deepseek4.quant.quality_validated", false);
    gguf_set_val_bool(ctx.get(), "deepseek4.quant.experts_only_smoke_artifact", options.experts_only);
    std::vector<std::unique_ptr<ggml_tensor>> descriptors;
    uint64_t data_bytes = 0;
    json rows = json::array();
    for (const auto & spec : plan) {
        uint64_t n = 1;
        for (auto dim : spec.ne) {
            if (dim <= 0) fail("iq85 invalid tensor dimension");
            n = checked_mul(n, dim, "iq85 descriptor element count");
        }
        if (n > size_t(-1)/ggml_type_size(spec.type)) fail("iq85 tensor exceeds host size bounds");
        descriptors.push_back(make_tensor_descriptor(spec));
        gguf_add_tensor(ctx.get(), descriptors.back().get());
        const int64_t id = gguf_find_tensor(ctx.get(), spec.name.c_str());
        const uint64_t offset = gguf_get_tensor_offset(ctx.get(), id), bytes = gguf_get_tensor_size(ctx.get(), id);
        if (bytes > UINT64_MAX - offset) fail("iq85 plan size overflow");
        data_bytes = offset + bytes;
        rows.push_back({{"name",spec.name},{"type",int(spec.type)},{"bytes",bytes}});
    }
    const uint64_t meta_bytes = gguf_get_meta_size(ctx.get());
    if (data_bytes > UINT64_MAX - meta_bytes) fail("iq85 file size overflow");
    const uint64_t total = meta_bytes + data_bytes;
    std::cout << json({{"recipe","iq85-v1"},{"file_bytes",total},{"decimal_GB",double(total)/1e9},
        {"metadata_bytes",meta_bytes},{"tensor_bytes_with_padding",data_bytes},
        {"imatrix_provenance",options.imatrix_provenance},{"quality_validated",false},
        {"experts_only",options.experts_only},{"tensors",rows}}).dump(2) << '\n';
    if (options.plan_only || options.validate_input_only) return;
    if (fs::exists(options.output) || fs::exists(options.output.string()+".gumix.bin")) fail("iq85 output exists");
    if (!options.output.parent_path().empty()) fs::create_directories(options.output.parent_path());
    const fs::path temporary = options.output.string() + ".partial";
    const int fd = ::open(temporary.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (fd < 0) fail("cannot exclusively create " + temporary.string());
    FILE * raw = ::fdopen(fd, "wb");
    if (!raw) { ::close(fd); fail("fdopen failed"); }
    std::unique_ptr<FILE, decltype(&std::fclose)> out(raw, std::fclose);
    std::vector<uint8_t> metadata(meta_bytes);
    gguf_get_meta_data(ctx.get(), metadata.data());
    fwrite_exact(out.get(), metadata.data(), metadata.size(), "iq85 metadata");
    std::array<uint8_t, kAlignment> zero{};
    for (const auto & spec : plan) {
        const int64_t id = gguf_find_tensor(ctx.get(), spec.name.c_str());
        const uint64_t expected = meta_bytes + gguf_get_tensor_offset(ctx.get(), id);
        const off_t position = ::ftello(out.get());
        if (position < 0 || uint64_t(position) > expected || expected - position >= kAlignment)
            fail("iq85 stream offset mismatch");
        fwrite_exact(out.get(), zero.data(), expected-position, "iq85 padding");
        if (spec.producer == Producer::Expert) iq85_write_experts(out.get(), source, spec, experts, imatrix, options.encode_threads);
        else if (spec.type == GGML_TYPE_Q8_0) iq85_write_dense(out.get(), spec);
        else if (spec.producer == Producer::Raw) copy_raw(out.get(), *spec.source);
        else if (spec.producer == Producer::DenseFp8) write_dense_fp8(out.get(), *spec.source, *spec.scale);
        else if (spec.producer == Producer::Int64ToInt32) write_int64_to_int32(out.get(), *spec.source);
        else fail("iq85 unknown producer");
        const off_t after = ::ftello(out.get());
        if (after < 0 || uint64_t(after) != expected + gguf_get_tensor_size(ctx.get(), id)) fail("iq85 producer size mismatch");
    }
    if (std::fflush(out.get()) || ::fsync(::fileno(out.get()))) fail("iq85 flush/fsync failed");
    if (std::fclose(out.release())) fail("iq85 close failed");
    // Validate the partial file BEFORE publication; link is atomic and refuses overwrite.
    gguf_init_params params = {true, nullptr};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> parsed(gguf_init_from_file(temporary.c_str(), params), gguf_free);
    if (!parsed || fs::file_size(temporary) != total || gguf_get_n_tensors(parsed.get()) != int64_t(plan.size()))
        fail("iq85 completed file size/header verification failed");
    FileDescriptor input(temporary);
    for (const auto & spec : plan) {
        const int64_t id = gguf_find_tensor(parsed.get(), spec.name.c_str());
        if (id < 0 || gguf_get_tensor_type(parsed.get(), id) != spec.type) fail("iq85 verification type mismatch");
        if (spec.producer == Producer::Raw && spec.type != GGML_TYPE_Q8_0)
            compare_raw_passthrough(input.fd, gguf_get_data_offset(parsed.get()) + gguf_get_tensor_offset(parsed.get(), id), *spec.source);
    }
    if (::link(temporary.c_str(), options.output.c_str())) fail("iq85 atomic no-overwrite publication failed");
    if (::unlink(temporary.c_str())) fail("iq85 published but partial link cleanup failed");
    std::cerr << "[iq85 done] verified/published " << total << " bytes (" << double(total)/1e9 << " decimal GB)\n";
}
