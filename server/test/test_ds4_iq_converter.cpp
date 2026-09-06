#include <sstream>
#define main ds4_mix_converter_main
#include "../tools/ds4_mix_converter/ds4_mix_converter.cpp"
#undef main

template<class F> void must_fail(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception &) { rejected = true; }
    if (!rejected) fail("expected rejection");
}

void test_source_artifact() {
    char pattern[]="/tmp/ds4-iq-source-XXXXXX";
    if(!::mkdtemp(pattern)) fail("mkdtemp failed");
    const fs::path root=pattern;
    struct Cleanup {fs::path path; ~Cleanup(){std::error_code ec;fs::remove_all(path,ec);}} cleanup{root};
    json config;
    for(const char * key:{"num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim",
        "qk_rope_head_dim","q_lora_rank","o_lora_rank","o_groups","num_experts_per_tok","n_shared_experts",
        "moe_intermediate_size","num_hash_layers","sliding_window","index_n_heads","index_head_dim",
        "index_topk","hc_mult","hc_sinkhorn_iters"}) config[key]=1;
    config["n_routed_experts"]=2;config["hidden_size"]=256;config["vocab_size"]=2;
    std::ofstream(root/"config.json")<<config;
    std::ofstream(root/"tokenizer.json")<<json{{"model",{{"vocab",{{"a",0},{"b",1}}},{"merges",json::array()}}}};
    std::ofstream(root/"tokenizer_config.json")<<json::object();
    json header,index={{"weight_map",json::object()}};
    std::vector<uint8_t> payload;
    auto add=[&](const std::string & name,const std::string & dtype,const std::vector<int64_t>& shape,std::vector<uint8_t> bytes){
        const size_t start=payload.size();payload.insert(payload.end(),bytes.begin(),bytes.end());
        header[name]={{"dtype",dtype},{"shape",shape},{"data_offsets",{start,payload.size()}}};
        index["weight_map"][name]="model.safetensors";
    };
    std::vector<uint8_t> bf16(1024);
    for(size_t i=0;i<512;++i){uint16_t v=float_to_bf16(std::sin(float(i)));std::memcpy(bf16.data()+i*2,&v,2);}
    add("embed.weight","BF16",{2,256},bf16);add("head.weight","BF16",{2,256},bf16);
    for(const char * name:{"norm.weight","hc_head_base","hc_head_fn","hc_head_scale"})
        add(name,"F32",{4},std::vector<uint8_t>(16));
    add("layers.0.attn.wq_a.weight","F8_E4M3",{2,256},std::vector<uint8_t>(512,0x38));
    add("layers.0.attn.wq_a.scale","F8_E8M0",{1,2},{127,128});
    add("layers.0.ffn.gate.tid2eid","I64",{2},std::vector<uint8_t>(16));
    add("vision.test.weight","BF16",{2,3},std::vector<uint8_t>(12,0x3f));
    for(uint32_t e=0;e<2;++e) for(const auto & recipe:kExpertRecipes) {
        auto packed=std::vector<uint8_t>(256);
        for(size_t i=0;i<packed.size();++i) packed[i]=uint8_t(i+e*17);
        add(source_expert_name(0,e,recipe,"weight"),"I8",{2,128},packed);
        add(source_expert_name(0,e,recipe,"scale"),"F8_E8M0",{2,8},std::vector<uint8_t>(16,127));
    }
    std::ofstream(root/"model.safetensors.index.json")<<index;
    const std::string hdr=header.dump();const uint64_t len=hdr.size();
    {std::ofstream file(root/"model.safetensors",std::ios::binary);file.write((const char*)&len,8);file<<hdr;
     file.write((const char*)payload.data(),payload.size());}
    const fs::path importance=root/"importance.dat";
    {std::ofstream f(importance,std::ios::binary);auto i32=[&](int32_t n){f.write((const char*)&n,4);};
     i32(3);for(const auto & recipe:kExpertRecipes){auto name=target_expert_name(0,recipe);i32(name.size());f<<name;i32(1);i32(512);
        for(int i=0;i<512;++i){float v=1.0f+float(i%17);f.write((const char*)&v,4);}}}
    SafeTensorSet source(root);
    Options options;options.recipe="iq85";options.input=root;options.output=root/"serial.gguf";
    options.imatrix=importance;options.imatrix_provenance="transferred-text-calibration";
    std::ostringstream captured;
    struct CoutRestore {std::streambuf * original;~CoutRestore(){std::cout.rdbuf(original);}} restore{std::cout.rdbuf(captured.rdbuf())};
    options.plan_only=true;run_iq85(options,source,1,2);
    if(fs::exists(options.output)||fs::exists(options.output.string()+".partial")) fail("plan-only wrote model data");
    const auto plan=json::parse(captured.str());captured.str("");captured.clear();
    unsigned q8=0;for(const auto & row:plan["tensors"]) if(row["type"]==int(GGML_TYPE_Q8_0))++q8;
    if(q8!=3) fail("actual source plan failed dense allowlist");
    options.plan_only=false;run_iq85(options,source,1,2);
    if(fs::file_size(options.output)!=plan["file_bytes"].get<uint64_t>()) fail("plan size differs from serialized artifact");
    const auto serial=read_file(options.output);
    must_fail([&]{run_iq85(options,source,1,2);});
    if(read_file(options.output)!=serial) fail("existing artifact was overwritten");
    options.output=root/"parallel.gguf";options.encode_threads=8;run_iq85(options,source,1,2);
    if(read_file(options.output)!=serial) fail("full original-source serial/parallel GGUF differs");
    // Already-open source metadata does not mask a later short read; no final artifact may appear.
    fs::resize_file(root/"model.safetensors",8+len+payload.size()-1);
    options.output=root/"short-read.gguf";
    must_fail([&]{run_iq85(options,source,1,2);});
    if(fs::exists(options.output)) fail("failed conversion published final artifact");
}

void test_fp8_dense_analytic() {
    // Independent analytical decoding oracle: these E4M3 bytes represent exactly
    // +1, -1, +2 and +0.5. E8M0 127/128 mean scale 1/2. Both tile axes cross 128.
    constexpr size_t rows=129,cols=256;
    const std::array<uint8_t,4> codes={0x38,0xb8,0x40,0x30};
    const std::array<float,4> decoded={1.0f,-1.0f,2.0f,0.5f};
    const std::array<uint8_t,4> scales={127,128,128,127};
    std::vector<uint8_t> payload(rows*cols);
    std::vector<float> expected_values(rows*cols);
    for(size_t row=0;row<rows;++row) for(size_t col=0;col<cols;++col) {
        const size_t choice=(row+col)%4;
        payload[row*cols+col]=codes[choice];
        const float scale=((row>=128) != (col>=128)) ? 2.0f : 1.0f;
        expected_values[row*cols+col]=decoded[choice]*scale;
    }
    char pattern[]="/tmp/ds4-iq-fp8-XXXXXX";
    const int fd=::mkstemp(pattern);
    if(fd<0) fail("FP8 fixture mkstemp failed");
    struct Cleanup {const char * path;~Cleanup(){::unlink(path);}} cleanup{pattern};
    if(::write(fd,payload.data(),payload.size())!=ssize_t(payload.size()) ||
       ::write(fd,scales.data(),scales.size())!=ssize_t(scales.size())) {
        ::close(fd);fail("FP8 fixture write failed");
    }
    ::close(fd);
    StEntry weight;weight.name="layers.0.attn.wq_a.weight";weight.dtype="F8_E4M3";
    weight.path=pattern;weight.shape={rows,cols};weight.size=payload.size();
    StEntry scale;scale.name="layers.0.attn.wq_a.scale";scale.dtype="F8_E8M0";
    scale.path=pattern;scale.shape={2,2};scale.offset=payload.size();scale.size=scales.size();
    TensorSpec spec;spec.name="blk.0.attn_q_a.weight";spec.source=&weight;spec.scale=&scale;
    spec.ne=reverse_shape(weight);spec.type=GGML_TYPE_Q8_0;spec.producer=Producer::DenseFp8;
    std::unique_ptr<FILE,decltype(&std::fclose)> out(std::tmpfile(),std::fclose);
    if(!out) fail("FP8 output tmpfile failed");
    iq85_write_dense(out.get(),spec);
    std::vector<uint8_t> expected(ggml_row_size(GGML_TYPE_Q8_0,cols)*rows),actual(expected.size());
    if(ggml_quantize_chunk(GGML_TYPE_Q8_0,expected_values.data(),expected.data(),0,rows,cols,nullptr)!=expected.size())
        fail("analytical Q8 expected byte size mismatch");
    std::rewind(out.get());
    if(std::fread(actual.data(),1,actual.size(),out.get())!=actual.size() || actual!=expected || std::fgetc(out.get())!=EOF)
        fail("FP8 Q8 differs from independent analytical source/scales oracle");
}

int main() {
    try {
        StEntry source;
        source.name = "layers.0.attn.wq_a.weight";
        source.shape = {2,256};
        TensorSpec spec;
        spec.source = &source; spec.type = GGML_TYPE_BF16; spec.ne = reverse_shape(source);
        for (const char * name : {"blk.0.attn_q_a.weight", "output.weight", "token_embd.weight"}) {
            spec.name = name;
            if (!iq85_dense(spec)) fail("eligible dense tensor excluded");
        }
        for (const char * name : {"blk.0.hc_attn_fn.weight", "blk.0.ffn_gate_inp.weight",
                "blk.0.indexer.proj.weight", "blk.0.attn_compressor_kv.weight", "vision.blocks.0.attn.wqkv.weight"}) {
            spec.name = name;
            if (iq85_dense(spec)) fail("protected tensor quantized");
        }
        spec.name = "output.weight"; source.shape={256}; spec.ne=reverse_shape(source);
        if (iq85_dense(spec)) fail("vector quantized");

        // Actual dense BF16 source -> Q8 bytes against canonical row encoding.
        char pattern[] = "/tmp/ds4-iq-test-XXXXXX";
        const int fd = ::mkstemp(pattern);
        if (fd < 0) fail("mkstemp failed");
        source.path = pattern; source.dtype = "BF16"; source.shape = {2,256}; source.size = 1024;
        std::vector<uint16_t> bf16(512);
        std::vector<float> values(512);
        for (size_t i=0;i<512;++i) { bf16[i] = float_to_bf16(std::sin(float(i))*2); values[i] = bf16_to_float(bf16[i]); }
        if (::write(fd,bf16.data(),1024)!=1024) fail("fixture write failed");
        ::close(fd);
        spec.ne = {256,2}; spec.type = GGML_TYPE_Q8_0; spec.producer = Producer::Raw;
        std::unique_ptr<FILE,decltype(&std::fclose)> output(std::tmpfile(),std::fclose);
        if (!output) fail("tmpfile failed");
        iq85_write_dense(output.get(),spec);
        std::vector<uint8_t> expected(ggml_row_size(GGML_TYPE_Q8_0,256)*2), actual(expected.size());
        ggml_quantize_chunk(GGML_TYPE_Q8_0,values.data(),expected.data(),0,2,256,nullptr);
        std::rewind(output.get());
        if (std::fread(actual.data(),1,actual.size(),output.get())!=actual.size() || actual!=expected)
            fail("source-to-Q8 bytes differ from canonical encoder");
        ::unlink(pattern);

        spec.producer=Producer::Expert; spec.name="blk.0.ffn_gate_exps.weight"; spec.ne={256,2,1};
        std::optional<Imatrix> imatrix=Imatrix{{spec.name,{1,std::vector<float>(256,1)}}};
        iq85_validate_importance({spec},imatrix,"uniform-unvalidated");
        must_fail([&]{iq85_validate_importance({spec},imatrix,"activation-derived");});
        std::fill(imatrix->at(spec.name).values.begin(),imatrix->at(spec.name).values.end(),0);
        must_fail([&]{iq85_validate_importance({spec},imatrix,"uniform-unvalidated");});
        imatrix->at(spec.name).values.resize(256*3);
        for(size_t i=0;i<256*3;++i)imatrix->at(spec.name).values[i]=float(i+1);
        if(iq85_importance(imatrix,spec,2,3)[0]!=513) fail("wrong per-expert importance slice");
        must_fail([&]{iq85_importance(imatrix,spec,0,2);});
        must_fail([&]{iq85_importance(imatrix,spec,3,3);});

        // Canonical IQ rows encoded concurrently must equal serial bytes, including
        // nonuniform importance and a batch tail. Shared lookup tables are initialized first.
        std::vector<float> importance(256);
        for(size_t i=0;i<256;++i) importance[i]=0.25f+float(i%13);
        for(auto type:{GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS}) {
            ggml_quantize_init(type);
            auto encode=[&](uint32_t e) {
                auto v=values; for(auto & x:v) x+=float(e)*0.015625f;
                ds4_mix_detail::EncodedExpert bytes(ggml_row_size(type,256)*2);
                if(ggml_quantize_chunk(type,v.data(),bytes.data(),0,2,256,importance.data())!=bytes.size())
                    fail("IQ size mismatch");
                return bytes;
            };
            std::vector<uint8_t> serial,parallel,parallel16;
            auto writer=[](std::vector<uint8_t>& out){return [&out](uint32_t,const auto & b){out.insert(out.end(),b.begin(),b.end());};};
            ds4_mix_detail::ordered_expert_batches(17,1,ggml_row_size(type,256)*2,encode,writer(serial));
            ds4_mix_detail::ordered_expert_batches(17,8,ggml_row_size(type,256)*2,encode,writer(parallel));
            ds4_mix_detail::ordered_expert_batches(17,16,ggml_row_size(type,256)*2,encode,writer(parallel16));
            if(serial!=parallel || serial!=parallel16) fail("IQ parallel output differs");
        }
        must_fail([&]{ds4_mix_detail::checked_encoded_size(UINT64_MAX,2,16);});
        must_fail([&]{ds4_mix_detail::checked_encoded_size(1,1,17);});
        std::vector<std::string> args={"converter","--input","/unused","--output","/unused/new.gguf",
            "--recipe","iq85","--imatrix","/unused/importance.dat","--imatrix-provenance","uniform-unvalidated"};
        auto parse=[&] {std::vector<char*> argv;for(auto & arg:args)argv.push_back(arg.data());return parse_options(argv.size(),argv.data());};
        if(parse().encode_threads!=1) fail("default encode thread count changed");
        args.push_back("--encode-threads");args.push_back("16");
        if(parse().encode_threads!=16) fail("CLI rejects sixteen encoder workers");
        args.back()="17";must_fail([&]{parse();});
        unsigned writes=0;
        must_fail([&]{ds4_mix_detail::ordered_expert_batches(17,8,1,
            [](uint32_t e){if(e==3) fail("injected worker failure");return ds4_mix_detail::EncodedExpert(1);},
            [&](uint32_t,const auto&){++writes;});});
        if(writes!=3) fail("publication continued after worker failure");
        test_fp8_dense_analytic();
        test_source_artifact();
        std::cout<<"PASS: dense preservation/canonical encoding, imatrix provenance, IQ parallel determinism, failure bounds\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
