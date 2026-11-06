// Bit-for-bit parity of the MMQ tile twins (ggml-cuda/mmq_tile_twins.h)
// with their parent instances, on the GPU. SKIP (77) without one.
//
// Each arm runs in a child process (this binary with --child), which turns
// the twins on or off with ggml_backend_cuda_set_mmq_twins_override as the
// parent asks (LUCE_TEST_MMQ_TWINS, a test-only message):
//   base  both twins off (the parent tiles)
//   ud    only the 64-row twin on
//   2w    only the 32-row twin on
//   both  the defaults (both on)
// Every child multiplies the same deterministic weights (random blocks of each
// type with finite scales) by the same F32 activations and prints one line per
// case: the FNV-1a hash of the output bytes, and how many MMQ calls and twin
// launches (ggml_backend_cuda_get_mmq_twin_launch_count) the case made.
//
// The parent requires, per case: every arm's hash equals base's; every case
// reached MMQ exactly once; base launched no twin; an arm launched the twin
// exactly where mmq-tile-twin-gate.h says it must (RDNA4, 16 or fewer columns, a
// type with a twin) and nowhere else. On a GPU that is not RDNA4 no twin may
// engage; the hashes are still compared, then the test reports SKIP.
//
//   test_mmq_tile_twins            all arms, compare
//   test_mmq_tile_twins --child    one arm, the environment decides

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cuda/mmq-tile-twin-gate.h"

#include <hip/hip_runtime_api.h>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

extern "C" char ** environ;

namespace {

struct Shape {
    int64_t k;  // row length
    int64_t m;  // output rows
};

struct Case {
    ggml_type type;
    Shape     shape;
    int64_t   n;  // columns
};

// The real shapes of Qwen3.8-27B UD-IQ4_XS plus gate edges: 6176 rows (just
// over the 2-warp limit, a partial 64-row tile), 6208 (a partial 128-row
// tile), 96 rows (partial for 64, three 32-row tiles) and 48. Every type runs
// every shape, so each case also checks that the excluded twin and shape
// combinations keep the parent. 8 and 16 columns may take a twin, 24 never.
std::vector<Case> cases() {
    const ggml_type types[] = {
        GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS,
        GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    };
    const Shape shapes[] = {
        {5120, 17408},  // ffn_gate / ffn_up
        {17408, 5120},  // ffn_down
        {6144, 5120},   // ssm_out / attn_output
        {5120, 6144},   // attn_gate
        {5120, 6176},
        {5120, 6208},
        {5120, 1024},   // attn_k / attn_v
        {5120, 96},     // ssm_beta|alpha
        {5120, 48},
    };
    std::vector<Case> out;
    for (ggml_type t : types) {
        for (const Shape & s : shapes) {
            for (int64_t n : {8, 16, 24}) {
                out.push_back({t, s, n});
            }
        }
    }
    return out;
}

std::string case_key(const Case & c) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s m=%lld k=%lld n=%lld", ggml_type_name(c.type),
                  (long long) c.shape.m, (long long) c.shape.k, (long long) c.n);
    return buf;
}

uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

struct Rng {
    uint64_t s;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

// Byte offsets of the fp16 scale fields in one block (ggml-common.h).
struct BlockScales {
    size_t block_bytes;
    std::vector<size_t> half_offsets;
};

bool block_scales(ggml_type type, BlockScales & out) {
    switch (type) {
        case GGML_TYPE_Q2_K:    out = {84,  {80, 82}}; break;  // scales, qs, d, dmin
        case GGML_TYPE_Q3_K:    out = {110, {108}};    break;  // hmask, qs, scales, d
        case GGML_TYPE_IQ2_XS:  out = {74,  {0}};      break;  // d, qs, scales
        case GGML_TYPE_IQ2_S:   out = {82,  {0}};      break;  // d, qs, qh, scales
        case GGML_TYPE_IQ3_XXS: out = {98,  {0}};      break;  // d, qs
        case GGML_TYPE_IQ3_S:   out = {110, {0}};      break;  // d, qs, qh, signs, scales
        case GGML_TYPE_IQ4_XS:  out = {136, {0}};      break;  // d, scales_h, scales_l, qs
        case GGML_TYPE_Q4_K:    out = {144, {0, 2}};   break;  // d, dmin, scales, qs
        case GGML_TYPE_Q5_K:    out = {176, {0, 2}};   break;  // d, dmin, scales, qh, qs
        case GGML_TYPE_Q6_K:    out = {210, {208}};    break;  // ql, qh, scales, d
        case GGML_TYPE_Q8_0:    out = {34,  {0}};      break;  // d, qs
        default: return false;
    }
    return out.block_bytes == ggml_type_size(type);
}

// Random blocks with every fp16 scale finite: magnitude 2^-12..2^-8, either sign.
bool make_weights(ggml_type type, int64_t k, int64_t m, uint64_t seed, std::vector<uint8_t> & w) {
    BlockScales bs;
    if (!block_scales(type, bs)) {
        return false;
    }
    w.resize(ggml_row_size(type, k) * (size_t) m);
    Rng rng{seed * 0x9E3779B97F4A7C15ull + 1};
    size_t i = 0;
    for (; i + 8 <= w.size(); i += 8) {
        const uint64_t r = rng.next();
        std::memcpy(w.data() + i, &r, 8);
    }
    for (; i < w.size(); ++i) {
        w[i] = (uint8_t) rng.next();
    }
    for (size_t b = 0; b < w.size(); b += bs.block_bytes) {
        for (size_t off : bs.half_offsets) {
            const uint64_t r = rng.next();
            const uint16_t sign = (uint16_t) ((r & 1) << 15);
            const uint16_t exp  = (uint16_t) (3 + (r >> 1) % 5);   // 2^(exp-15)
            const uint16_t mant = (uint16_t) ((r >> 8) & 0x3ff);
            const uint16_t h = sign | (uint16_t) (exp << 10) | mant;
            std::memcpy(w.data() + b + off, &h, 2);
        }
    }
    return ggml_validate_row_data(type, w.data(), w.size());
}

struct Result {
    uint64_t hash = 0;
    size_t mmq = 0;
    size_t p64 = 0;
    size_t p32 = 0;
    bool finite = false;
};

bool run_case(ggml_backend_t gpu, const Case & c, const std::vector<uint8_t> & weights, Result & r) {
    ggml_init_params ip{ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, c.type, c.shape.k, c.shape.m);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.shape.k, c.n);
    ggml_set_input(w);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_output(y);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (buf == nullptr) {
        ggml_free(ctx);
        return false;
    }
    std::vector<float> in((size_t) (c.shape.k * c.n));
    Rng rng{(uint64_t) (c.shape.k * 131 + c.n)};
    for (float & v : in) {
        v = (float) ((rng.next() >> 40) & 0xffffff) / 8388608.0f - 1.0f;
    }
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(x, in.data(), 0, in.size() * sizeof(float));
    const size_t mmq0 = ggml_backend_cuda_get_mmq_launch_count();
    const size_t p64_0 = ggml_backend_cuda_get_mmq_twin_launch_count(64);
    const size_t p32_0 = ggml_backend_cuda_get_mmq_twin_launch_count(32);
    const bool ok = ggml_backend_graph_compute(gpu, g) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(gpu);
    r.mmq = ggml_backend_cuda_get_mmq_launch_count() - mmq0;
    r.p64 = ggml_backend_cuda_get_mmq_twin_launch_count(64) - p64_0;
    r.p32 = ggml_backend_cuda_get_mmq_twin_launch_count(32) - p32_0;
    std::vector<float> out((size_t) (c.shape.m * c.n));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    r.hash = fnv1a(out.data(), out.size() * sizeof(float));
    r.finite = true;
    for (float v : out) {
        r.finite = r.finite && std::isfinite(v);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

bool device_is_rdna4() {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        return false;
    }
    return std::strncmp(prop.gcnArchName, "gfx12", 5) == 0;
}

int child_main() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (gpu == nullptr) {
        std::printf("skip no GPU\n");
        return 77;
    }
    ggml_backend_cuda_set_graphs_disabled_override(true);
    ggml_backend_cuda_set_mmvq_max_ncols_override(1);  // 8+ columns go to MMQ
    // The arm: which twins are on, as the parent passed it ("<rows64><rows32>").
    const char * twins = std::getenv("LUCE_TEST_MMQ_TWINS");
    if (twins != nullptr && std::strlen(twins) == 2) {
        ggml_backend_cuda_set_mmq_twins_override(twins[0] == '1', twins[1] == '1');
    }
    std::printf("device rdna4=%d\n", device_is_rdna4() ? 1 : 0);
    int rc = 0;
    std::map<std::string, std::vector<uint8_t>> weights;
    for (const Case & c : cases()) {
        char wkey[64];
        std::snprintf(wkey, sizeof wkey, "%d %lld %lld", (int) c.type, (long long) c.shape.k, (long long) c.shape.m);
        auto it = weights.find(wkey);
        if (it == weights.end()) {
            weights.clear();  // one weight at a time: cases of a weight are adjacent
            std::vector<uint8_t> w;
            if (!make_weights(c.type, c.shape.k, c.shape.m, (uint64_t) c.type * 1000003u + (uint64_t) c.shape.m, w)) {
                std::printf("error %s: invalid generated weights\n", case_key(c).c_str());
                rc = 1;
                continue;
            }
            it = weights.emplace(wkey, std::move(w)).first;
        }
        Result r;
        if (!run_case(gpu, c, it->second, r)) {
            std::printf("error %s: compute failed\n", case_key(c).c_str());
            rc = 1;
            continue;
        }
        std::printf("case %s hash=%016llx mmq=%zu p64=%zu p32=%zu finite=%d\n", case_key(c).c_str(),
                    (unsigned long long) r.hash, r.mmq, r.p64, r.p32, r.finite ? 1 : 0);
        std::fflush(stdout);
    }
    ggml_backend_free(gpu);
    return rc;
}

struct Arm {
    const char * name;
    std::vector<std::string> env;  // which twins this arm's child turns on
    mmq_twins_enabled on;
};

struct ArmRun {
    int rc = -1;
    bool rdna4 = false;
    std::map<std::string, Result> results;
    std::vector<std::string> errors;
};

bool run_child(const Arm & arm, ArmRun & run) {
    std::vector<std::string> env_store;
    for (char ** e = environ; e != nullptr && *e != nullptr; ++e) {
        if (std::strncmp(*e, "LUCE_TEST_MMQ_TWINS=", 20) != 0) {
            env_store.emplace_back(*e);
        }
    }
    for (const std::string & e : arm.env) {
        env_store.push_back(e);
    }
    std::vector<char *> envp;
    for (std::string & e : env_store) {
        envp.push_back(e.data());
    }
    envp.push_back(nullptr);
    char self[] = "/proc/self/exe";
    char flag[] = "--child";
    char * argv[] = {self, flag, nullptr};

    int fds[2];
    if (pipe(fds) != 0) {
        return false;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    pid_t pid = 0;
    const int err = posix_spawn(&pid, self, &fa, nullptr, argv, envp.data());
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (err != 0) {
        close(fds[0]);
        return false;
    }
    std::string text;
    char buf[4096];
    ssize_t got;
    while ((got = read(fds[0], buf, sizeof buf)) > 0) {
        text.append(buf, (size_t) got);
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    run.rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.rfind("device rdna4=", 0) == 0) {
            run.rdna4 = line == "device rdna4=1";
        } else if (line.rfind("case ", 0) == 0) {
            const size_t h = line.find(" hash=");
            Result r;
            unsigned long long hash = 0;
            int finite = 0;
            if (h == std::string::npos ||
                std::sscanf(line.c_str() + h, " hash=%llx mmq=%zu p64=%zu p32=%zu finite=%d",
                            &hash, &r.mmq, &r.p64, &r.p32, &finite) != 5) {
                run.errors.push_back("unparsed: " + line);
                continue;
            }
            r.hash = hash;
            r.finite = finite != 0;
            run.results[line.substr(5, h - 5)] = r;
        } else if (line.rfind("error ", 0) == 0) {
            run.errors.push_back(line);
        }
    }
    return true;
}

int parent_main() {
    if (ggml_backend_cuda_get_device_count() < 1) {
        std::printf("[mmq-twins] SKIP: no GPU\n");
        return 77;
    }
    std::vector<Arm> arms;
    arms.push_back({"base", {"LUCE_TEST_MMQ_TWINS=00"}, {}});
    {
        Arm a{"ud", {"LUCE_TEST_MMQ_TWINS=10"}, {}};
        a.on.ud_smalltile = true;
        arms.push_back(a);
    }
    {
        Arm a{"2w", {"LUCE_TEST_MMQ_TWINS=01"}, {}};
        a.on.two_warp = true;
        arms.push_back(a);
    }
    {
        Arm a{"both", {"LUCE_TEST_MMQ_TWINS=11"}, {}};
        a.on.ud_smalltile = true;
        a.on.two_warp = true;
        arms.push_back(a);
    }

    std::vector<ArmRun> runs(arms.size());
    for (size_t i = 0; i < arms.size(); ++i) {
        if (!run_child(arms[i], runs[i])) {
            std::printf("[mmq-twins] FAIL: cannot spawn the %s arm\n", arms[i].name);
            return 1;
        }
        if (runs[i].rc == 77) {
            std::printf("[mmq-twins] SKIP: the %s arm found no GPU\n", arms[i].name);
            return 77;
        }
    }

    bool ok = true;
    const bool rdna4 = runs[0].rdna4;
    for (size_t i = 0; i < arms.size(); ++i) {
        for (const std::string & e : runs[i].errors) {
            std::printf("[mmq-twins] FAIL %s: %s\n", arms[i].name, e.c_str());
            ok = false;
        }
        if (runs[i].rc != 0) {
            std::printf("[mmq-twins] FAIL %s: child exit %d\n", arms[i].name, runs[i].rc);
            ok = false;
        }
    }
    size_t compared = 0;
    size_t engaged = 0;
    for (const Case & c : cases()) {
        const std::string key = case_key(c);
        const auto base_it = runs[0].results.find(key);
        if (base_it == runs[0].results.end()) {
            std::printf("[mmq-twins] FAIL %s: no base result\n", key.c_str());
            ok = false;
            continue;
        }
        const Result & base = base_it->second;
        for (size_t i = 0; i < arms.size(); ++i) {
            const auto it = runs[i].results.find(key);
            if (it == runs[i].results.end()) {
                std::printf("[mmq-twins] FAIL %s %s: no result\n", arms[i].name, key.c_str());
                ok = false;
                continue;
            }
            const Result & r = it->second;
            mmq_twin_call call;
            call.type  = c.type;
            call.nrows = c.shape.m;
            call.ncols = c.n;
            call.rdna4 = rdna4;
            const mmq_twin_choice want = mmq_twin_choose(arms[i].on, call);
            const size_t want_p64 = want.tile == mmq_twin_tile::rows64 ? 1 : 0;
            const size_t want_p32 = want.tile == mmq_twin_tile::rows32 ? 1 : 0;
            const bool pass = r.hash == base.hash && r.finite && r.mmq == 1 &&
                              r.p64 == want_p64 && r.p32 == want_p32;
            if (!pass) {
                std::printf("[mmq-twins] FAIL %s %s: hash=%016llx base=%016llx finite=%d mmq=%zu "
                            "p64=%zu (want %zu) p32=%zu (want %zu)\n",
                            arms[i].name, key.c_str(), (unsigned long long) r.hash,
                            (unsigned long long) base.hash, r.finite ? 1 : 0, r.mmq,
                            r.p64, want_p64, r.p32, want_p32);
                ok = false;
            }
            compared += i > 0 ? 1 : 0;
            engaged += r.p64 + r.p32;
        }
    }
    std::printf("[mmq-twins] %zu arm cases compared bit for bit with base, %zu ran a twin\n", compared, engaged);
    if (!ok) {
        std::printf("[mmq-twins] FAIL\n");
        return 1;
    }
    if (!rdna4) {
        std::printf("[mmq-twins] SKIP: not an RDNA4 GPU, the twins never engage here\n");
        return 77;
    }
    std::printf("[mmq-twins] PASS\n");
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--child") == 0) {
        return child_main();
    }
    return parent_main();
}
