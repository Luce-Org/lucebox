#include "blas.h"

#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

namespace luce::moss_image {

namespace {

constexpr size_t  kWorkspaceBytes = 64u << 20;
constexpr int     kCandidates     = 16;
constexpr int64_t kRowBucket      = 256;

struct TypeInfo {
    hipDataType          ab;
    hipDataType          d;
    hipblasComputeType_t compute;
    hipDataType          scale;
};

TypeInfo type_info(const GemmType type) {
    switch (type) {
        case GemmType::bf16_f32: return { HIP_R_16BF, HIP_R_32F, HIPBLAS_COMPUTE_32F, HIP_R_32F };
        case GemmType::f16_f32:  return { HIP_R_16F, HIP_R_32F, HIPBLAS_COMPUTE_32F, HIP_R_32F };
    }
    return {};
}

// alpha 1 and beta 0 (every type scales in f32).
const void * scalar(const bool one) {
    static const float f32[2] = { 0.0f, 1.0f };
    return &f32[one];
}

int64_t bucket_rows(const int64_t rows) {
    return (rows + kRowBucket - 1) / kRowBucket * kRowBucket;
}

} // namespace

Blas::~Blas() {
    for (auto & [problem, plan] : plans_) {
        destroy_plan(plan);
    }
    if (workspace_ != nullptr) {
        (void) hipFree(workspace_);
    }
    if (handle_ != nullptr) {
        hipblasLtDestroy(handle_);
    }
}

bool Blas::init(hipStream_t stream) {
    stream_ = stream;
    return hipblasLtCreate(&handle_) == HIPBLAS_STATUS_SUCCESS && hipMalloc(&workspace_, kWorkspaceBytes) == hipSuccess;
}

bool Blas::make_plan(const Problem & p, Plan & plan) const {
    const TypeInfo           t       = type_info(p.type);
    const hipblasOperation_t trans_a = HIPBLAS_OP_T;
    return hipblasLtMatmulDescCreate(&plan.desc, t.compute, t.scale) == HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) ==
               HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutCreate(&plan.a_layout, t.ab, p.k, p.n, p.lda) == HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutCreate(&plan.b_layout, t.ab, p.k, p.rows, p.ldb) == HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutCreate(&plan.d_layout, t.d, p.n, p.rows, p.ldd) == HIPBLAS_STATUS_SUCCESS;
}

void Blas::destroy_plan(Plan & plan) const {
    if (plan.d_layout != nullptr) {
        hipblasLtMatrixLayoutDestroy(plan.d_layout);
    }
    if (plan.b_layout != nullptr) {
        hipblasLtMatrixLayoutDestroy(plan.b_layout);
    }
    if (plan.a_layout != nullptr) {
        hipblasLtMatrixLayoutDestroy(plan.a_layout);
    }
    if (plan.desc != nullptr) {
        hipblasLtMatmulDescDestroy(plan.desc);
    }
    plan = Plan{};
}

// hipBLASLt's best candidates for the plan's problem, at most kCandidates; how many.
int Blas::heuristic(const Plan & plan, hipblasLtMatmulHeuristicResult_t * candidates) const {
    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) {
        return 0;
    }
    const uint64_t max_workspace = kWorkspaceBytes;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace,
                                          sizeof(max_workspace));
    int                   n_found = 0;
    const hipblasStatus_t status  = hipblasLtMatmulAlgoGetHeuristic(handle_, plan.desc, plan.a_layout, plan.b_layout,
                                                                    plan.d_layout, plan.d_layout, pref, kCandidates,
                                                                    candidates, &n_found);
    hipblasLtMatmulPreferenceDestroy(pref);
    return status == HIPBLAS_STATUS_SUCCESS ? n_found : 0;
}

// The bucket's kernel when it runs this exact problem; else, while tuning, the fastest of the heuristic's candidates,
// timed on the call's own operands (each candidate writes the same product to d); else the implicit heuristic's. False
// inside stream capture, which cannot time.
bool Blas::choose_algo(const Problem & p, Plan & plan, const void * a, const void * b, void * d) {
    const void * alpha = scalar(true);
    const void * beta  = scalar(false);
    Problem bucket = p;
    bucket.rows    = bucket_rows(p.rows);
    const auto known = bucket_algos_.find(bucket);
    if (known != bucket_algos_.end()) {
        hipblasLtMatmulAlgo_t algo           = known->second.algo;
        size_t                workspace_size = 0;
        if (hipblaslt_ext::matmulIsAlgoSupported(handle_, plan.desc, alpha, plan.a_layout, plan.b_layout, beta,
                                                 plan.d_layout, plan.d_layout, algo,
                                                 workspace_size) == HIPBLAS_STATUS_SUCCESS &&
            workspace_size <= kWorkspaceBytes) {
            plan.algo  = algo;
            plan.tuned = true;
            return true;
        }
    }
    if (!tuning_) {
        return true;
    }
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream_, &capture) != hipSuccess || capture != hipStreamCaptureStatusNone) {
        return false;
    }

    hipblasLtMatmulHeuristicResult_t candidates[kCandidates];
    const int                        n_found = heuristic(plan, candidates);
    if (n_found == 0) {
        return true;
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop  = nullptr;
    if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) {
        return true;
    }
    int   best    = -1;
    float best_ms = 0.0f;
    for (int i = 0; i < n_found; ++i) {
        if (candidates[i].workspaceSize > kWorkspaceBytes) {
            continue;
        }
        const auto run = [&]() {
            return hipblasLtMatmul(handle_, plan.desc, alpha, a, plan.a_layout, b, plan.b_layout, beta, d,
                                   plan.d_layout, d, plan.d_layout, &candidates[i].algo, workspace_, kWorkspaceBytes,
                                   stream_);
        };
        if (run() != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }
        float ms = 0.0f;
        if (hipEventRecord(start, stream_) != hipSuccess || run() != HIPBLAS_STATUS_SUCCESS ||
            run() != HIPBLAS_STATUS_SUCCESS || hipEventRecord(stop, stream_) != hipSuccess ||
            hipEventSynchronize(stop) != hipSuccess || hipEventElapsedTime(&ms, start, stop) != hipSuccess) {
            continue;
        }
        if (best < 0 || ms < best_ms) {
            best    = i;
            best_ms = ms;
        }
    }
    (void) hipEventDestroy(start);
    (void) hipEventDestroy(stop);
    if (best >= 0) {
        plan.algo             = candidates[best].algo;
        plan.tuned            = true;
        bucket_algos_[bucket] = { plan.algo, p.rows };
        changed_              = true;
    }
    return true;
}

// The GPU and the hipBLASLt build a tuning file belongs to: its kernels are indexes into this hipBLASLt's solutions,
// timed on this GPU.
std::string Blas::tuning_header() const {
    int             device        = 0;
    hipDeviceProp_t prop          = {};
    int             version       = 0;
    char            revision[256] = {};
    (void) hipGetDevice(&device);
    (void) hipGetDeviceProperties(&prop, device);
    (void) hipblasLtGetVersion(handle_, &version);
    (void) hipblasLtGetGitRevision(handle_, revision);
    return std::string("moss-image-gemm 1 ") + prop.gcnArchName + " " + std::to_string(prop.multiProcessorCount) +
           " hipblaslt " + std::to_string(version) + " " + revision;
}

// After the header, one line per bucket: the problem its kernel was timed on (type, n, rows, k, lda, ldb, ldd) and
// the kernel's solution index.
size_t Blas::load_tuning(const std::string & path) {
    std::ifstream in(path);
    std::string   line;
    if (in && std::getline(in, line) && line == tuning_header()) {
        while (std::getline(in, line)) {
            std::istringstream fields(line);
            int                type  = -1;
            int                index = -1;
            Problem            p{};
            if (fields >> type >> p.n >> p.rows >> p.k >> p.lda >> p.ldb >> p.ldd >> index &&
                type >= int(GemmType::bf16_f32) && type <= int(GemmType::f16_f32)) {
                p.type     = GemmType(type);
                stored_[p] = index;
            }
        }
    }
    return stored_.size();
}

size_t Blas::preload(hipStream_t stream) {
    // A stored index only ever picks one of the heuristic's own candidates, so a stale one is never run.
    std::vector<std::pair<Problem, hipblasLtMatmulAlgo_t>> found;
    for (const auto & [p, index] : stored_) {
        hipblasLtMatmulHeuristicResult_t candidates[kCandidates];
        Plan                             plan;
        const int                        n_found = make_plan(p, plan) ? heuristic(plan, candidates) : 0;
        destroy_plan(plan);
        for (int i = 0; i < n_found; ++i) {
            if (candidates[i].workspaceSize <= kWorkspaceBytes &&
                hipblaslt_ext::getIndexFromAlgo(candidates[i].algo) == index) {
                Problem bucket        = p;
                bucket.rows           = bucket_rows(p.rows);
                bucket_algos_[bucket] = { candidates[i].algo, p.rows };
                found.emplace_back(p, candidates[i].algo);
                break;
            }
        }
    }
    changed_ = changed_ || found.size() < stored_.size();  // the file loses the kernels not found again
    stored_.clear();

    // One allocation holds the largest A, B and D of them.
    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t d_bytes = 0;
    for (const auto & [p, algo] : found) {
        a_bytes = std::max(a_bytes, size_t(p.n * p.lda) * 2);  // bf16 or f16
        b_bytes = std::max(b_bytes, size_t(p.rows * p.ldb) * 2);
        d_bytes = std::max(d_bytes, size_t(p.rows * p.ldd) * 4);
    }
    a_bytes          = (a_bytes + 255) / 256 * 256;
    b_bytes          = (b_bytes + 255) / 256 * 256;
    uint8_t * memory = nullptr;
    if (found.empty() || hipMalloc(&memory, a_bytes + b_bytes + d_bytes) != hipSuccess) {
        return found.size();
    }
    uint8_t * const a = memory;
    uint8_t * const b = memory + a_bytes;
    uint8_t * const d = memory + a_bytes + b_bytes;
    for (auto & [p, algo] : found) {
        Plan plan;
        if (make_plan(p, plan)) {
            (void) hipblasLtMatmul(handle_, plan.desc, scalar(true), a, plan.a_layout, b, plan.b_layout,
                                   scalar(false), d, plan.d_layout, d, plan.d_layout, &algo, workspace_,
                                   kWorkspaceBytes, stream);
        }
        destroy_plan(plan);
    }
    (void) hipStreamSynchronize(stream);
    (void) hipFree(memory);
    return found.size();
}

bool Blas::save_tuning(const std::string & path) {
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), error);
    const std::string temp = path + ".tmp";
    std::ofstream     out(temp, std::ios::trunc);
    out << tuning_header() << '\n';
    for (const auto & [bucket, tuned] : bucket_algos_) {
        hipblasLtMatmulAlgo_t algo = tuned.algo;
        out << int(bucket.type) << ' ' << bucket.n << ' ' << tuned.rows << ' ' << bucket.k << ' ' << bucket.lda << ' '
            << bucket.ldb << ' ' << bucket.ldd << ' ' << hipblaslt_ext::getIndexFromAlgo(algo) << '\n';
    }
    out.close();
    if (!out || std::rename(temp.c_str(), path.c_str()) != 0) {
        std::remove(temp.c_str());
        return false;
    }
    changed_ = false;
    return true;
}

bool Blas::gemm(const GemmType type, const void * a, const int64_t lda, const void * b, const int64_t ldb, void * d,
                const int64_t ldd, const int64_t n, const int64_t rows, const int64_t k) {
    const Problem p{ type, n, rows, k, lda, ldb, ldd };
    auto          it = plans_.find(p);
    Plan          untimed;  // a plan made inside stream capture, used once and not kept
    Plan *        plan = nullptr;
    if (it != plans_.end()) {
        plan = &it->second;
    } else {
        Plan made;
        if (!make_plan(p, made)) {
            destroy_plan(made);
            return false;
        }
        if (choose_algo(p, made, a, b, d)) {
            plan = &(plans_[p] = made);
        } else {
            untimed = made;
            plan    = &untimed;
        }
    }
    const hipblasStatus_t status =
        hipblasLtMatmul(handle_, plan->desc, scalar(true), a, plan->a_layout, b, plan->b_layout, scalar(false), d,
                        plan->d_layout, d, plan->d_layout, plan->tuned ? &plan->algo : nullptr, workspace_,
                        plan->tuned ? kWorkspaceBytes : 0, stream_);
    if (plan == &untimed) {
        destroy_plan(untimed);
    }
    return status == HIPBLAS_STATUS_SUCCESS;
}

} // namespace luce::moss_image
