// hipBLASLt GEMMs of MOSS-Image: the bf16 linears (the text embedder, the DiT's first and last) and the VAE's attention
// products.
//
// All products are D[rows][n] = B[rows][k] * A[n][k]^T in row-major terms: A holds the weights (or the keys),
// one row per output column, B the activations, one row per token. hipBLASLt sees column-major D (n x rows) =
// A^T (n x k) * B (k x rows) with TRANSA = T.
//
// hipBLASLt's implicit heuristic is not its fastest kernel for every shape: while tuning is on (the warm-up), the
// first product of a shape times the heuristic's top 16 candidates and keeps the fastest. The token count moves with
// every prompt while the best kernel does not, so one tuning serves a 256-row bucket and each new exact problem only
// checks the bucket's kernel. A file keeps the tuned kernels for the next run on the same GPU and hipBLASLt, which then
// finds them again without timing. With tuning off, a problem its bucket's kernel cannot run takes the implicit
// heuristic's, and nothing is timed or kept: a product depends on its problem, the tuning file, the GPU and hipBLASLt,
// never on timing or on the products before it.

#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>

namespace luce::moss_image {

// The values are the tuning file's: 0 was an int8 type that nothing runs any more, and its lines are skipped.
enum class GemmType {
    bf16_f32 = 1,  // bf16 A and B, f32 accumulation and D
    f16_f32  = 2,  // f16 A and B, f32 accumulation and D
};

class Blas {
public:
    Blas() = default;
    ~Blas();
    Blas(const Blas &) = delete;
    Blas & operator=(const Blas &) = delete;

    // The handle and the workspace; every product runs on `stream`.
    bool init(hipStream_t stream);

    // The kernels an earlier run tuned, from the file save_tuning() wrote; returns how many. preload() puts them in
    // use. A file from another GPU or hipBLASLt build reads none, since their kernels differ.
    size_t load_tuning(const std::string & path);

    // The kernels load_tuning() read, found again among the heuristic's candidates for the problems they were timed
    // on, so that every bucket they tuned is tuned again without timing; then one product of each on scratch memory
    // and `stream`, since hipBLASLt reads a kernel's library on its first query and loads its code on its first
    // product. Returns how many are in use.
    size_t preload(hipStream_t stream);

    // Every tuned bucket into `path` for load_tuning(); false when the file cannot be written.
    bool save_tuning(const std::string & path);

    // True when the file save_tuning() writes would differ from the one load_tuning() read.
    bool tuning_changed() const { return changed_; }

    // Tuning on or off (off at first): see above.
    void set_tuning(bool on) { tuning_ = on; }

    // D = B * A^T with leading dimensions (row strides, in elements) lda >= k, ldb >= k, ldd >= n.
    bool gemm(GemmType type, const void * a, int64_t lda, const void * b, int64_t ldb, void * d, int64_t ldd,
              int64_t n, int64_t rows, int64_t k);

private:
    struct Problem {
        GemmType type;
        int64_t  n, rows, k, lda, ldb, ldd;

        bool operator<(const Problem & o) const {
            return std::tie(type, n, rows, k, lda, ldb, ldd) < std::tie(o.type, o.n, o.rows, o.k, o.lda, o.ldb, o.ldd);
        }
    };

    // A problem's descriptors and kernel, made on its first product.
    struct Plan {
        hipblasLtMatmulDesc_t   desc     = nullptr;
        hipblasLtMatrixLayout_t a_layout = nullptr;
        hipblasLtMatrixLayout_t b_layout = nullptr;
        hipblasLtMatrixLayout_t d_layout = nullptr;
        hipblasLtMatmulAlgo_t   algo     = {};
        bool                    tuned    = false;  // false: hipBLASLt's implicit heuristic
    };

    // A bucket's kernel, and the rows of the problem it was timed on.
    struct Tuned {
        hipblasLtMatmulAlgo_t algo = {};
        int64_t               rows = 0;
    };

    bool        make_plan(const Problem & p, Plan & plan) const;
    void        destroy_plan(Plan & plan) const;
    int         heuristic(const Plan & plan, hipblasLtMatmulHeuristicResult_t * candidates) const;
    bool        choose_algo(const Problem & p, Plan & plan, const void * a, const void * b, void * d);
    std::string tuning_header() const;

    hipStream_t       stream_    = nullptr;
    hipblasLtHandle_t handle_    = nullptr;
    void *            workspace_ = nullptr;
    std::map<Problem, Plan>  plans_;
    std::map<Problem, Tuned> bucket_algos_;  // the tuned kernel of each 256-row bucket
    std::map<Problem, int>   stored_;        // from the tuning file: each timed problem's kernel index
    bool                     changed_ = false;
    bool                     tuning_  = false;
};

} // namespace luce::moss_image
