// GGUF model inspection utilities.
//
// Reads minimal metadata (architecture, layer count) from a GGUF file
// without loading weights. Useful for arch detection and layer-split planning.

#pragma once

#include <cstdint>
#include <string>

namespace dflash::common {

struct GgufModelInfo {
    std::string arch;       // e.g. "qwen35", "laguna", "qwen3", "gemma4"
    int         n_layer = -1;
};

// Read architecture and layer count from a GGUF file.
// Returns info with arch="" and n_layer=-1 on failure.
GgufModelInfo inspect_gguf_model_info(const char * path);

// Richer GGUF identity captured at server startup and re-emitted at /props.
// All header values are best-effort: missing keys leave the corresponding
// field at the listed default (empty string or -1). `ok` is false only if
// the file itself couldn't be opened (path missing, not a GGUF, etc.).
//
// The intent is "exactly what binary + GGUF + quant + sha256 is loaded";
// any field the file doesn't carry stays at the default so consumers can
// distinguish "not in GGUF" (-1) from "0" (legitimately zero).
struct GgufMetadata {
    bool        ok          = false;        // false: open failed, all other fields ignorable
    std::string path;                       // absolute filesystem path passed in
    int64_t     size_bytes  = -1;           // file size (-1 if stat failed)
    std::string sha256;                     // lowercase hex sha256 (empty if not computed)

    // Header fields (`general.*` + `<arch>.*`). All optional.
    std::string general_architecture;       // raw value of "general.architecture"
    std::string general_name;               // "general.name" (display string)
    int32_t     file_type        = -1;      // "general.file_type" (LLAMA_FTYPE_* int)
    std::string file_type_name;             // decoded LLAMA_FTYPE_* (e.g. "Q4_K_M", "IQ4_XS")
    int32_t     quantization_version = -1;  // "general.quantization_version"

    int32_t     block_count       = -1;     // "<arch>.block_count"
    int32_t     embedding_length  = -1;     // "<arch>.embedding_length"
    int32_t     context_length    = -1;     // "<arch>.context_length"
    int32_t     vocab_size        = -1;     // "<arch>.vocab_size" (or tokenizer.ggml.tokens length)
};

// Read GGUF identity for /props. Set `compute_sha256` to hash the file (slow,
// O(size) — multi-GB GGUFs take ~30s on a fast SSD). When false, `sha256`
// stays empty. The header read is cheap (no weight load).
//
// When `compute_sha256` is true and a sidecar file `<path>.sha256` exists,
// its first 64-hex-char token is trusted as the file's sha256 and the file
// is not re-hashed. After a successful hash, the result is written to the
// sidecar so subsequent restarts skip the rehash. Sidecar I/O failures are
// non-fatal — the in-memory hash still gets returned.
GgufMetadata read_gguf_metadata(const std::string & path,
                                bool compute_sha256);

}  // namespace dflash::common
