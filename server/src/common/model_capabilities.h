// Architecture-level feature capabilities.
//
// One table describing what each architecture supports, plus predicates that
// read it. This is not configuration: every row states what create_backend()
// actually plumbs through to that architecture's backend, so the table can
// only ever describe the dispatch chain — it cannot lead it. A new
// architecture needs a backend, a factory case, and a row here, in that order.
// Keeping it in the binary is deliberate; an external file would add a third
// artifact to keep in sync and turn a compile-time constant into a parse that
// can fail at startup.
//
// Read a row to see what a model can do; read a column to see which
// architectures honor a flag. The row order matches create_backend()'s
// dispatch chain so the two can be diffed by eye.
//
// Placement matters as much as architecture: laguna and gemma4 forward a
// draft model only when monolithic, so those entries are FeatureSupport
// rather than bool. qwen35 paged attention is Monolithic for the same reason:
// only the single-device backend owns a paged K/V pool.
//
// qwen35 and qwen35moe share Qwen35Config, so their rows can differ on a
// column the config carries — the moe backend simply never reads the field.
// paged_attn and draft_block_size are such columns today; see the cross-check
// comment in backend_factory.cpp for what that costs.
//
// Note on "qwen36": it is not a dispatchable architecture. model_card.cpp's
// family fallback has a branch for it, but there is no factory case, so a
// GGUF declaring general.architecture=qwen36 is rejected by
// arch_is_supported() before any model card is resolved. If that arch string
// ever ships it needs a backend, a factory case, and a row below.

#pragma once

#include <cstddef>
#include <string>

namespace dflash::common {

// Whether an option reaches the backend, as a function of placement.
enum class FeatureSupport {
    Never,       // the architecture has no path for it at all
    Monolithic,  // single-device only; the layer-split adapter drops it
    Both,        // forwarded on both the monolithic and layer-split paths
};

struct ArchCapabilities {
    const char * arch;

    // Placement-independent.
    bool layer_split;         // has a LayerSplitAdapter (--target-devices)
    bool remote_draft;        // draft over IPC (--draft-ipc-bin)
    bool pflash_compression;  // PFlash prefill compression (--prefill-compression)
    bool expert_offload;      // hot/cold MoE expert residency (--spark, --freq,
                              // --collect-routing, --adaptive-experts). Note
                              // deepseek4 is mixture-of-experts but has no such
                              // path, so this is narrower than "is MoE".

    // Placement-dependent.
    FeatureSupport decode_draft;  // --draft
    FeatureSupport ddtree;        // --ddtree, --ddtree-budget, --ddtree-temp
    FeatureSupport verify_width;  // --verify-width
    FeatureSupport draft_block_size; // --draft-block-size
    FeatureSupport fa_window;     // --fa-window
    FeatureSupport draft_swa;     // --draft-swa
    FeatureSupport paged_attn;    // --paged-attention
};

inline constexpr FeatureSupport kNever = FeatureSupport::Never;
inline constexpr FeatureSupport kMono  = FeatureSupport::Monolithic;
inline constexpr FeatureSupport kBoth  = FeatureSupport::Both;

inline constexpr ArchCapabilities kArchCapabilities[] = {
//   arch          split  rdraft pflash offload  draft  ddtree vwidth dblock fa_win dswa    paged
    {"qwen35",     true,  true,  true,  false,   kBoth, kBoth, kNever, kMono,  kBoth, kBoth,  kMono},
    {"qwen35moe",  false, false, false, true,    kMono, kMono, kNever, kNever, kMono, kMono,  kNever},
    {"laguna",     true,  false, false, true,    kMono, kMono, kMono,  kNever, kNever, kNever, kNever},
    {"qwen3",      false, false, true,  false,   kNever, kNever, kNever, kNever, kNever, kNever, kNever},
    {"gemma4",     true,  false, false, false,   kMono, kNever, kNever, kNever, kBoth, kNever, kNever},
    {"deepseek4",  true,  false, false, false,   kNever, kNever, kNever, kNever, kNever, kNever, kNever},
};

inline constexpr std::size_t kArchCount =
    sizeof(kArchCapabilities) / sizeof(kArchCapabilities[0]);

constexpr bool arch_name_equal(const char * a, const char * b) {
    while (*a != '\0' && *a == *b) { ++a; ++b; }
    return *a == *b;
}

// constexpr so the table can be checked against the factory's config structs
// at compile time; see the static_asserts in backend_factory.cpp.
constexpr const ArchCapabilities * find_arch_capabilities(const char * arch) {
    for (const ArchCapabilities & caps : kArchCapabilities) {
        if (arch_name_equal(arch, caps.arch)) return &caps;
    }
    return nullptr;
}

inline const ArchCapabilities * find_arch_capabilities(const std::string & arch) {
    return find_arch_capabilities(arch.c_str());
}

// Row for an architecture known to be in the table. Calling this with an
// unlisted name is a compile error in a constant expression (null deref),
// which is what makes the cross-checks in backend_factory.cpp safe.
constexpr const ArchCapabilities & arch_capabilities(const char * arch) {
    return *find_arch_capabilities(arch);
}

namespace detail {

constexpr bool row_has_both(const ArchCapabilities & c) {
    return c.decode_draft == FeatureSupport::Both ||
           c.ddtree       == FeatureSupport::Both ||
           c.verify_width == FeatureSupport::Both ||
           c.draft_block_size == FeatureSupport::Both ||
           c.fa_window    == FeatureSupport::Both ||
           c.draft_swa    == FeatureSupport::Both ||
           c.paged_attn   == FeatureSupport::Both;
}

constexpr bool table_rows_named() {
    for (const ArchCapabilities & c : kArchCapabilities) {
        if (c.arch == nullptr || c.arch[0] == '\0') return false;
    }
    return true;
}

constexpr bool table_rows_unique() {
    for (std::size_t i = 0; i < kArchCount; ++i) {
        for (std::size_t j = i + 1; j < kArchCount; ++j) {
            if (arch_name_equal(kArchCapabilities[i].arch,
                                kArchCapabilities[j].arch)) {
                return false;
            }
        }
    }
    return true;
}

// "Both" means "and also on the layer-split path", which an architecture
// without a layer-split adapter does not have. Such a row must say
// Monolithic, so that the split half of every cross-check below is false.
constexpr bool table_split_coherent() {
    for (const ArchCapabilities & c : kArchCapabilities) {
        if (!c.layer_split && row_has_both(c)) return false;
    }
    return true;
}

// Remote draft execution runs a draft model over IPC. An architecture that
// never accepts a draft model cannot run one remotely.
constexpr bool table_remote_draft_coherent() {
    for (const ArchCapabilities & c : kArchCapabilities) {
        if (c.remote_draft && c.decode_draft == FeatureSupport::Never) {
            return false;
        }
    }
    return true;
}

}  // namespace detail

static_assert(detail::table_rows_named(),
              "every capability row needs a non-empty architecture name");
static_assert(detail::table_rows_unique(),
              "duplicate architecture row in kArchCapabilities");
static_assert(detail::table_split_coherent(),
              "an architecture with no layer-split adapter cannot support an "
              "option on 'Both' placements; use Monolithic");
static_assert(detail::table_remote_draft_coherent(),
              "remote draft execution requires an architecture that accepts a "
              "draft model");

namespace detail {

inline bool supported_at(FeatureSupport support, bool is_layer_split) {
    switch (support) {
        case FeatureSupport::Never:      return false;
        case FeatureSupport::Monolithic: return !is_layer_split;
        case FeatureSupport::Both:       return true;
    }
    return false;
}

inline bool arch_has(const std::string & arch,
                     bool ArchCapabilities::* field) {
    const ArchCapabilities * caps = find_arch_capabilities(arch);
    return caps && caps->*field;
}

inline bool arch_has(const std::string & arch,
                     FeatureSupport ArchCapabilities::* field,
                     bool is_layer_split) {
    const ArchCapabilities * caps = find_arch_capabilities(arch);
    return caps && supported_at(caps->*field, is_layer_split);
}

}  // namespace detail

// An unknown architecture answers false to every predicate below, so no rule
// can admit a model the factory cannot build.
inline bool arch_is_supported(const std::string & arch) {
    return find_arch_capabilities(arch) != nullptr;
}

inline bool arch_supports_layer_split(const std::string & arch) {
    return detail::arch_has(arch, &ArchCapabilities::layer_split);
}

inline bool arch_supports_remote_draft(const std::string & arch) {
    return detail::arch_has(arch, &ArchCapabilities::remote_draft);
}

inline bool arch_supports_pflash_compression(const std::string & arch) {
    return detail::arch_has(arch, &ArchCapabilities::pflash_compression);
}

inline bool arch_has_expert_offload(const std::string & arch) {
    return detail::arch_has(arch, &ArchCapabilities::expert_offload);
}

inline bool arch_supports_decode_draft(const std::string & arch,
                                       bool is_layer_split) {
    return detail::arch_has(arch, &ArchCapabilities::decode_draft, is_layer_split);
}

inline bool arch_supports_ddtree(const std::string & arch,
                                 bool is_layer_split) {
    return detail::arch_has(arch, &ArchCapabilities::ddtree, is_layer_split);
}

inline bool arch_supports_verify_width(const std::string & arch,
                                       bool is_layer_split) {
    return detail::arch_has(arch, &ArchCapabilities::verify_width, is_layer_split);
}

inline bool arch_supports_draft_block_size(const std::string & arch,
                                           bool is_layer_split) {
    return detail::arch_has(
        arch, &ArchCapabilities::draft_block_size, is_layer_split);
}

inline bool arch_supports_fa_window(const std::string & arch,
                                    bool is_layer_split) {
    return detail::arch_has(arch, &ArchCapabilities::fa_window, is_layer_split);
}

inline bool arch_supports_draft_swa(const std::string & arch,
                                    bool is_layer_split) {
    return detail::arch_has(arch, &ArchCapabilities::draft_swa, is_layer_split);
}

inline bool arch_supports_paged_attention(const std::string & arch,
                                          bool is_layer_split) {
    return detail::arch_has(arch, &ArchCapabilities::paged_attn, is_layer_split);
}

}  // namespace dflash::common
