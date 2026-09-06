// Cross-feature compatibility gate.
//
// One place for admission rules that combine a requested feature with facts
// only known after model and placement resolution. Before this existed the
// same rules were spelled out at two or three layers, each seeing a different
// slice of the config, so no single site was authoritative.
//
// Syntax and value parsing stay at the parse site. Once raw arguments have
// been parsed, every launch-compatibility rule belongs here: cross-flag
// requirements, compiled-backend placement, layer-split topology, and
// architecture-specific feature support.
//
// This is policy; model_capabilities.h is the fact it reasons over. The two
// stay separate because they change for different reasons — the table changes
// when create_backend()'s dispatch changes, these rules change when a
// combination is judged inadmissible — and because the table is a leaf header
// that callers with no interest in BackendArgs can include cheaply. It is
// pulled in here rather than left to each caller because every rule below is
// phrased in its vocabulary; keep the include even though no declaration in
// this header names a capability type.
//
// Warnings are a separate return channel, not a separate layer.
// collect_feature_warnings() reports flags that are accepted but inert on the
// resolved architecture and placement — the request still runs, just without
// that feature, so it must not fail admission. Degradations that mutate config
// on the way through ("--draft-residency=request-scoped ignored", "--spark
// ignored") stay with the setting they rewrite; a pure function returning
// strings cannot express those.

#pragma once

#include "backend_args.h"
#include "model_capabilities.h"
#include "placement/placement_backend.h"

#include <string>
#include <vector>

namespace dflash::common {

// Returns an empty string when the requested feature set is coherent, or a
// description of the first violated rule.
//
// `admission` carries launch facts owned above the backend factory. `arch`,
// `target_backend`, and `compiled_backend` are resolved facts: the architecture
// read from the GGUF, the requested target with PlacementBackend::Auto mapped
// to the compiled default, and the binary's compiled placement. Passing these
// in keeps the gate a pure function that unit tests can drive without a model
// file or GPU.
//
// prepare_backend() owns the public admission decision. Construction trusts
// the immutable BackendPlan produced by that boundary.
std::string check_feature_compatibility(
    const BackendArgs & args,
    const BackendAdmissionContext & admission,
    const std::string & arch,
    PlacementBackend    target_backend,
    PlacementBackend    compiled_backend);

// Flags that were accepted but do nothing on this architecture and placement.
// Each entry is a complete operator-facing sentence, without a log prefix.
//
// These never block a launch: the server runs exactly as it would have, so
// turning them into errors would reject configurations that work today. They
// exist because the alternative — a flag that is silently dropped in the
// factory's dispatch — is indistinguishable from a flag that took effect.
//
// Only meaningful once check_feature_compatibility() has passed; a rejected
// configuration has no useful warnings to report.
std::vector<std::string> collect_feature_warnings(
    const BackendArgs & args,
    const std::string & arch);

}  // namespace dflash::common
