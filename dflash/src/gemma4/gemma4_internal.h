// gemma4_internal.h — Gemma4 struct definitions and graph/cache declarations.
//
// Re-exports the Gemma4 section from the shared internal.h so that gemma4/
// translation units can use the layout-canonical include path
// "gemma4_internal.h" as in howard0su PR #175, while the structs themselves
// remain in internal.h (shared with non-Gemma4 paths).
//
// Structs re-exported here:
//   GemmaTargetLayer, GemmaTargetWeights, GemmaTargetCache
//   GemmaGraphInputs, GemmaGraphOutputs
//   GemmaDraftLayer, GemmaDraftWeights
//   MtpLayerWeights, MtpDrafterWeights, MtpStepGraph
//   SwaView
//   All associated lifecycle/graph-builder function declarations

#pragma once
#include "../internal.h"
