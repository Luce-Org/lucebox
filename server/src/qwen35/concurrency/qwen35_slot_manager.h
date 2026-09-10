#pragma once

#include "common/concurrency/seq_slot_manager.h"

namespace dflash::common {

// Compatibility aliases keep the Qwen engine source stable while both Qwen
// and DeepSeek share the current model-neutral slot lifecycle implementation.
using Qwen35SlotPhase = SeqSlotPhase;
using Qwen35Slot = SeqSlot;
using Qwen35SlotManager = SeqSlotManager;

}  // namespace dflash::common
