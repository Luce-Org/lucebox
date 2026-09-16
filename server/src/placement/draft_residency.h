// Compression scorer lifetime, separate from the persistent decoding drafter.
#pragma once
namespace dflash::common {
enum class DraftResidencyAction {
    KeepLoaded,
    ReleaseAfterUse,
};
}  // namespace dflash::common
