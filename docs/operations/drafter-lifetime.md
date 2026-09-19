# Decoding drafter lifetime

The HTTP server keeps the DFlash decoding drafter resident across requests and at startup. It does not park/reload the drafter at request boundaries. The separate PFlash scorer is released after scoring, including the remote scorer process.

`--draft-residency` and `--lazy-draft` have been removed. Remove them from gateway `extra_args` and direct launch commands; the CLI rejects them as unknown options. `/props.runtime.draft_residency` reports `persistent`; `/props.pflash.draft_residency` reports `release-after-use`.

This removes request-boundary residency policy. It does not remove backend park/unpark APIs: explicit compression parking without `--prefill-skip-park`, unloading the model, and shutdown retain their own lifecycles. Production hybrid compression uses skip-park.

Hybrid admission uses actual free GPU memory with the drafter already loaded. The drafter is not an optional cache eviction victim. The existing fixed working-memory allowance remains a separate limitation; this change does not implement phase-specific peak prediction.

For migration, preserve the old executable and model configuration, build `dflash_server` and `test_server_unit`, run the unit suite with the ROCm library path, check both removed flags are rejected, and restart only while idle. Verify consecutive generations and an eligible compressed prompt/repeat without decoding-drafter parking or reloading in the log.
