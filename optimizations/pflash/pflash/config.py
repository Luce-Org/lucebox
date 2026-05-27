"""Default env flags for the dflash daemon when spawned by pflash."""

# These are the only daemon-side flags pflash assumes. The C++ kernel knobs
# (DFLASH_FP_USE_BSA, DFLASH_FP_ALPHA) are set per-call by the daemon owner.
DFLASH_REQUIRED_ENV = {
    "DFLASH27B_FA_WINDOW": "0",       # full attn on the (already compressed) prompt
    "DFLASH27B_KV_TQ3": "1",          # 3-bit KV cache; saves ~4 GB at 128K
    "DFLASH27B_LM_HEAD_FIX": "0",     # disable cuBLAS LM-head dequant (OOMs on 24 GB)
}
