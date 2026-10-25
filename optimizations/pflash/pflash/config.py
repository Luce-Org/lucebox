"""Default env flags for the luce daemon when spawned by pflash."""

# The only daemon-side env flags pflash assumes.
LUCE_REQUIRED_ENV = {
    "LUCE_FA_WINDOW": "0",       # full attn on the (already compressed) prompt
    "LUCE_KV_TQ3": "1",          # 3-bit KV cache; saves ~4 GB at 128K
}
