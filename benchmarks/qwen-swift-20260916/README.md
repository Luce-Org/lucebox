# Qwen Swift controlled comparison and Unsloth follow-up

See `swift-controlled-report.md` for results and exact scoring criteria.

- `swift-controlled-v1`: 96 measured requests: 16 frozen inputs × two backends × three runs. Rejected preflight data are separately labelled and excluded.
- `unsloth-controlled-once`: same 16 requests with regular Unsloth weights, one run.
- `swift-mtp` and `unsloth-reproduction`: earlier exploratory evidence; do not pool these with the controlled study.
- `report-source`: authored report content and reviewed data; the installed Data app runtime is not vendored.

Scripts retain the absolute linuxmacan paths used during execution. No weights or compiled binaries are included. Timing includes backend phase metrics and external request wall time; see report for token-count definitions.
