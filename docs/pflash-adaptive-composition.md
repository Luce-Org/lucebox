# pflash adaptive composition (Design 1)

When pflash compresses a prompt, the target spec-decode verify window must
cover the entire compressed sequence — otherwise verify sees only the last
fa_window positions and loses needle context.

`http_server.cpp`: when pflash_compressed, sets
`req.fa_window_override = effective_prompt.size() + 256`.
This never caps visibility; pflash already paid compute to pick which tokens
matter, so every kept token must be visible in verify.

`qwen35_backend.cpp` C2 gate: after prefill, checks whether spec-decode
arithmetic still earns its drafter cost at the override window size.

- override <= 2 * cfg_.fa_window → spec-decode
- override >  2 * cfg_.fa_window → AR fallback (fa_window=0, full attention)

Both paths see every kept token. The gate chooses mechanism, not visibility.
