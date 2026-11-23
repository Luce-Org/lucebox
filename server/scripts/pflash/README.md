# PFlash development tools

The two tools here cover request-probe GGUF export and parity fixtures.
They preserve the existing probe schemas and seeded fixtures. NumPy is
required; GGUF export uses the vendored `gguf-py` package.

## Export the probe

```sh
python server/scripts/pflash/request_probe_gguf.py request.gguf \
  --drafter-sha256 HEX --tau THRESHOLD --weights probe.npz
```

`--random-seed` remains available for runtime plumbing tests. Those weights are
not a trained probe. The exporter retains the request-probe tensor shapes,
schema and extraction-version metadata.

## Regenerate parity fixtures

```sh
python server/scripts/pflash/gen_request_probe_v2_fixtures.py TRAINER_SRC_DIR \
  server/test/pflash/fixtures/request_probe_v2_parity.inc
```

The generator imports the trainer's offline rule, checks its candidate-rule
contract and emits the same 80 seed-0 cases. Keeping that offline implementation
as the reference makes the fixtures independent of the C++ rule under test.
