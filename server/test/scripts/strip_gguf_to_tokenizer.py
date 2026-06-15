#!/usr/bin/env python3
"""Strip a GGUF to tokenizer metadata only — no tensor data.

The dflash Tokenizer (server/src/server/tokenizer.cpp) only reads KV metadata:
  tokenizer.ggml.tokens, .merges, .token_type, .model, .pre, .bos_token_id,
  .eos_token_id, .eot_token_id (best-effort), .chat_template

Everything else — including all 851+ tensor weights — can be dropped. The
result is small enough to commit as a CI test fixture for the CPU-only
HttpServer driver.

Usage:
  python strip_gguf_to_tokenizer.py <input.gguf> <output.gguf>
"""

from __future__ import annotations

import sys
from pathlib import Path

import gguf


KEEP_PREFIXES = (
    "tokenizer.",
    # general.architecture is read by backend_factory's detect_arch() to pick
    # the chat-format enum. Cheap to keep and useful for any caller that
    # peeks at the file.
    "general.architecture",
    "general.name",
)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    reader = gguf.GGUFReader(str(src))
    arch = reader.fields["general.architecture"].contents()
    print(f"[strip] reading {src} (arch={arch}, {len(reader.tensors)} tensors)")

    writer = gguf.GGUFWriter(str(dst), arch)

    kept = 0
    for key, field in reader.fields.items():
        if not key.startswith(KEEP_PREFIXES):
            continue
        # GGUFReader does not expose a "copy field to writer" helper, so we
        # have to translate the field's value type back through GGUFWriter's
        # typed API. For our tokenizer-only subset that's just strings,
        # uint32 scalars, bool scalars, and arrays of strings / int32.
        val = field.contents()
        ftype = field.types[0] if field.types else None
        if ftype == gguf.GGUFValueType.STRING:
            writer.add_string(key, val)
        elif ftype == gguf.GGUFValueType.UINT32:
            writer.add_uint32(key, int(val))
        elif ftype == gguf.GGUFValueType.INT32:
            writer.add_int32(key, int(val))
        elif ftype == gguf.GGUFValueType.BOOL:
            writer.add_bool(key, bool(val))
        elif ftype == gguf.GGUFValueType.ARRAY:
            # Array element type is types[1].
            elem = field.types[1]
            if elem == gguf.GGUFValueType.STRING:
                writer.add_array(key, list(val))
            elif elem in (gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32):
                writer.add_array(key, [int(x) for x in val])
            else:
                print(f"[strip] skipping array key {key} (elem type {elem})")
                continue
        else:
            print(f"[strip] skipping key {key} (unsupported type {ftype})")
            continue
        kept += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    # write_tensors_to_file() is omitted — we have no tensors.
    writer.close()

    out_size = dst.stat().st_size
    print(f"[strip] wrote {dst} (kept {kept} KV pairs, {out_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
