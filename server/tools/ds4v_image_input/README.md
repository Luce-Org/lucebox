# DS4V image transport unit

This is an unintegrated parser and log-redaction component. It does not decode
JPEG/PNG pixels, invoke a model, fetch URLs, or add an HTTP endpoint.

`parse_image_data_url` accepts canonical padded base64 data URLs with media type
`image/jpeg` or `image/png`. It rejects malformed alphabet/padding, nonzero unused
padding bits, empty payloads, incorrect media signatures, and oversized payloads
before publishing an output. JPEG/PNG structural validity belongs to the native
decoder; transport tests intentionally include signature-only inputs.

`extract_chat_images` retains the order of user text and image parts and substitutes
the parent's `<｜deepseek_image｜>` placeholder for each image. It returns owned
JPEG/PNG bytes separately. Defaults are16MiB per encoded image,32MiB aggregate,
and four images. These limits count image bytes after base64 decoding, before
pixel decoding. Output is cleared on failure and input JSON is unchanged.
Literal image placeholders in all message string values are rejected, including
tool-call fields and placeholders split across adjacent text parts. Message nesting
is limited to64levels by an iterative walk before copying JSON. Images in other message roles are rejected.
The standard `detail` values are accepted; source preprocessing uses its fixed
model recipe for all three values.

Only base64 data URLs are supported in this first unit. Remote HTTP(S), filesystem
paths, other media types, and other APIs' image part schemas fail explicitly.
Text-only message arrays preserve their JSON representation.

`redact_image_urls` iteratively replaces image_url fields before a caller serializes status
or diagnostic JSON. Request integration must invoke it before dumping messages
or raw bodies. This helper alone does not prove that a live server is redacted.

The pure unit was tested on soulf: RED at0306a3f (`valid JPEG transport rejected`),
GREEN ate09f8da. Tests cover canonical JPEG/PNG bytes, malformed base64, per-image
and aggregate/count limits, exact limit boundaries, content order, input immutability,
placeholder injection, malformed descriptors, failure cleanup, and nested redaction.

```sh
cmake -S server/tools/ds4v_image_input -B /tmp/ds4v-transport-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ds4v-transport-build -j2
ctest --test-dir /tmp/ds4v-transport-build --output-on-failure
```

The standalone CMake target uses the same pinned nlohmann/json commit as the server.
It needs no GPU SDK or GGML. All builds and tests ran on soulf in
`~/lucebox-ds4v-transport`; evidence is in `artifacts/transport/` there.
