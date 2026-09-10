# DeepSeek-V4 vision preprocessing probe

This standalone CPU target verifies the reusable decoded-RGB preprocessing unit without configuring the server or a GPU SDK. It implements the fixed source recipe only: patch size 14, downsample ratio 3, maximum 384 layout tokens, minimum 147456 planned pixels, maximum wide-image ratio 8, and mean/std 0.5 normalization.

Generate reference fixtures on a machine containing the original parent source and its Python environment:

```sh
python generate_reference_fixtures.py \
  --source /path/to/DeepSeek-V4-Flash-Vision-Uncensored \
  --output /tmp/ds4v-preprocess-fixtures
```

Build and run the probe. Its default codec gate downloads the pinned source archives for libjpeg-turbo 3.1.4.1 and LodePNG commit `ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
./build/ds4v_preprocess_probe --fixtures /tmp/ds4v-preprocess-fixtures
```

The fixture check compares JPEG/PNG decoded RGB, including PNG alpha removal and a JPEG EXIF orientation that must not be transposed, resized RGB bytes, BF16 patch words, dimensions, all five grounded start-position layouts, permutations, spans, and second deterministic decode and preprocessing runs. The built-in self-test covers malformed and truncated JPEG/PNG, unsupported formats, encoded and decoded bounds, invalid fixed config, invalid decoded dimensions, wrong RGB byte counts, output bounds, layout budget overflow, and absolute-position overflow.

The production boundary remains split: `decode_image` owns bounded JPEG/PNG byte decoding, while `preprocess_rgb` accepts already decoded interleaved RGB and has no image-codec or Python runtime dependency. Configure with `-DDS4V_PREPROCESS_WITH_CODECS=OFF` to build and test the RGB unit without downloading codec dependencies.

PNG decoding sets the IDAT inflation limit to the checked filtered scanline size derived from the validated IHDR color depth and interlace layout. LodePNG may retain capacity up to roughly 1.5 times that limit plus one 65,535-byte uncompressed DEFLATE block before reporting the limit. Text and unknown-chunk storage stay disabled; CRC, Adler-32, and DEFLATE length checks stay enabled; ICC decompression retains LodePNG's 16 MiB cap. GREY16 samples follow Pillow's `I;16` RGB conversion by clamping values above 255. CMYK and YCCK JPEG inputs return `unsupported_format`.

The resampler follows Pillow 12.3.0 `src/libImaging/Resample.c`, including signed 22-bit fixed-point bicubic coefficients and the tall-image vertical-first path.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the Pillow-derived resampler notice and the notices for statically linked codec dependencies.
