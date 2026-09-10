#!/usr/bin/env python3
"""Generate decoded-RGB preprocessing fixtures with the original Python source."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps


START_POSITIONS = (0, 1, 2, 3, 127)


def pattern(width: int, height: int, seed: int) -> bytes:
    y, x = np.indices((height, width), dtype=np.uint32)
    channels = []
    for channel in range(3):
        values = (x * 17 + y * 31 + (x * y) % 251 + channel * 73 + seed) % 256
        channels.append(values.astype(np.uint8))
    return np.stack(channels, axis=2).tobytes()


def source_resize(image, model_args, safe_resize):
    patch = model_args.vision_patch_size
    width, height = image.size
    if model_args.vision_max_wh_ratio is not None and width > height * model_args.vision_max_wh_ratio:
        width = height * model_args.vision_max_wh_ratio
    if 0 < width * height < model_args.vision_min_pixels:
        ratio = (model_args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / patch) * patch
    best_height = math.ceil(height / patch) * patch
    llm_h, llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch,
        model_args.vision_downsample_ratio,
        model_args.vision_max_n_token,
    )
    if image.width >= model_args.vision_max_wh_ratio * image.height:
        resized = image.resize((best_width, best_height))
    else:
        resized = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    return resized, best_height // patch, best_width // patch, llm_h, llm_w


def save_bytes(path: Path, data: bytes, digests: dict[str, str], root: Path):
    path.write_bytes(data)
    digests[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="parent model directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source / "inference"))
    from image_processor import build_image_block, load_image, safe_resize

    config = json.loads((args.source / "config.json").read_text())
    config["dim"] = config["hidden_size"]
    model_args = SimpleNamespace(**config)
    torch.set_num_threads(2)

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    cases: list[tuple[str, Image.Image, bytes]] = []
    synthetic = (
        ("tiny", 3, 5),
        ("odd-padding", 37, 23),
        ("portrait", 41, 113),
        ("wide-direct", 257, 31),
        ("very-tall", 10, 20000),
        ("max-budget", 2048, 354),
    )
    for seed, (label, width, height) in enumerate(synthetic, start=1):
        raw = pattern(width, height, seed * 19)
        image = Image.frombytes("RGB", (width, height), raw)
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        cases.append((label, image, encoded.getvalue()))

    rgba_rgb = np.frombuffer(pattern(29, 17, 211), dtype=np.uint8).reshape(17, 29, 3)
    alpha = ((np.indices((17, 29), dtype=np.uint16).sum(axis=0) * 23) % 256).astype(np.uint8)
    rgba = Image.fromarray(np.dstack((rgba_rgb, alpha)))
    encoded_rgba = io.BytesIO()
    rgba.save(encoded_rgba, format="PNG")
    with Image.open(io.BytesIO(encoded_rgba.getvalue())) as opened:
        expected_rgba_rgb = opened.convert("RGB")
    cases.append(("png-rgba", expected_rgba_rgb, encoded_rgba.getvalue()))

    jpeg_source = Image.frombytes("RGB", (19, 11), pattern(19, 11, 233))
    exif = Image.Exif()
    exif[274] = 6
    encoded_jpeg = io.BytesIO()
    jpeg_source.save(encoded_jpeg, format="JPEG", quality=91, exif=exif)
    with Image.open(io.BytesIO(encoded_jpeg.getvalue())) as opened:
        expected_jpeg_rgb = opened.convert("RGB")
    assert expected_jpeg_rgb.size == (19, 11), "source unexpectedly transposed EXIF orientation"
    cases.append(("jpeg-exif", expected_jpeg_rgb, encoded_jpeg.getvalue()))

    for label in ("carrots", "corn"):
        encoded = (args.source / "inference/examples/images" / f"{label}.jpeg").read_bytes()
        with Image.open(io.BytesIO(encoded)) as opened:
            image = opened.convert("RGB")
        cases.append((label, image, encoded))

    manifest_lines = [
        "label\tinput_width\tinput_height\tresized_width\tresized_height\tvit_rows\tvit_cols\taligner_rows\taligner_cols"
    ]
    digests: dict[str, str] = {}
    for label, image, encoded in cases:
        case_dir = args.output / label
        case_dir.mkdir()
        resized, vit_rows, vit_cols, aligner_rows, aligner_cols = source_resize(
            image, model_args, safe_resize
        )
        patches, source_vit_rows, source_vit_cols, source_aligner_rows, source_aligner_cols = load_image(
            {"data": encoded}, model_args
        )
        assert (vit_rows, vit_cols, aligner_rows, aligner_cols) == (
            source_vit_rows,
            source_vit_cols,
            source_aligner_rows,
            source_aligner_cols,
        )

        # This checks that the separately materialized resized RGB image produces
        # the same BF16 tensor as the original load_image implementation.
        rebuilt = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255
        rebuilt = ((rebuilt - 0.5) / 0.5).to(torch.bfloat16)
        rebuilt = (
            rebuilt.reshape(3, vit_rows, model_args.vision_patch_size, vit_cols, model_args.vision_patch_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(vit_rows * vit_cols, 3, model_args.vision_patch_size, model_args.vision_patch_size)
        )
        assert torch.equal(rebuilt, patches)

        save_bytes(case_dir / "input.rgb", image.tobytes(), digests, args.output)
        save_bytes(case_dir / "encoded.bin", encoded, digests, args.output)
        save_bytes(case_dir / "resized.rgb", resized.tobytes(), digests, args.output)
        save_bytes(
            case_dir / "patches.bf16",
            patches.contiguous().view(torch.uint16).cpu().numpy().tobytes(),
            digests,
            args.output,
        )
        for start in START_POSITIONS:
            types, permutation = build_image_block(aligner_rows, aligner_cols, start)
            save_bytes(
                case_dir / f"types-{start}.i64",
                types.contiguous().cpu().numpy().tobytes(),
                digests,
                args.output,
            )
            save_bytes(
                case_dir / f"permutation-{start}.i64",
                permutation.contiguous().cpu().numpy().tobytes(),
                digests,
                args.output,
            )
        manifest_lines.append(
            "\t".join(
                str(value)
                for value in (
                    label,
                    image.width,
                    image.height,
                    resized.width,
                    resized.height,
                    vit_rows,
                    vit_cols,
                    aligner_rows,
                    aligner_cols,
                )
            )
        )
        print(
            f"{label}: decoded={image.width}x{image.height} resized={resized.width}x{resized.height} "
            f"vit={vit_rows}x{vit_cols} aligner={aligner_rows}x{aligner_cols}",
            flush=True,
        )

    (args.output / "manifest.tsv").write_text("\n".join(manifest_lines) + "\n")
    (args.output / "sha256.json").write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
