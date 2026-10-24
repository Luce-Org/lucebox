#!/usr/bin/env python3
"""
Convert MOSS-Image to the GGUF that luce_server serves images from.

Inputs are the DiT, either the bf16 release (moss_image_sft_bf16.safetensors) or its int8 convrot form
(moss_image_sft_int8_convrot.safetensors), and the FLUX.2 VAE (flux2-vae.safetensors). Both DiT inputs give the same
GGUF (general.architecture "moss-image"), which holds:
  dit.*  the DiT: the four big linears of every block (qkv, o, ffn_up_gate, ffn_down) int8 with an f32 per-row
         weight_scale, bf16 everything else (the comfy_quant markers and the unused rope.bands are left out)
  vae.*  the VAE decoder: convolution weights f16 (rounded to nearest even), biases and norms f32, and
         vae.latents_mean / vae.latents_std, stable-diffusion.cpp's FLUX.2 latent statistics
The Qwen3.5-9B text encoder stays its own GGUF, given to luce_server with --text-encoder.

The int8 linears use the convrot layout: each 256-wide input group of a weight row is rotated by the normalized
Hadamard matrix H4^(x4), the transform the engine applies to the activations, and the row is quantized with one scale.
The rotation sums are exact in float64 and rounded once to float32, so the result does not depend on the machine.

Usage:
  python convert_moss_image.py --dit moss_image_sft_bf16.safetensors \
      --vae flux2-vae.safetensors --out moss-image.gguf
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "llama.cpp" / "gguf-py"))
import gguf

ARCH = "moss-image"
N_LAYERS = 42
GROUP = 256  # convrot rotation width
TARGETS = (".proj.qkv.weight", ".proj.o.weight", ".proj.ffn_up_gate.weight", ".proj.ffn_down.weight")

# Per patchified latent channel (128), as stable-diffusion.cpp's AutoEncoderKL decodes FLUX.2 latents.
LATENTS_MEAN = [
    -0.0676, -0.0715, -0.0753, -0.0745, 0.0223, 0.0180, 0.0142, 0.0184,
    -0.0001, -0.0063, -0.0002, -0.0031, -0.0272, -0.0281, -0.0276, -0.0290,
    -0.0769, -0.0672, -0.0902, -0.0892, 0.0168, 0.0152, 0.0079, 0.0086,
    0.0083, 0.0015, 0.0003, -0.0043, -0.0439, -0.0419, -0.0438, -0.0431,
    -0.0102, -0.0132, -0.0066, -0.0048, -0.0311, -0.0306, -0.0279, -0.0180,
    0.0030, 0.0015, 0.0126, 0.0145, 0.0347, 0.0338, 0.0337, 0.0283,
    0.0020, 0.0047, 0.0047, 0.0050, 0.0123, 0.0081, 0.0081, 0.0146,
    0.0681, 0.0679, 0.0767, 0.0732, -0.0462, -0.0474, -0.0392, -0.0511,
    -0.0528, -0.0477, -0.0470, -0.0517, -0.0317, -0.0316, -0.0345, -0.0283,
    0.0510, 0.0445, 0.0578, 0.0458, -0.0412, -0.0458, -0.0487, -0.0467,
    -0.0088, -0.0106, -0.0088, -0.0046, -0.0376, -0.0432, -0.0436, -0.0499,
    0.0118, 0.0166, 0.0203, 0.0279, 0.0113, 0.0129, 0.0016, 0.0072,
    -0.0118, -0.0018, -0.0141, -0.0054, -0.0091, -0.0138, -0.0145, -0.0187,
    0.0323, 0.0305, 0.0259, 0.0300, 0.0540, 0.0614, 0.0495, 0.0590,
    -0.0511, -0.0603, -0.0478, -0.0524, -0.0227, -0.0274, -0.0154, -0.0255,
    -0.0572, -0.0565, -0.0518, -0.0496, 0.0116, 0.0054, 0.0163, 0.0104,
]
LATENTS_STD = [
    1.8029, 1.7786, 1.7868, 1.7837, 1.7717, 1.7590, 1.7610, 1.7479,
    1.7336, 1.7373, 1.7340, 1.7343, 1.8626, 1.8527, 1.8629, 1.8589,
    1.7593, 1.7526, 1.7556, 1.7583, 1.7363, 1.7400, 1.7355, 1.7394,
    1.7342, 1.7246, 1.7392, 1.7304, 1.7551, 1.7513, 1.7559, 1.7488,
    1.8449, 1.8454, 1.8550, 1.8535, 1.8240, 1.7813, 1.7854, 1.7945,
    1.8047, 1.7876, 1.7695, 1.7676, 1.7782, 1.7667, 1.7925, 1.7848,
    1.7579, 1.7407, 1.7483, 1.7368, 1.7961, 1.7998, 1.7920, 1.7925,
    1.7780, 1.7747, 1.7727, 1.7749, 1.7526, 1.7447, 1.7657, 1.7495,
    1.7775, 1.7720, 1.7813, 1.7813, 1.8162, 1.8013, 1.8023, 1.8033,
    1.7527, 1.7331, 1.7563, 1.7482, 1.7610, 1.7507, 1.7681, 1.7613,
    1.7665, 1.7545, 1.7828, 1.7726, 1.7896, 1.7999, 1.7864, 1.7760,
    1.7613, 1.7625, 1.7560, 1.7577, 1.7783, 1.7671, 1.7810, 1.7799,
    1.7201, 1.7068, 1.7265, 1.7091, 1.7793, 1.7578, 1.7502, 1.7455,
    1.7587, 1.7500, 1.7525, 1.7362, 1.7616, 1.7572, 1.7444, 1.7430,
    1.7509, 1.7610, 1.7634, 1.7612, 1.7254, 1.7135, 1.7321, 1.7226,
    1.7664, 1.7624, 1.7718, 1.7664, 1.7457, 1.7441, 1.7569, 1.7530,
]

DTYPES = {"I8": np.int8, "U8": np.uint8, "F16": np.float16, "BF16": np.uint16, "F32": np.float32, "I64": np.int64}


def read_safetensors(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    """name -> (safetensors dtype, array mapped from the file) in file order; BF16 maps as its raw uint16 bits."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    tensors = {}
    for name, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        begin, end = info["data_offsets"]
        array = data[begin:end].view(DTYPES[info["dtype"]]).reshape(info["shape"])
        tensors[name] = (info["dtype"], array)
    return tensors


def hadamard_256() -> np.ndarray:
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64) / 2
    rot = h4
    for _ in range(3):
        rot = np.kron(rot, h4)
    return rot


def quantize_convrot(bits: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A bf16 linear weight [rows][cols] (raw bits) -> int8 [rows][cols] and the f32 per-row scale [rows]."""
    rows, cols = bits.shape
    w = (bits.astype(np.uint32) << 16).view(np.float32)
    w_rot = (w.astype(np.float64).reshape(rows, cols // GROUP, GROUP) @ rot).astype(np.float32).reshape(rows, cols)
    scale = np.abs(w_rot).max(axis=1) / np.float32(127.0)
    safe = np.where(scale == 0.0, np.float32(1.0), scale)
    q = np.clip(np.rint(w_rot / safe[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


class Plan:
    """The GGUF's tensors in order, each made when it is written, so the int8 weights never sit in memory together."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, tuple[int, ...], np.dtype, object, object]] = []

    def add(self, name: str, shape, dtype, make, raw_dtype=None) -> None:
        self.entries.append((name, tuple(shape), np.dtype(dtype), raw_dtype, make))

    def write(self, writer: gguf.GGUFWriter) -> None:
        for name, shape, dtype, raw_dtype, _ in self.entries:
            writer.add_tensor_info(name, shape, dtype, int(np.prod(shape)) * dtype.itemsize, raw_dtype=raw_dtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for _, _, _, _, make in self.entries:
            writer.write_tensor_data(np.ascontiguousarray(make()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dit", type=Path, required=True,
                        help="moss_image_sft_bf16.safetensors or moss_image_sft_int8_convrot.safetensors")
    parser.add_argument("--vae", type=Path, required=True, help="flux2-vae.safetensors")
    parser.add_argument("--out", type=Path, required=True, help="output GGUF")
    args = parser.parse_args()

    dit = read_safetensors(args.dit)
    vae = read_safetensors(args.vae)
    if f"layers.{N_LAYERS - 1}.proj.qkv.weight" not in dit or dit["layers.0.proj.qkv.weight"][0] not in ("BF16", "I8"):
        sys.exit(f"{args.dit}: not the MOSS-Image DiT (bf16 or int8) with {N_LAYERS} layers")
    if "decoder.conv_in.weight" not in vae:
        sys.exit(f"{args.vae}: no VAE decoder")

    writer = gguf.GGUFWriter(str(args.out), ARCH)
    writer.add_name("MOSS-Image")
    writer.add_block_count(N_LAYERS)
    writer.add_embedding_length(3072)
    writer.add_head_count(24)

    plan = Plan()
    rot = hadamard_256()
    scales: dict[str, np.ndarray] = {}  # the scale of a quantized weight, written right after it

    def quantized(bits: np.ndarray, module: str):
        def make() -> np.ndarray:
            q, scales[module] = quantize_convrot(bits, rot)
            return q
        return make

    for name, (dtype, array) in dit.items():
        if name.endswith(".comfy_quant") or name == "rope.bands":
            continue
        if dtype == "BF16" and name.endswith(TARGETS):
            module = name[: -len(".weight")]
            plan.add(f"dit.{name}", array.shape, np.int8, quantized(array, module))
            plan.add(f"dit.{module}.weight_scale", array.shape[:1], np.float32, lambda module=module: scales.pop(module))
        elif dtype == "BF16":
            plan.add(f"dit.{name}", array.shape, np.uint16, lambda array=array: array, gguf.GGMLQuantizationType.BF16)
        else:
            plan.add(f"dit.{name}", array.shape, array.dtype, lambda array=array: array)

    for name, (dtype, array) in vae.items():
        if not name.startswith("decoder."):
            continue
        if dtype != "F32":
            sys.exit(f"{args.vae}: {name} is {dtype}, expected F32")
        if array.ndim == 4:
            plan.add(f"vae.{name}", array.shape, np.float16, lambda array=array: array.astype(np.float16))
        else:
            plan.add(f"vae.{name}", array.shape, np.float32, lambda array=array: array)
    for name, values in (("vae.latents_mean", LATENTS_MEAN), ("vae.latents_std", LATENTS_STD)):
        plan.add(name, (len(values),), np.float32, lambda values=values: np.array(values, dtype=np.float32))

    plan.write(writer)
    writer.close()
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
