# Parent source how pass

PASS. Source files came from the verified parent checkpoint on soulf. The CPU reference fixtures ran the original Python files with torch 2.10.0+cpu, numpy, Pillow, and safetensors 0.7.0. Both supplied photos produced finite tower outputs and aligned embeddings. The manifest records source file hashes, image hashes, dimensions, and fixture hashes.

The tower has 32 blocks, width 1024, 16 heads, and 14 by 14 RGB patches. Each block has RMSNorm with epsilon 1e-6, combined QKV projection and bias, full bidirectional attention, output projection and bias, then another RMSNorm and a fused gate/up SwiGLU MLP. The final tower norm also uses epsilon 1e-6.

The two-dimensional rotary code uses a 64-dimensional head. It splits the head into two halves, each width 32. Height and width frequencies occupy 16 values each within those halves. It does not use adjacent-pair rotation. The positional phase derives from the row-major patch grid. This must match the reference rather than borrowing a different model's RoPE arrangement.

The aligner pads the patch grid on its bottom and right to a multiple of 3. It gathers nonoverlapping 3 by 3 patches in channel-first unfold order. The resulting input width is 9216. A biased linear maps to 4096, exact GELU follows, and another biased 4096 linear produces language embeddings.

Preprocessing uses the parent's resize budget, aspect-ratio policy, RGB padding, and pixel normalization. It casts normalized pixels to BF16 before arranging channel-major patches. ImageOps.pad and Pillow's resize behavior are part of the numerical reference. A replacement decoder/resizer needs an empirical comparison, not an assumption that bilinear resizing is equivalent.

build_image_block introduces compression-alignment padding based on the current prompt position. It interleaves pairs of image rows in N order and carries a separate permutation for aligned image embeddings. Types are start=0, pad=1, image=2, newline=3, end=4. Sentinel IDs are vocabulary size plus type. All sentinel types use image routing, including padding outside the bidirectional start/end interval.

The reference images are already present on soulf at the parent inference/examples/images directory. Carrots uses a 42 by 61 ViT grid and a 14 by 21 aligner grid. Corn uses a 23 by 34 ViT grid and an 8 by 12 aligner grid. Fixtures cover start positions 0, 1, 2, 3, and 127.

The reference data stays on soulf under ~/lucebox-ds4v-mix-fix/artifacts/vision-reference. Only the manifest and run log have been copied to this Mac.
