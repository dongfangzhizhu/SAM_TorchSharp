"""
gen_memory_encoder.py

生成 SAM2 MemoryEncoder(含 MaskDownSampler/CXBlock/Fuser) 的数值校验测试向量。

用法:
    python gen_memory_encoder.py [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_paths import add_sam2_to_path
from safetensors_io import set_seed, save_state_dict, save_tensor_dict

add_sam2_to_path()

import torch
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from sam2.modeling.position_encoding import PositionEmbeddingSine


def build_model(seed: int = 0) -> MemoryEncoder:
    set_seed(seed)
    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    fuser = Fuser(
        layer=CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True),
        num_layers=2,
    )
    position_encoding = PositionEmbeddingSine(num_pos_feats=64, normalize=True)
    model = MemoryEncoder(out_dim=64, mask_downsampler=mask_downsampler, fuser=fuser, position_encoding=position_encoding)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "memory_encoder"
    )
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(args.seed)
    save_state_dict(model, os.path.join(out_dir, "weights.safetensors"))

    torch.manual_seed(args.seed + 1)
    pix_feat = torch.randn(1, 256, 64, 64)
    masks = torch.randn(1, 1, 1024, 1024)  # 原始分辨率 mask logits，未过 sigmoid

    with torch.no_grad():
        out = model(pix_feat, masks, skip_mask_sigmoid=False)

    save_tensor_dict({"pix_feat": pix_feat, "masks": masks}, os.path.join(out_dir, "input.safetensors"))
    save_tensor_dict(
        {"vision_features": out["vision_features"], "vision_pos_enc_0": out["vision_pos_enc"][0]},
        os.path.join(out_dir, "output_py.safetensors"),
    )

    print(f"vision_features={tuple(out['vision_features'].shape)} vision_pos_enc_0={tuple(out['vision_pos_enc'][0].shape)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
