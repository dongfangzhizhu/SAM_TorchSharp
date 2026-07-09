"""
gen_image_encoder.py

生成 SAM2 ImageEncoder(trunk=Hiera, neck=FpnNeck) 的数值校验测试向量。

用法:
    python gen_image_encoder.py [--variant tiny|large] [--out-dir <dir>]

输出文件（默认放在 SAM_TorchSharp/testdata/image_encoder_<variant>/ 下）:
    weights.safetensors    ImageEncoder 完整 state_dict（trunk.* + neck.*）
    input.safetensors      输入张量 {"x": [B,3,H,W]}
    output_py.safetensors  {
        "vision_features": [B,d_model,H,W],
        "backbone_fpn_0..N": 每层融合后的特征,
        "vision_pos_enc_0..N": 每层对应的位置编码,
    }
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
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from sam2.modeling.position_encoding import PositionEmbeddingSine


HIERA_VARIANTS = {
    "tiny": dict(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 7, 2),
        global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
        window_spec=(8, 4, 14, 7),
    ),
    "large": dict(
        embed_dim=144,
        num_heads=2,
        stages=(2, 6, 36, 4),
        global_att_blocks=(23, 33, 43),
        window_pos_embed_bkg_spatial_size=(7, 7),
        window_spec=(8, 4, 16, 8),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(HIERA_VARIANTS.keys()), default="tiny")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    trunk = Hiera(**HIERA_VARIANTS[args.variant])
    backbone_channel_list = trunk.channel_list  # 低分辨率->高分辨率倒序，如 [768,384,192,96]

    position_encoding = PositionEmbeddingSine(num_pos_feats=args.d_model, normalize=True)
    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=args.d_model,
        backbone_channel_list=backbone_channel_list,
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )

    model = ImageEncoder(trunk=trunk, neck=neck, scalp=1)
    model.eval()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", f"image_encoder_{args.variant}"
    )
    os.makedirs(out_dir, exist_ok=True)

    x = torch.randn(1, 3, args.image_size, args.image_size)
    with torch.no_grad():
        out = model(x)

    save_state_dict(model, os.path.join(out_dir, "weights.safetensors"))
    save_tensor_dict({"x": x}, os.path.join(out_dir, "input.safetensors"))

    out_dict = {"vision_features": out["vision_features"]}
    for i, t in enumerate(out["backbone_fpn"]):
        out_dict[f"backbone_fpn_{i}"] = t
    for i, t in enumerate(out["vision_pos_enc"]):
        out_dict[f"vision_pos_enc_{i}"] = t
    save_tensor_dict(out_dict, os.path.join(out_dir, "output_py.safetensors"))

    print(f"variant={args.variant} image_size={args.image_size} d_model={args.d_model}")
    print(f"backbone_channel_list={backbone_channel_list}")
    print(f"vision_features: {tuple(out['vision_features'].shape)}")
    for i, t in enumerate(out["backbone_fpn"]):
        print(f"  backbone_fpn_{i}: {tuple(t.shape)}")
    for i, t in enumerate(out["vision_pos_enc"]):
        print(f"  vision_pos_enc_{i}: {tuple(t.shape)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
