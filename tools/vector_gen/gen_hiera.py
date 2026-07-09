"""
gen_hiera.py

生成 SAM2 Hiera backbone 的数值校验测试向量（随机权重 + 随机输入 + Python 前向输出）。

用法:
    python gen_hiera.py [--variant tiny|large] [--out-dir <dir>]

输出文件（默认放在 SAM_TorchSharp/testdata/hiera_<variant>/ 下）:
    weights.safetensors   模块 state_dict
    input.safetensors     输入张量 {"x": [B,3,H,W]}
    output_py.safetensors 各 stage 输出 {"feat_0":..., "feat_1":..., ...}（从高分�辨率到低分辨率，
                           与 .NET Hiera.forward 返回顺序保持一致）
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


VARIANTS = {
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
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="tiny")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--image-size", type=int, default=256, help="使用较小分辨率加快随机权重下的测试速度")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = VARIANTS[args.variant]
    model = Hiera(**cfg)
    model.eval()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", f"hiera_{args.variant}"
    )
    os.makedirs(out_dir, exist_ok=True)

    x = torch.randn(1, 3, args.image_size, args.image_size)
    with torch.no_grad():
        outputs = model(x)

    save_state_dict(model, os.path.join(out_dir, "weights.safetensors"))
    save_tensor_dict({"x": x}, os.path.join(out_dir, "input.safetensors"))

    out_dict = {f"feat_{i}": t for i, t in enumerate(outputs)}
    save_tensor_dict(out_dict, os.path.join(out_dir, "output_py.safetensors"))

    print(f"variant={args.variant} image_size={args.image_size}")
    print(f"channel_list (低->高分辨率倒序, 即 stage4..stage1) = {model.channel_list}")
    for i, t in enumerate(outputs):
        print(f"  feat_{i}: shape={tuple(t.shape)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
