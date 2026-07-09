"""
gen_sam2_real_checkpoint.py

使用真实的 SAM2 / SAM2.1 官方 checkpoint（如 sam2.1_hiera_tiny.pt）和真实图片，
跑一次 SAM2ImagePredictor 的 set_image + predict，导出：
  - 用于 .NET 端加载的 image_encoder/sam_prompt_encoder/sam_mask_decoder/no_mem_embed 权重子集（safetensors）
  - 输入图片（预处理后的 1024x1024 张量）+ 点提示
  - Python 端输出（image_embed、high_res_feats、low_res_masks、iou_predictions、最终 masks）

用法:
    python gen_sam2_real_checkpoint.py --config sam2.1_hiera_t --checkpoint <path/to/sam2.1_hiera_tiny.pt> --image <path/to/image.jpg>
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_paths import add_sam2_to_path
from safetensors_io import save_tensor_dict

add_sam2_to_path()

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="例如 configs/sam2.1/sam2.1_hiera_t.yaml 或简写 sam2.1_hiera_t")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = args.config
    if not config.endswith(".yaml"):
        # 允许简写，自动定位到 sam2/configs/**/xxx.yaml
        candidates = [
            f"configs/sam2.1/{config}.yaml",
            f"configs/sam2/{config}.yaml",
        ]
        found = None
        import sam2 as sam2_pkg

        sam2_root = os.path.dirname(os.path.dirname(sam2_pkg.__file__))
        for c in candidates:
            if os.path.exists(os.path.join(sam2_root, "sam2", c)):
                found = c
                break
        if found is None:
            raise FileNotFoundError(f"找不到配置文件，尝试过: {candidates}")
        config = found

    device = "cpu"
    sam2_model = build_sam2(config, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    image_pil = Image.open(args.image).convert("RGB")
    image_np = np.array(image_pil)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "sam2_real_checkpoint"
    )
    os.makedirs(out_dir, exist_ok=True)

    # 保存图片推理路径实际用到的权重子集
    full_sd = sam2_model.state_dict()
    used_prefixes = ("image_encoder.", "sam_prompt_encoder.", "sam_mask_decoder.", "no_mem_embed")
    used_sd = {k: v for k, v in full_sd.items() if k.startswith(used_prefixes)}
    save_file({k: v.detach().contiguous().float() for k, v in used_sd.items()}, os.path.join(out_dir, "weights.safetensors"))

    predictor.set_image(image_np)

    # 记录预处理后的输入张量（送入 image_encoder 之前的 1024x1024 归一化图像）
    preprocessed = predictor._transforms(image_np)[None, ...].to(device)  # 1x3x1024x1024

    h, w = image_np.shape[:2]
    point_coords_orig = np.array([[w * 0.5, h * 0.5], [w * 0.2, h * 0.2]], dtype=np.float32)
    point_labels = np.array([1, 0], dtype=np.float32)

    with torch.no_grad():
        masks, iou_predictions, low_res_masks = predictor.predict(
            point_coords=point_coords_orig,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=True,
        )

    image_embed = predictor._features["image_embed"]
    high_res_feats = predictor._features["high_res_feats"]

    # 记录变换后（未归一化到 [0,1024]，即原图坐标）的点，供 .NET 端复现同样的坐标变换。
    save_tensor_dict(
        {
            "preprocessed_image": preprocessed,
            "point_coords_orig": torch.as_tensor(point_coords_orig).unsqueeze(0),
            "point_labels": torch.as_tensor(point_labels).unsqueeze(0).float(),
            "orig_hw": torch.tensor([[float(h), float(w)]]),
        },
        os.path.join(out_dir, "input.safetensors"),
    )

    out_dict = {
        "image_embed": image_embed,
        "high_res_feat_0": high_res_feats[0],
        "high_res_feat_1": high_res_feats[1],
        "low_res_masks": torch.as_tensor(low_res_masks),
        "iou_predictions": torch.as_tensor(iou_predictions),
        "masks": torch.as_tensor(masks),
    }
    save_tensor_dict(out_dict, os.path.join(out_dir, "output_py.safetensors"))

    print(f"config={config} checkpoint={args.checkpoint}")
    print(f"image size (h,w)={image_np.shape[:2]}")
    for k, v in out_dict.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
