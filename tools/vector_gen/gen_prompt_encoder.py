"""
gen_prompt_encoder.py

生成 SAM2 PromptEncoder 的数值校验测试向量。
覆盖三种提示组合：仅点、仅框、点+框+mask，分别生成子目录。

用法:
    python gen_prompt_encoder.py [--out-dir <dir>]
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
from sam2.modeling.sam.prompt_encoder import PromptEncoder


def build_model(seed: int = 0) -> PromptEncoder:
    set_seed(seed)
    model = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )
    model.eval()
    return model


def run_case(name: str, model: PromptEncoder, points, boxes, masks, out_root: str) -> None:
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        sparse, dense = model(points=points, boxes=boxes, masks=masks)

    inputs = {}
    if points is not None:
        inputs["point_coords"] = points[0]
        inputs["point_labels"] = points[1]
    if boxes is not None:
        inputs["boxes"] = boxes
    if masks is not None:
        inputs["masks"] = masks

    save_tensor_dict(inputs, os.path.join(out_dir, "input.safetensors"))
    save_tensor_dict({"sparse_embeddings": sparse, "dense_embeddings": dense}, os.path.join(out_dir, "output_py.safetensors"))
    print(f"[{name}] sparse={tuple(sparse.shape)} dense={tuple(dense.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_root = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "prompt_encoder"
    )
    os.makedirs(out_root, exist_ok=True)

    model = build_model(args.seed)
    save_state_dict(model, os.path.join(out_root, "weights.safetensors"))

    torch.manual_seed(args.seed)
    # Case 1: 仅点（含 1 个正点 + 1 个负点），pad=True（无 box）
    points_coords = torch.rand(1, 2, 2) * 1024
    points_labels = torch.tensor([[1, 0]], dtype=torch.float32)
    run_case("points_only", model, (points_coords, points_labels), None, None, out_root)

    # Case 2: 仅框
    boxes = torch.rand(1, 4) * 1024
    boxes = boxes.reshape(1, 4)
    run_case("boxes_only", model, None, boxes, None, out_root)

    # Case 3: 点 + 框 + mask（此时点不 pad）
    points_coords2 = torch.rand(1, 3, 2) * 1024
    points_labels2 = torch.tensor([[1, 0, 1]], dtype=torch.float32)
    boxes2 = torch.rand(1, 4) * 1024
    masks = torch.randn(1, 1, 256, 256)
    run_case("points_boxes_masks", model, (points_coords2, points_labels2), boxes2, masks, out_root)

    print(f"saved to: {out_root}")


if __name__ == "__main__":
    main()
