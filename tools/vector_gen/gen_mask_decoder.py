"""
gen_mask_decoder.py

生成 SAM2 MaskDecoder(含TwoWayTransformer) 的数值校验测试向量。
覆盖两种典型配置：
  - full: 与 sam2.1_hiera_l 配置一致（pred_obj_scores, use_high_res_features, dynamic_multimask_via_stability 等全部开启）
  - multimask_output: 同上配置，但 forward 时 multimask_output=True

用法:
    python gen_mask_decoder.py [--out-dir <dir>]
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
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.transformer import TwoWayTransformer


def build_model(seed: int = 0) -> MaskDecoder:
    set_seed(seed)
    transformer = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)
    model = MaskDecoder(
        transformer_dim=256,
        transformer=transformer,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=True,
        iou_prediction_use_sigmoid=True,
        dynamic_multimask_via_stability=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,
    )
    model.eval()
    return model


def run_case(name: str, model: MaskDecoder, multimask_output: bool, out_root: str, seed: int) -> None:
    torch.manual_seed(seed)
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    image_embeddings = torch.randn(1, 256, 64, 64)
    image_pe = torch.randn(1, 256, 64, 64)
    sparse_prompt_embeddings = torch.randn(1, 3, 256)
    dense_prompt_embeddings = torch.randn(1, 256, 64, 64)
    feat_s0 = torch.randn(1, 32, 256, 256)
    feat_s1 = torch.randn(1, 64, 128, 128)

    with torch.no_grad():
        masks, iou_pred, sam_tokens_out, object_score_logits = model(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=[feat_s0, feat_s1],
        )

    save_tensor_dict(
        {
            "image_embeddings": image_embeddings,
            "image_pe": image_pe,
            "sparse_prompt_embeddings": sparse_prompt_embeddings,
            "dense_prompt_embeddings": dense_prompt_embeddings,
            "feat_s0": feat_s0,
            "feat_s1": feat_s1,
        },
        os.path.join(out_dir, "input.safetensors"),
    )
    save_tensor_dict(
        {
            "masks": masks,
            "iou_pred": iou_pred,
            "sam_tokens_out": sam_tokens_out,
            "object_score_logits": object_score_logits,
        },
        os.path.join(out_dir, "output_py.safetensors"),
    )
    print(f"[{name}] masks={tuple(masks.shape)} iou_pred={tuple(iou_pred.shape)} "
          f"sam_tokens_out={tuple(sam_tokens_out.shape)} object_score_logits={tuple(object_score_logits.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_root = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "mask_decoder"
    )
    os.makedirs(out_root, exist_ok=True)

    model = build_model(args.seed)
    save_state_dict(model, os.path.join(out_root, "weights.safetensors"))

    run_case("single_mask", model, multimask_output=False, out_root=out_root, seed=args.seed + 1)
    run_case("multi_mask", model, multimask_output=True, out_root=out_root, seed=args.seed + 1)

    print(f"saved to: {out_root}")


if __name__ == "__main__":
    main()
