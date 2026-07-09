"""
gen_sam2_image_pipeline.py

生成 SAM2 图片推理管线的端到端数值校验测试向量。
构造一个精简的 SAM2Base（不含 memory_attention/memory_encoder，
用 num_maskmem=0 等价于 SAM2ImagePredictor 的图片推理路径），
使用与 SAM2ImagePredictor.set_image + predict 等价的手写流程生成测试向量。

用法:
    python gen_sam2_image_pipeline.py [--variant tiny|large] [--out-dir <dir>]
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
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock


HIERA_VARIANTS = {
    "tiny": dict(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 7, 2),
        global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
        window_spec=(8, 4, 14, 7),
    ),
}


def build_model(variant: str, image_size: int, seed: int) -> SAM2Base:
    set_seed(seed)

    trunk = Hiera(**HIERA_VARIANTS[variant])
    position_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True)
    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=trunk.channel_list,
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )
    image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)

    # 图片管线不需要真正 track，因此 memory_attention/memory_encoder 仅用于满足
    # SAM2Base 构造函数签名，构造后不会被 forward_image/_prepare_backbone_features 使用。
    self_attn = RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1, rope_theta=10000.0, feat_sizes=[64, 64])
    cross_attn = RoPEAttention(
        embedding_dim=256, num_heads=1, downsample_rate=1, rope_theta=10000.0,
        feat_sizes=[64, 64], rope_k_repeat=True, kv_in_dim=64,
    )
    mem_layer = MemoryAttentionLayer(
        activation="relu", dim_feedforward=2048, dropout=0.1,
        pos_enc_at_attn=False, self_attention=self_attn,
        d_model=256, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attn,
    )
    memory_attention = MemoryAttention(d_model=256, pos_enc_at_input=True, layer=mem_layer, num_layers=4)

    mem_position_encoding = PositionEmbeddingSine(num_pos_feats=64, normalize=True)
    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    fuser = Fuser(layer=CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True), num_layers=2)
    memory_encoder = MemoryEncoder(out_dim=64, position_encoding=mem_position_encoding, mask_downsampler=mask_downsampler, fuser=fuser)

    model = SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=image_size,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        no_obj_embed_spatial=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        sam_mask_decoder_extra_args=dict(
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        ),
    )
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(HIERA_VARIANTS.keys()), default="tiny")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", f"sam2_image_{args.variant}"
    )
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(args.variant, args.image_size, args.seed)

    # 只保存图片推理路径实际用到的子模块权重（image_encoder, sam_prompt_encoder, sam_mask_decoder, no_mem_embed）
    full_sd = model.state_dict()
    used_prefixes = ("image_encoder.", "sam_prompt_encoder.", "sam_mask_decoder.", "no_mem_embed")
    used_sd = {k: v for k, v in full_sd.items() if k.startswith(used_prefixes)}
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous() for k, v in used_sd.items()}, os.path.join(out_dir, "weights.safetensors"))

    torch.manual_seed(args.seed + 1)
    img = torch.randn(1, 3, args.image_size, args.image_size)

    with torch.no_grad():
        backbone_out = model.forward_image(img)
        _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

        bb_feat_sizes = [(args.image_size // 4, args.image_size // 4),
                          (args.image_size // 8, args.image_size // 8),
                          (args.image_size // 16, args.image_size // 16)]
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        high_res_feats = feats[:-1]

        # 模拟一次点提示预测（等价于 SAM2ImagePredictor._predict, multimask_output=True）
        point_coords = torch.rand(1, 2, 2) * args.image_size
        point_labels = torch.tensor([[1, 0]], dtype=torch.int32)

        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

    save_tensor_dict({"x": img, "point_coords": point_coords, "point_labels": point_labels.float()},
                      os.path.join(out_dir, "input.safetensors"))
    out_dict = {
        "image_embed": image_embed,
        "high_res_feat_0": high_res_feats[0],
        "high_res_feat_1": high_res_feats[1],
        "low_res_masks": low_res_masks,
        "iou_predictions": iou_predictions,
    }
    save_tensor_dict(out_dict, os.path.join(out_dir, "output_py.safetensors"))

    print(f"variant={args.variant} image_size={args.image_size}")
    for k, v in out_dict.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
