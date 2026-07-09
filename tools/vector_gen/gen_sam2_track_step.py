"""
gen_sam2_track_step.py

生成 SAM2Base.track_step 视频传播的数值校验测试向量。
构造一个完整的 SAM2Base（随机权重），模拟两帧视频：
  - frame 0: 初始条件帧，带一个点提示 (is_init_cond_frame=True)
  - frame 1: 传播帧，不带提示，依赖 frame 0 的 memory (is_init_cond_frame=False)

用法:
    python gen_sam2_track_step.py [--variant tiny] [--image-size 256] [--out-dir <dir>]
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
from sam2.modeling.sam.transformer import TwoWayTransformer, RoPEAttention
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
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
        position_encoding=position_encoding, d_model=256, backbone_channel_list=trunk.channel_list,
        fpn_top_down_levels=[2, 3], fpn_interp_model="nearest",
    )
    image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)

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
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", f"sam2_track_step_{args.variant}"
    )
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(args.variant, args.image_size, args.seed)
    save_state_dict(model, os.path.join(out_dir, "weights.safetensors"))

    torch.manual_seed(args.seed + 1)
    img0 = torch.randn(1, 3, args.image_size, args.image_size)
    img1 = torch.randn(1, 3, args.image_size, args.image_size)

    with torch.no_grad():
        # ---- frame 0: 初始条件帧 ----
        backbone_out0 = model.forward_image(img0)
        _, vision_feats0, vision_pos0, feat_sizes0 = model._prepare_backbone_features(backbone_out0)

        point_coords = torch.rand(1, 2, 2) * args.image_size
        point_labels = torch.tensor([[1, 0]], dtype=torch.int32)
        point_inputs = {"point_coords": point_coords, "point_labels": point_labels}

        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        out0 = model.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats0,
            current_vision_pos_embeds=vision_pos0,
            feat_sizes=feat_sizes0,
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=True,
        )
        output_dict["cond_frame_outputs"][0] = out0

        # ---- frame 1: 传播帧（无提示，依赖 frame0 的 memory）----
        backbone_out1 = model.forward_image(img1)
        _, vision_feats1, vision_pos1, feat_sizes1 = model._prepare_backbone_features(backbone_out1)

        out1 = model.track_step(
            frame_idx=1,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats1,
            current_vision_pos_embeds=vision_pos1,
            feat_sizes=feat_sizes1,
            point_inputs=None,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=True,
        )

    save_tensor_dict(
        {
            "img0": img0, "img1": img1,
            "point_coords": point_coords, "point_labels": point_labels.float(),
        },
        os.path.join(out_dir, "input.safetensors"),
    )

    out_dict = {
        "frame0_pred_masks": out0["pred_masks"],
        "frame0_pred_masks_high_res": out0["pred_masks_high_res"],
        "frame0_obj_ptr": out0["obj_ptr"],
        "frame0_object_score_logits": out0["object_score_logits"],
        "frame0_maskmem_features": out0["maskmem_features"],
        "frame0_maskmem_pos_enc_0": out0["maskmem_pos_enc"][0],
        "frame1_pred_masks": out1["pred_masks"],
        "frame1_pred_masks_high_res": out1["pred_masks_high_res"],
        "frame1_obj_ptr": out1["obj_ptr"],
        "frame1_object_score_logits": out1["object_score_logits"],
        "frame1_maskmem_features": out1["maskmem_features"],
        "frame1_maskmem_pos_enc_0": out1["maskmem_pos_enc"][0],
    }
    save_tensor_dict(out_dict, os.path.join(out_dir, "output_py.safetensors"))

    print(f"variant={args.variant} image_size={args.image_size}")
    for k, v in out_dict.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
