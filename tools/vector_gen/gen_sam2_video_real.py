"""
gen_sam2_video_real.py

使用真实 SAM2.1 官方 checkpoint 和真实短视频（几帧图片序列），跑通 SAM2VideoPredictor 的
init_state + add_new_points_or_box + propagate_in_video 完整流程，作为 Phase2（视频推理）
的端到端真实数据验证基准。

用法:
    python gen_sam2_video_real.py --config sam2.1_hiera_t --checkpoint <path> --video-dir <帧图片目录> --num-frames 5
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

from sam2.build_sam import build_sam2_video_predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-dir", required=True, help="包含 00000.jpg, 00001.jpg... 的帧图片目录")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = args.config
    if not config.endswith(".yaml"):
        candidates = [f"configs/sam2.1/{config}.yaml", f"configs/sam2/{config}.yaml"]
        import sam2 as sam2_pkg
        sam2_root = os.path.dirname(os.path.dirname(sam2_pkg.__file__))
        found = None
        for c in candidates:
            if os.path.exists(os.path.join(sam2_root, "sam2", c)):
                found = c
                break
        if found is None:
            raise FileNotFoundError(f"找不到配置文件，尝试过: {candidates}")
        config = found

    device = "cpu"
    predictor = build_sam2_video_predictor(config, args.checkpoint, device=device)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "sam2_video_real"
    )
    os.makedirs(out_dir, exist_ok=True)

    # 保存图片推理 + 视频推理用到的完整权重（真实 checkpoint 是严格全量的，直接整份保存即可）
    full_sd = predictor.state_dict()
    save_file({k: v.detach().contiguous().float() for k, v in full_sd.items()}, os.path.join(out_dir, "weights.safetensors"))

    # 只用前 num_frames 帧构造一个"小视频"临时目录
    frame_files = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".jpg"))[: args.num_frames]
    import tempfile
    import shutil
    tmp_video_dir = tempfile.mkdtemp(prefix="sam2_mini_video_")
    orig_images = []
    for i, fname in enumerate(frame_files):
        src = os.path.join(args.video_dir, fname)
        dst = os.path.join(tmp_video_dir, f"{i:05d}.jpg")
        shutil.copyfile(src, dst)
        orig_images.append(np.array(Image.open(src).convert("RGB")))

    try:
        inference_state = predictor.init_state(video_path=tmp_video_dir)
        # 记录预处理后的帧张量(resize到image_size+归一化)，供 .NET 端直接消费，避免重新实现PIL的resize逻辑。
        preprocessed_frames = inference_state["images"].clone()

        h, w = orig_images[0].shape[:2]
        point_coords = np.array([[w * 0.5, h * 0.5], [w * 0.2, h * 0.2]], dtype=np.float32)
        point_labels = np.array([1, 0], dtype=np.float32)

        _, _, masks_frame0 = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=point_coords,
            labels=point_labels,
        )

        all_frame_masks = {0: masks_frame0}
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
            all_frame_masks[frame_idx] = video_res_masks
    finally:
        shutil.rmtree(tmp_video_dir, ignore_errors=True)

    save_tensor_dict(
        {
            "point_coords": torch.as_tensor(point_coords).unsqueeze(0),
            "point_labels": torch.as_tensor(point_labels).unsqueeze(0).float(),
            "preprocessed_frames": preprocessed_frames,
            "orig_hw": torch.tensor([[float(h), float(w)]]),
        },
        os.path.join(out_dir, "input.safetensors"),
    )

    out_dict = {}
    for frame_idx in sorted(all_frame_masks.keys()):
        out_dict[f"frame{frame_idx}_masks"] = torch.as_tensor(all_frame_masks[frame_idx])
    save_tensor_dict(out_dict, os.path.join(out_dir, "output_py.safetensors"))

    print(f"config={config} checkpoint={args.checkpoint}")
    print(f"video frames used: {frame_files}")
    print(f"orig image size (h,w)=({h},{w})")
    for k, v in out_dict.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
