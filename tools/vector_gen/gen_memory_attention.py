"""
gen_memory_attention.py

生成 SAM2 MemoryAttention 的数值校验测试向量。
模拟真实使用场景：当前帧特征(64x64=4096 token) attend 到 3 帧历史 memory(每帧4096 token)
+ 4 个 object pointer token（不参与 RoPE）。

用法:
    python gen_memory_attention.py [--out-dir <dir>]
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
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention


def build_model(seed: int = 0) -> MemoryAttention:
    set_seed(seed)
    self_attn = RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1, rope_theta=10000.0, feat_sizes=[64, 64])
    cross_attn = RoPEAttention(
        embedding_dim=256, num_heads=1, downsample_rate=1, rope_theta=10000.0,
        feat_sizes=[64, 64], rope_k_repeat=True, kv_in_dim=64,
    )
    layer = MemoryAttentionLayer(
        activation="relu", dim_feedforward=2048, dropout=0.1,
        pos_enc_at_attn=False, self_attention=self_attn,
        d_model=256, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attn,
    )
    model = MemoryAttention(d_model=256, pos_enc_at_input=True, layer=layer, num_layers=4)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "memory_attention"
    )
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(args.seed)
    save_state_dict(model, os.path.join(out_dir, "weights.safetensors"))

    torch.manual_seed(args.seed + 1)
    N = 64 * 64
    num_mem_frames = 3
    num_obj_ptr_tokens = 4

    curr = torch.randn(N, 1, 256)
    curr_pos = torch.randn(N, 1, 256)
    # memory 的通道数取决于 cross_attn_image 的 kv_in_dim（这里是 MemoryEncoder.out_dim=64）
    memory = torch.randn(N * num_mem_frames + num_obj_ptr_tokens, 1, 64)
    memory_pos = torch.randn(N * num_mem_frames + num_obj_ptr_tokens, 1, 64)

    with torch.no_grad():
        out = model(
            curr=curr, memory=memory, curr_pos=curr_pos, memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )

    save_tensor_dict(
        {"curr": curr, "memory": memory, "curr_pos": curr_pos, "memory_pos": memory_pos},
        os.path.join(out_dir, "input.safetensors"),
    )
    save_tensor_dict({"output": out}, os.path.join(out_dir, "output_py.safetensors"))

    print(f"output={tuple(out.shape)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
