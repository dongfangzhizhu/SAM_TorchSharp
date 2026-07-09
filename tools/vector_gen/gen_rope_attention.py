"""
gen_rope_attention.py

生成 SAM2 RoPEAttention 的数值校验测试向量。
覆盖两种典型用法：
  - self_attn: q/k/v 长度一致 (memory_attention 的 self-attention，num_k_exclude_rope=0)
  - cross_attn: k 长度是 q 的整数倍且带有若干 object pointer token 需排除 rope
                (memory_attention 的 cross-attention，kv_in_dim=64, rope_k_repeat=True)

用法:
    python gen_rope_attention.py [--out-dir <dir>]
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
from sam2.modeling.sam.transformer import RoPEAttention


def run_self_attn(out_root: str, seed: int) -> None:
    set_seed(seed)
    model = RoPEAttention(
        embedding_dim=256, num_heads=1, downsample_rate=1,
        rope_theta=10000.0, feat_sizes=[64, 64],
    )
    model.eval()

    out_dir = os.path.join(out_root, "self_attn")
    os.makedirs(out_dir, exist_ok=True)

    N = 64 * 64
    q = torch.randn(1, N, 256)
    k = torch.randn(1, N, 256)
    v = torch.randn(1, N, 256)

    with torch.no_grad():
        out = model(q, k, v)

    save_state_dict(model, os.path.join(out_dir, "weights.safetensors"))
    save_tensor_dict({"q": q, "k": k, "v": v}, os.path.join(out_dir, "input.safetensors"))
    save_tensor_dict({"output": out}, os.path.join(out_dir, "output_py.safetensors"))
    print(f"[self_attn] out={tuple(out.shape)}")


def run_cross_attn(out_root: str, seed: int) -> None:
    set_seed(seed)
    model = RoPEAttention(
        embedding_dim=256, num_heads=1, downsample_rate=1,
        rope_theta=10000.0, feat_sizes=[64, 64],
        rope_k_repeat=True, kv_in_dim=64,
    )
    model.eval()

    out_dir = os.path.join(out_root, "cross_attn")
    os.makedirs(out_dir, exist_ok=True)

    N = 64 * 64
    num_mem_frames = 3
    num_obj_ptr_tokens = 4

    q = torch.randn(1, N, 256)
    k = torch.randn(1, N * num_mem_frames + num_obj_ptr_tokens, 64)
    v = torch.randn(1, N * num_mem_frames + num_obj_ptr_tokens, 64)

    with torch.no_grad():
        out = model(q, k, v, num_k_exclude_rope=num_obj_ptr_tokens)

    save_state_dict(model, os.path.join(out_dir, "weights.safetensors"))
    save_tensor_dict({"q": q, "k": k, "v": v}, os.path.join(out_dir, "input.safetensors"))
    save_tensor_dict({"output": out}, os.path.join(out_dir, "output_py.safetensors"))
    print(f"[cross_attn] out={tuple(out.shape)} num_k_exclude_rope={num_obj_ptr_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_root = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "rope_attention"
    )
    os.makedirs(out_root, exist_ok=True)

    run_self_attn(out_root, args.seed)
    run_cross_attn(out_root, args.seed + 1)

    print(f"saved to: {out_root}")


if __name__ == "__main__":
    main()
