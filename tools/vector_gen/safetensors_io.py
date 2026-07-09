"""
safetensors_io.py

用于在“Python参考实现 vs .NET(TorchSharp)迁移实现”的逐模块数值校验流程中，
提供统一的 safetensors 导入导出小工具。

约定：
- 所有张量在导出前会被 detach + 转为 float32（除非显式指定 dtype）+ 转到 CPU + 变为 contiguous，
  以保证 .NET 端（TorchSharp.PyBridge.Safetensors）可以无障碍读取。
- 权重文件：与 module.state_dict() 完全一致的 key 命名，可直接被 TorchSharp 的
  `module.load_safetensors(path)` 加载（要求 .NET 侧模块的参数名与 Python 侧一致）。
- 输入/输出文件：key 为自定义的张量名（如 "input", "output_0" 等），仅用于数值比对，
  不要求与 state_dict key 命名规则一致。

典型用法（以后每个模块的生成脚本都遵循这个模式）：

    import torch
    from safetensors_io import save_state_dict, save_tensor_dict, set_seed

    set_seed(0)
    model = SomeModule(...)
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)

    save_state_dict(model, "weights.safetensors")
    save_tensor_dict({"input": x}, "input.safetensors")
    save_tensor_dict({"output": y}, "output_py.safetensors")
"""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, Mapping, Optional, Union

import numpy as np
import torch
from safetensors.torch import save_file, load_file


TensorDict = Dict[str, torch.Tensor]


def set_seed(seed: int = 0) -> None:
    """固定随机种子，保证 Python 端每次生成的随机权重/输入可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize(t: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    t = t.detach()
    if dtype is not None:
        t = t.to(dtype)
    elif t.dtype == torch.float64:
        # TorchSharp.PyBridge 支持 F64，但为减少精度歧义，默认统一到 float32。
        t = t.to(torch.float32)
    return t.to("cpu").contiguous()


def save_tensor_dict(
    tensors: Mapping[str, torch.Tensor],
    path: str,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """保存任意命名的张量字典为 safetensors 文件（用于输入/输出向量）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    clean = {k: _sanitize(v, dtype) for k, v in tensors.items()}
    save_file(clean, path)


def save_state_dict(
    module: torch.nn.Module,
    path: str,
    dtype: Optional[torch.dtype] = None,
    keys_to_keep: Optional[Iterable[str]] = None,
) -> None:
    """保存 module.state_dict() 为 safetensors 文件，供 .NET 端 load_safetensors 使用。"""
    sd = module.state_dict()
    if keys_to_keep is not None:
        keep = set(keys_to_keep)
        sd = {k: v for k, v in sd.items() if k in keep}
    save_tensor_dict(sd, path, dtype=dtype)


def load_tensor_dict(path: str, device: Union[str, torch.device] = "cpu") -> TensorDict:
    """读取 safetensors 文件为张量字典（主要用于比对脚本或调试）。"""
    return load_file(path, device=str(device))


def compare_tensor_dicts(
    a: TensorDict,
    b: TensorDict,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Dict[str, Dict[str, float]]:
    """
    比较两个张量字典（例如 python 输出 vs .NET 输出）的数值差异。
    返回 { key: {"max_abs_diff":.., "max_rel_diff":.., "shape_match": 1/0 } }
    仅用于python侧的辅助自检；正式的.NET<->Python比对由 .NET 端的 SafetensorsCompare 工具完成。
    """
    report: Dict[str, Dict[str, float]] = {}
    keys = set(a.keys()) | set(b.keys())
    for k in sorted(keys):
        if k not in a or k not in b:
            report[k] = {"max_abs_diff": float("inf"), "max_rel_diff": float("inf"), "shape_match": 0.0}
            continue
        ta, tb = a[k].to(torch.float64), b[k].to(torch.float64)
        if ta.shape != tb.shape:
            report[k] = {"max_abs_diff": float("inf"), "max_rel_diff": float("inf"), "shape_match": 0.0}
            continue
        diff = (ta - tb).abs()
        max_abs = diff.max().item() if diff.numel() > 0 else 0.0
        denom = tb.abs().clamp_min(1e-8)
        max_rel = (diff / denom).max().item() if diff.numel() > 0 else 0.0
        report[k] = {"max_abs_diff": max_abs, "max_rel_diff": max_rel, "shape_match": 1.0}
    return report
