"""
repo_paths.py

将同一工作区下的 sam2 / sam3 源码目录加入 sys.path，方便测试向量生成脚本
直接 `import sam2` / `import sam3` 而不需要 pip install -e。

假设的目录布局（与当前工作区一致）：

    SAMALL/
      sam2/            <- 官方 sam2 仓库（含 sam2/ 包）
      sam3/             <- 官方 sam3 仓库（含 sam3/ 包）
      SAM_TorchSharp/   <- 本项目
        tools/vector_gen/repo_paths.py  <- 本文件

如果实际路径不同，可通过环境变量覆盖：
    SAM2_REPO_PATH, SAM3_REPO_PATH
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))  # SAMALL/

SAM2_REPO_PATH = os.environ.get("SAM2_REPO_PATH", os.path.join(_DEFAULT_ROOT, "sam2"))
SAM3_REPO_PATH = os.environ.get("SAM3_REPO_PATH", os.path.join(_DEFAULT_ROOT, "sam3"))


def add_sam2_to_path() -> None:
    if SAM2_REPO_PATH not in sys.path:
        sys.path.insert(0, SAM2_REPO_PATH)


def add_sam3_to_path() -> None:
    if SAM3_REPO_PATH not in sys.path:
        sys.path.insert(0, SAM3_REPO_PATH)
