"""Pinned model assets shared by runtime loading and desktop release checks."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path


TABICL_CHECKPOINT_NAME = "tabicl-classifier-v2-20260212.ckpt"
TABICL_CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"


def verified_tabicl_checkpoint(resource_root: Path) -> Path:
    """Return the local checkpoint after checking existence and its pinned hash.

    Raises ValueError for a missing or mismatched asset. Never downloads a file or
    falls back to a different model. Filesystem access errors propagate to callers.
    """
    path = resource_root / "assets" / "tabicl" / TABICL_CHECKPOINT_NAME
    if not path.is_file():
        raise ValueError("缺少随App提供的TabICLv2权重，请重新安装完整版。")
    stat = path.stat()
    _verify_checkpoint(str(path.resolve()), stat.st_size, stat.st_mtime_ns, TABICL_CHECKPOINT_SHA256)
    return path


@lru_cache(maxsize=4)
def _verify_checkpoint(path: str, size: int, modified: int, expected_sha256: str) -> None:
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError("TabICLv2权重校验失败，拒绝训练、加载或发布。")
