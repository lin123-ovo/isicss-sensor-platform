"""Select the file or MySQL model registry explicitly."""

from __future__ import annotations

from pathlib import Path

from .model_registry import ModelRegistryError


def get_model_registry(project_root: Path):
    """Return the MySQL model registry; local model indexes are forbidden."""

    from os import getenv

    if getenv("CSG124_STORAGE_BACKEND", "mysql").strip().lower() != "mysql":
        raise ModelRegistryError(
            "CSG124 桌面 App 只支持 MySQL，禁止使用本地 CSV 模型注册表。"
        )
    from .mysql_model_registry import MySQLModelRegistry

    return MySQLModelRegistry(project_root)
