"""Pinned weights must be present and verified in source and frozen layouts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import pytest

from sensor_app import model_assets
from tilt_predictor import crossline_probability


def checkpoint_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = root / "assets" / "tabicl" / model_assets.TABICL_CHECKPOINT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test model weights")
    monkeypatch.setattr(model_assets, "TABICL_CHECKPOINT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_missing_checkpoint_is_rejected_without_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="缺少随App提供"):
        model_assets.verified_tabicl_checkpoint(tmp_path)


def test_validated_file_is_returned_and_changed_file_is_rechecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = checkpoint_fixture(tmp_path, monkeypatch)
    assert model_assets.verified_tabicl_checkpoint(tmp_path) == path
    before = path.stat()
    path.write_bytes(b"fake model weights")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="校验失败"):
        model_assets.verified_tabicl_checkpoint(tmp_path)


def test_frozen_runtime_resolves_bundled_weight_not_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "App.app" / "Contents" / "Frameworks"
    path = checkpoint_fixture(root, monkeypatch)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(root), raising=False)
    assert crossline_probability.checkpoint_path() == path
    path.unlink()
    with pytest.raises(ValueError, match="缺少随App提供"):
        crossline_probability.checkpoint_path()


def test_source_runtime_resolves_source_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = checkpoint_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(
        crossline_probability, "__file__",
        str(tmp_path / "src" / "tilt_predictor" / "crossline_probability.py"),
    )
    assert crossline_probability.checkpoint_path() == path
