"""Resolve and safely initialize the writable desktop application data root."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


APPLICATION_NAME = "CSG124 Sensor Platform"
MIGRATION_VERSION = 1
MIGRATION_MARKER = ".migration.json"
MIGRATED_DIRECTORIES = (
    "reports",
    "backups",
    "logs",
    "configs",
    "date",
)


@dataclass(frozen=True)
class DataRootStatus:
    """Describe how the runtime data root was selected or initialized."""

    root: Path
    migrated: bool
    existing: bool
    source_root: Path | None


def default_data_root() -> Path:
    """Return the macOS application support directory for this application."""

    override = os.environ.get("SENSOR_APP_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    application_name = os.environ.get(
        "CSG124_APPLICATION_NAME",
        APPLICATION_NAME,
    ).strip()
    if not application_name:
        application_name = APPLICATION_NAME
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / application_name
    ).resolve()


def resolve_runtime_root(project_root: Path) -> Path:
    """Resolve an explicit runtime root while preserving source-tree development."""

    override = os.environ.get("SENSOR_APP_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return project_root.expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(source_root: Path) -> list[dict[str, str | int]]:
    files: list[dict[str, str | int]] = []
    for directory in MIGRATED_DIRECTORIES:
        source_directory = source_root / directory
        if not source_directory.is_dir():
            continue
        for path in sorted(item for item in source_directory.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": str(path.relative_to(source_root)),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return files


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def initialize_data_root(source_root: Path, target_root: Path) -> DataRootStatus:
    """Migrate a source tree once, without overwriting an existing target tree.

    The copy is assembled in a sibling temporary directory and renamed into place
    only after every requested directory and the migration manifest are complete.
    """

    source_root = source_root.expanduser().resolve()
    target_root = target_root.expanduser().resolve()
    marker = target_root / MIGRATION_MARKER
    if marker.exists() or any(target_root.iterdir()) if target_root.exists() else False:
        return DataRootStatus(
            root=target_root,
            migrated=False,
            existing=True,
            source_root=None,
        )

    target_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target_root.name}.migration-", dir=target_root.parent)
    )
    try:
        for directory in MIGRATED_DIRECTORIES:
            source_directory = source_root / directory
            if source_directory.is_dir():
                shutil.copytree(source_directory, staging / directory)
        manifest = {
            "migration_version": MIGRATION_VERSION,
            "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "source_files": _source_manifest(source_root),
        }
        _write_json_atomically(staging / MIGRATION_MARKER, manifest)
        os.replace(staging, target_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DataRootStatus(
        root=target_root,
        migrated=True,
        existing=False,
        source_root=source_root,
    )
