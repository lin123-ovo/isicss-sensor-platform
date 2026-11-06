"""Local registry with immutable model artifacts and editable user notes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator, TypeVar
from uuid import uuid4

import pandas as pd

from .crossline_residual import CrosslineResidualConfig
from . import crossline_probability as probability
from . import crossline_compact36 as compact36
from . import crossline_seven as seven
from . import crossline_nine_regional as nine
from . import crossline_tabicl_single as tabicl_single
from . import crossline_tabicl_16 as tabicl_16
from . import crossline_tabm_single as tabm_single
from .temperature_pipeline import (
    HYBRID_FEATURE_VERSION,
    TemperaturePipelineError,
    normalize_model_parameters,
)


NumberT = TypeVar("NumberT", int, float)


VERSION_COLUMNS = (
    "version_id",
    "created_at_utc",
    "note",
    "gateway",
    "model",
    "epsilon",
    "alpha",
    "inline_epsilon",
    "inline_alpha",
    "crossline_epsilon",
    "crossline_alpha",
    "max_iter",
    "split_ratio",
    "dataset_version",
    "source_sha256",
    "status",
    "model_sha256",
)
ARTIFACT_NAMES = (
    "model.joblib",
    "metadata.json",
    "metrics.csv",
    "parameter_candidates.csv",
    "parameter_search.csv",
    "test_predictions.csv",
    "split_plan.json",
    "selected_parameters.json",
)
HYBRID_ARTIFACT_NAMES = ("crossline_residual_search.csv",)


class ModelRegistryError(RuntimeError):
    """Raised when a version cannot be saved, read, or activated safely."""


@dataclass(frozen=True)
class ModelVersion:
    """One saved model version indexed by the registry."""

    version_id: str
    created_at_utc: str
    note: str
    gateway: str
    model: str
    epsilon: float | None
    alpha: float | None
    inline_epsilon: float | None
    inline_alpha: float | None
    crossline_epsilon: float | None
    crossline_alpha: float | None
    max_iter: int | None
    split_ratio: str
    dataset_version: str
    source_sha256: str
    status: str
    model_sha256: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ModelVersion":
        """Build a typed version from one CSV row."""

        def optional_number(name: str, convert: Callable[[Any], NumberT]) -> NumberT | None:
            value = row[name]
            return None if value is None or value == "" or pd.isna(value) else convert(value)

        return cls(
            version_id=str(row["version_id"]),
            created_at_utc=str(row["created_at_utc"]),
            note=str(row.get("note", "")),
            gateway=str(row["gateway"]),
            model=str(row["model"]),
            epsilon=optional_number("epsilon", float),
            alpha=optional_number("alpha", float),
            inline_epsilon=optional_number("inline_epsilon", float),
            inline_alpha=optional_number("inline_alpha", float),
            crossline_epsilon=optional_number("crossline_epsilon", float),
            crossline_alpha=optional_number("crossline_alpha", float),
            max_iter=optional_number("max_iter", int),
            split_ratio=str(row["split_ratio"]),
            dataset_version=str(row["dataset_version"]),
            source_sha256=str(row["source_sha256"]),
            status=str(row["status"]),
            model_sha256=str(row["model_sha256"]),
        )

    def to_row(self) -> dict[str, Any]:
        """Return a stable row matching ``versions.csv``."""

        return {
            "version_id": self.version_id,
            "created_at_utc": self.created_at_utc,
            "note": self.note,
            "gateway": self.gateway,
            "model": self.model,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "inline_epsilon": self.inline_epsilon,
            "inline_alpha": self.inline_alpha,
            "crossline_epsilon": self.crossline_epsilon,
            "crossline_alpha": self.crossline_alpha,
            "max_iter": self.max_iter,
            "split_ratio": self.split_ratio,
            "dataset_version": self.dataset_version,
            "source_sha256": self.source_sha256,
            "status": self.status,
            "model_sha256": self.model_sha256,
        }


@dataclass(frozen=True)
class DraftRun:
    """Persistent training candidate that is not yet a model version."""

    run_id: str
    created_at_utc: str
    directory: Path


class ModelRegistry:
    """Own model version persistence, migration, and active publication."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.registry_dir = self.project_root / "models" / "registry"
        self.drafts_dir = self.registry_dir / "drafts"
        self.index_path = self.registry_dir / "versions.csv"
        self.active_path = self.registry_dir / "active.json"
        self.lock_path = self.registry_dir / ".registry.lock"
        self.legacy_bootstrap_disabled_path = (
            self.registry_dir / "legacy_bootstrap_disabled.json"
        )

    def bootstrap_current_version(self) -> ModelVersion | None:
        """Register the legacy active Huber model once and return the active version."""

        with self._exclusive_lock():
            versions = self._read_versions()
            if not versions.empty:
                self._persist_direction_columns_if_needed(versions)
                versions = self._migrate_legacy_version_ids_unlocked(versions)
                return self._get_active_unlocked(versions)

            # Cleaning-v2 deliberately invalidates every legacy model.  The
            # migration writes this durable marker so an empty registry stays
            # empty instead of silently resurrecting the standalone artifact.
            if self.legacy_bootstrap_disabled_path.is_file():
                return None

            legacy_model = (
                self.project_root / "models" / "CSG124" / "tilt_temperature_v1.joblib"
            )
            metadata_path = (
                self.project_root
                / "reports"
                / "metrics"
                / "temperature_model_decision.json"
            )
            if not legacy_model.is_file() or not metadata_path.is_file():
                raise ModelRegistryError("找不到可迁移的现有 Huber 模型或训练报告。")
            metadata = self._read_json(metadata_path)
            self._validate_metadata(metadata)
            version_id = "v1"
            version_dir = self.version_directory(version_id)
            self._stage_legacy_version(version_dir, legacy_model, metadata)
            version = self._version_from_metadata(
                version_id,
                metadata,
                note="首次启用版本库时自动迁移",
                model_sha256=self._sha256(version_dir / "model.joblib"),
            )
            migrated_metadata = dict(metadata)
            migrated_metadata["version_id"] = version_id
            migrated_metadata["note"] = version.note
            self._write_json_atomic(version_dir / "metadata.json", migrated_metadata)
            self._write_versions(pd.DataFrame([version.to_row()], columns=VERSION_COLUMNS))
            self._write_json_atomic(
                self.active_path,
                {
                    "schema_version": 1,
                    "version_id": version.version_id,
                    "activated_at_utc": self._now(),
                },
            )
            return version

    def create_training_draft(
        self,
        *,
        model_payload: Any,
        metadata: dict[str, Any],
        metrics: pd.DataFrame,
        parameter_candidates: pd.DataFrame,
        parameter_search: pd.DataFrame,
        test_predictions: pd.DataFrame,
        crossline_residual_search: pd.DataFrame | None = None,
    ) -> DraftRun:
        """Persist one complete training run without changing the active model."""

        self._validate_metadata(metadata)

        if metadata.get('feature_version') in (compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION):
            from .temperature_pipeline import validate_compact36_artifact

            try:
                validate_compact36_artifact(model_payload['model'], metadata)
            except (ValueError, KeyError, TypeError, TemperaturePipelineError) as error:
                raise ModelRegistryError(f'36维组合工件无效：{error}') from error
        if metadata.get('feature_version') == probability.FEATURE_VERSION:
            from .temperature_pipeline import validate_probability_artifact

            try:
                model = model_payload['model']
                validate_probability_artifact(model, metadata)
            except (ValueError, KeyError, TypeError, TemperaturePipelineError) as error:
                raise ModelRegistryError(f'横线概率融合模型工件无效：{error}') from error
        with self._exclusive_lock():
            run_id = self._new_identifier("run")
            draft_dir = self.drafts_dir / run_id
            staging = self._temporary_directory(self.drafts_dir, f".{run_id}-")
            try:
                import joblib

                persisted_metadata = dict(metadata)
                persisted_metadata["run_id"] = run_id
                persisted_predictions = test_predictions.copy()
                persisted_predictions["model_version"] = run_id
                joblib.dump(model_payload, staging / "model.joblib")
                self._write_json_file(staging / "metadata.json", persisted_metadata)
                metrics.to_csv(staging / "metrics.csv", index=False, encoding="utf-8-sig")
                parameter_candidates.to_csv(
                    staging / "parameter_candidates.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                parameter_search.to_csv(
                    staging / "parameter_search.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                persisted_predictions.to_csv(
                    staging / "test_predictions.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                if crossline_residual_search is not None:
                    crossline_residual_search.to_csv(
                        staging / "crossline_residual_search.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                self._write_json_file(
                    staging / "split_plan.json", persisted_metadata["split"]
                )
                self._write_json_file(
                    staging / "selected_parameters.json",
                    self._selected_parameters_document(persisted_metadata),
                )
                self._validate_artifact_directory(staging)
                os.replace(staging, draft_dir)
            except Exception as error:
                shutil.rmtree(staging, ignore_errors=True)
                if isinstance(error, ModelRegistryError):
                    raise
                raise ModelRegistryError(f"训练候选保存失败：{error}") from error
            return DraftRun(run_id, self._now(), draft_dir)

    def create_draft_from_directory(self, source_dir: Path) -> DraftRun:
        """Copy a complete artifact directory into drafts (used by migration/tests)."""

        self._validate_artifact_directory(source_dir)
        with self._exclusive_lock():
            run_id = self._new_identifier("run")
            draft_dir = self.drafts_dir / run_id
            self.drafts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, draft_dir)
            return DraftRun(run_id, self._now(), draft_dir)

    def delete_draft(self, run_id: str) -> None:
        """Permanently remove one unpublished training candidate."""

        self._validate_identifier(run_id, "run")
        with self._exclusive_lock():
            draft_dir = self.drafts_dir / run_id
            if not draft_dir.is_dir():
                raise ModelRegistryError(f"找不到训练候选：{run_id}")
            shutil.rmtree(draft_dir)

    def save_draft(
        self,
        run_id: str,
        note: str = "",
        *,
        activate: bool = False,
        delete_source: bool = False,
    ) -> ModelVersion:
        """Finalize one draft and optionally activate or consume its source."""

        self._validate_identifier(run_id, "run")
        requested_note = self._sanitize_note(note)
        with self._exclusive_lock():
            draft_dir = self.drafts_dir / run_id
            self._validate_artifact_directory(draft_dir)
            metadata = self._read_json(draft_dir / "metadata.json")
            self._validate_metadata(metadata)
            versions = self._read_versions()
            safe_note = requested_note or self._default_version_note(metadata)
            version_id = self._next_version_id(versions)
            version_dir = self.version_directory(version_id)
            staging = self._temporary_directory(self.registry_dir, f".{version_id}-")
            try:
                shutil.copytree(draft_dir, staging, dirs_exist_ok=True)
                predictions_path = staging / "test_predictions.csv"
                predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
                predictions["model_version"] = version_id
                predictions.to_csv(
                    predictions_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                version = self._version_from_metadata(
                    version_id,
                    metadata,
                    safe_note,
                    self._sha256(staging / "model.joblib"),
                )
                version_metadata = dict(metadata)
                version_metadata["version_id"] = version_id
                version_metadata["note"] = safe_note
                self._write_json_file(staging / "metadata.json", version_metadata)
                self._validate_artifact_directory(staging)
                os.replace(staging, version_dir)
            except Exception as error:
                shutil.rmtree(staging, ignore_errors=True)
                if isinstance(error, ModelRegistryError):
                    raise
                raise ModelRegistryError(f"模型版本保存失败：{error}") from error
            versions = pd.concat(
                [versions, pd.DataFrame([version.to_row()])], ignore_index=True
            )
            self._write_versions(versions)
            if activate:
                self._activate_unlocked(version, versions)
            if delete_source:
                shutil.rmtree(draft_dir)
            return version

    def _default_version_note(self, metadata: dict[str, Any]) -> str:
        """Return the split ratio when the user saves a draft without a note."""

        ratio = str(metadata["split"]["requested_ratio"]["label"])
        return self._sanitize_note(ratio)

    def list_versions(self) -> tuple[ModelVersion, ...]:
        """List saved versions from newest to oldest."""

        with self._exclusive_lock():
            versions = self._read_versions()
            if versions.empty:
                return ()
            ordered = versions.assign(
                version_number=versions["version_id"].str[1:].astype(int)
            ).sort_values("version_number", ascending=False)
            return tuple(
                ModelVersion.from_row(row)
                for row in ordered.loc[:, VERSION_COLUMNS].to_dict(orient="records")
            )

    def get_version(self, version_id: str) -> ModelVersion:
        """Read one indexed version and validate that its artifact is intact."""

        self._validate_identifier(version_id, "model")
        with self._exclusive_lock():
            versions = self._read_versions()
            version = self._find_version(versions, version_id)
            self._validate_version_artifact(version)
            return version

    def update_note(self, version_id: str, note: str) -> ModelVersion:
        """Update only a saved version's user note and return the refreshed version."""

        self._validate_identifier(version_id, "model")
        safe_note = self._sanitize_note(note)
        with self._exclusive_lock():
            versions = self._read_versions()
            current = self._find_version(versions, version_id)
            self._validate_version_artifact(current)
            metadata_path = self.version_directory(version_id) / "metadata.json"
            previous_metadata = self._read_json(metadata_path)
            updated_metadata = dict(previous_metadata)
            updated_metadata["note"] = safe_note
            updated_versions = versions.copy()
            matching_rows = updated_versions["version_id"].astype(str) == version_id
            updated_versions.loc[matching_rows, "note"] = safe_note

            try:
                self._write_json_atomic(metadata_path, updated_metadata)
                self._write_versions(updated_versions)
            except Exception as error:
                if not metadata_path.is_file():
                    raise ModelRegistryError(f"模型备注保存失败：{error}") from error
                try:
                    self._write_json_atomic(metadata_path, previous_metadata)
                except Exception as rollback_error:
                    raise ModelRegistryError(
                        "模型备注保存失败，且元数据回滚失败："
                        f"{error}；回滚错误：{rollback_error}"
                    ) from error
                if isinstance(error, ModelRegistryError):
                    raise
                raise ModelRegistryError(f"模型备注保存失败：{error}") from error
            return self._find_version(updated_versions, version_id)

    def activate_version(self, version_id: str) -> ModelVersion:
        """Validate and atomically select a saved model for prediction."""

        self._validate_identifier(version_id, "model")
        with self._exclusive_lock():
            versions = self._read_versions()
            version = self._find_version(versions, version_id)
            self._activate_unlocked(version, versions)
            return version

    def get_active_version(self) -> ModelVersion:
        """Return the canonical active version after integrity validation."""

        with self._exclusive_lock():
            return self._get_active_unlocked(self._read_versions())

    def get_parameters(self, version_id: str) -> dict[str, Any]:
        """Return validated parameters from the immutable version metadata."""

        return normalize_model_parameters(self.get_metadata(version_id)["parameters"])

    def get_metadata(self, version_id: str) -> dict[str, Any]:
        """Return validated immutable metadata for one saved version."""

        version = self.get_version(version_id)
        self._validate_version_artifact(version)
        return self._read_json(self.version_directory(version_id) / "metadata.json")

    def version_directory(self, version_id: str) -> Path:
        """Return the artifact directory for a saved version."""

        return self.registry_dir / version_id

    def draft_directory(self, run_id: str) -> Path:
        """Return the artifact directory for a persistent draft."""

        self._validate_identifier(run_id, "run")
        return self.drafts_dir / run_id

    def list_drafts(self) -> tuple[DraftRun, ...]:
        """Return complete persistent training candidates, newest first."""

        if not self.drafts_dir.is_dir():
            return ()
        drafts: list[DraftRun] = []
        for directory in self.drafts_dir.iterdir():
            if not directory.is_dir() or not directory.name.startswith("run-"):
                continue
            try:
                self._validate_artifact_directory(directory)
                metadata = self._read_json(directory / "metadata.json")
            except ModelRegistryError:
                continue
            drafts.append(
                DraftRun(
                    directory.name,
                    str(metadata.get("created_at_utc", "")),
                    directory,
                )
            )
        return tuple(
            sorted(drafts, key=lambda item: item.created_at_utc, reverse=True)
        )

    def _get_active_unlocked(self, versions: pd.DataFrame) -> ModelVersion:
        if not self.active_path.is_file():
            raise ModelRegistryError("模型版本库缺少活动版本指针。")
        pointer = self._read_json(self.active_path)
        version = self._find_version(versions, str(pointer.get("version_id", "")))
        self._validate_version_artifact(version)
        return version

    def _migrate_legacy_version_ids_unlocked(
        self,
        versions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Replace timestamp-style model IDs with v1, v2, ... exactly once."""

        identifiers = versions["version_id"].astype(str).tolist()
        simple_flags = [self._is_simple_version_id(value) for value in identifiers]
        if all(simple_flags):
            return versions
        if any(simple_flags):
            raise ModelRegistryError("模型版本库同时包含新旧版本号，已停止自动迁移。")
        if not self.active_path.is_file():
            raise ModelRegistryError("旧模型版本库缺少活动版本指针，无法迁移。")

        pointer = self._read_json(self.active_path)
        active_old_id = str(pointer.get("version_id", ""))
        mapping = {
            old_id: f"v{index}"
            for index, old_id in enumerate(identifiers, start=1)
        }
        if active_old_id not in mapping:
            raise ModelRegistryError("活动版本不在版本索引中，无法迁移。")
        if len(mapping) != len(identifiers):
            raise ModelRegistryError("旧模型版本索引存在重复版本号。")

        for old_id in identifiers:
            version = self._find_version(versions, old_id)
            self._validate_version_artifact(version)
        staging_root = self._temporary_directory(
            self.registry_dir,
            ".version-id-migration-",
        )
        created_directories: list[Path] = []
        original_index = self.index_path.read_bytes()
        original_active = self.active_path.read_bytes()
        try:
            for old_id, new_id in mapping.items():
                staged_directory = staging_root / new_id
                shutil.copytree(self.version_directory(old_id), staged_directory)
                metadata_path = staged_directory / "metadata.json"
                metadata = self._read_json(metadata_path)
                metadata["version_id"] = new_id
                self._write_json_file(metadata_path, metadata)
                self._validate_artifact_directory(staged_directory)

            migrated = versions.copy()
            migrated["version_id"] = migrated["version_id"].map(mapping)
            for new_id in mapping.values():
                target = self.version_directory(new_id)
                if target.exists():
                    raise ModelRegistryError(f"迁移目标目录已存在：{new_id}")
                os.replace(staging_root / new_id, target)
                created_directories.append(target)
            self._write_versions(migrated)
            next_pointer = dict(pointer)
            next_pointer["version_id"] = mapping[active_old_id]
            self._write_json_atomic(self.active_path, next_pointer)
        except Exception as error:
            self._write_bytes_atomic(self.index_path, original_index)
            self._write_bytes_atomic(self.active_path, original_active)
            for directory in created_directories:
                shutil.rmtree(directory, ignore_errors=True)
            shutil.rmtree(staging_root, ignore_errors=True)
            if isinstance(error, ModelRegistryError):
                raise
            raise ModelRegistryError(f"模型版本号迁移失败，已回滚：{error}") from error

        shutil.rmtree(staging_root, ignore_errors=True)
        for old_id in identifiers:
            shutil.rmtree(self.version_directory(old_id))
        return migrated

    def _activate_unlocked(
        self,
        version: ModelVersion,
        versions: pd.DataFrame,
    ) -> None:
        self._validate_version_artifact(version)
        source = self.version_directory(version.version_id)
        mirror_mapping = {
            source / "model.joblib": self.project_root
            / "models"
            / "CSG124"
            / "tilt_temperature_v1.joblib",
            source / "metadata.json": self.project_root
            / "reports"
            / "metrics"
            / "temperature_model_decision.json",
            source / "split_plan.json": self.project_root
            / "reports"
            / "metrics"
            / "temperature_split_plan.json",
            source / "metrics.csv": self.project_root
            / "reports"
            / "metrics"
            / "temperature_model_metrics.csv",
            source / "parameter_candidates.csv": self.project_root
            / "reports"
            / "metrics"
            / "temperature_hyperparameter_summary.csv",
            source / "parameter_search.csv": self.project_root
            / "reports"
            / "metrics"
            / "temperature_hyperparameter_search.csv",
            source / "test_predictions.csv": self.project_root
            / "reports"
            / "predictions"
            / "CSG124_temperature_holdout_predictions.csv",
            source / "selected_parameters.json": self.project_root
            / "configs"
            / "selected_model_parameters.json",
        }
        for source_path, target_path in mirror_mapping.items():
            self._copy_atomic(source_path, target_path)
        self._write_json_atomic(
            self.active_path,
            {
                "schema_version": 1,
                "version_id": version.version_id,
                "activated_at_utc": self._now(),
            },
        )

    def _stage_legacy_version(
        self,
        version_dir: Path,
        legacy_model: Path,
        metadata: dict[str, Any],
    ) -> None:
        staging = self._temporary_directory(self.registry_dir, ".migration-")
        mapping = {
            legacy_model: staging / "model.joblib",
            self.project_root
            / "reports"
            / "metrics"
            / "temperature_model_decision.json": staging / "metadata.json",
            self.project_root
            / "reports"
            / "metrics"
            / "temperature_model_metrics.csv": staging / "metrics.csv",
            self.project_root
            / "reports"
            / "metrics"
            / "temperature_hyperparameter_summary.csv": staging
            / "parameter_candidates.csv",
            self.project_root
            / "reports"
            / "metrics"
            / "temperature_hyperparameter_search.csv": staging
            / "parameter_search.csv",
            self.project_root
            / "reports"
            / "predictions"
            / "CSG124_temperature_holdout_predictions.csv": staging
            / "test_predictions.csv",
            self.project_root
            / "reports"
            / "metrics"
            / "temperature_split_plan.json": staging / "split_plan.json",
            self.project_root
            / "configs"
            / "selected_model_parameters.json": staging / "selected_parameters.json",
        }
        try:
            for source, target in mapping.items():
                if not source.is_file():
                    raise ModelRegistryError(f"迁移所需文件缺失：{source.name}")
                shutil.copy2(source, target)
            self._validate_artifact_directory(staging)
            os.replace(staging, version_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _validate_artifact_directory(self, directory: Path) -> None:
        if not directory.is_dir():
            raise ModelRegistryError(f"模型工件目录不存在：{directory}")
        metadata = self._read_json(directory / "metadata.json")
        missing = [name for name in ARTIFACT_NAMES if not (directory / name).is_file()]
        if metadata.get("feature_version") == HYBRID_FEATURE_VERSION:
            missing.extend(
                name
                for name in HYBRID_ARTIFACT_NAMES
                if not (directory / name).is_file()
            )
        if missing:
            raise ModelRegistryError(f"模型工件缺失：{missing}")
        self._validate_metadata(metadata)
        if metadata.get('feature_version') in (compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION):
            import joblib
            from .temperature_pipeline import validate_compact36_artifact

            try:
                validate_compact36_artifact(joblib.load(directory / 'model.joblib')['model'], metadata)
            except (ValueError, KeyError, TypeError, TemperaturePipelineError) as error:
                raise ModelRegistryError(f'36维组合工件无效：{error}') from error
        if metadata.get('feature_version') == probability.FEATURE_VERSION:
            import joblib
            from .temperature_pipeline import validate_probability_artifact

            try:
                validate_probability_artifact(joblib.load(directory/'model.joblib')['model'], metadata)
            except (ValueError, KeyError, TypeError, TemperaturePipelineError) as error:
                raise ModelRegistryError(f'横线概率融合模型工件无效：{error}') from error

    def _validate_version_artifact(self, version: ModelVersion) -> None:
        directory = self.version_directory(version.version_id)
        self._validate_artifact_directory(directory)
        actual = self._sha256(directory / "model.joblib")
        if actual != version.model_sha256:
            raise ModelRegistryError("模型文件 SHA-256 哈希不匹配，已拒绝激活。")
        metadata = self._read_json(directory / "metadata.json")
        if metadata.get("feature_version") == HYBRID_FEATURE_VERSION:
            try:
                import joblib

                payload = joblib.load(directory / "model.joblib")
                model = payload.get("model") if isinstance(payload, dict) else None
            except Exception as error:
                raise ModelRegistryError(
                    f"横线两阶段模型工件无法读取：{error}"
                ) from error
            if not getattr(model, "has_crossline_residual", False):
                raise ModelRegistryError(
                    "横线两阶段模型缺少残差估计器，已拒绝激活。"
                )

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        required = {
            "gateway",
            "dataset_version",
            "source_sha256",
            "model",
            "parameters",
            "status",
            "split",
        }
        missing = sorted(required - set(metadata))
        if missing:
            raise ModelRegistryError(f"训练元数据缺少字段：{missing}")
        try:
            parameters = normalize_model_parameters(metadata["parameters"])
            if metadata["model"] != parameters["model"]:
                raise TemperaturePipelineError("整体模型类型与参数不一致。")
        except TemperaturePipelineError as error:
            raise ModelRegistryError(f"模型参数无效：{error}") from error
        if self.legacy_bootstrap_disabled_path.is_file():
            if metadata.get("dataset_version") != "csg124_cleaned_database_v2":
                raise ModelRegistryError("模型未使用 cleaning-v2 数据合同，禁止保存或发布。")
            policy_sha = str(metadata.get("cleaning_policy_sha256", ""))
            if len(policy_sha) != 64:
                raise ModelRegistryError("模型元数据缺少有效的清洗策略 SHA-256。")
        if metadata.get("feature_version") == HYBRID_FEATURE_VERSION:
            residual = metadata.get("crossline_residual")
            if not isinstance(residual, dict) or not isinstance(
                residual.get("parameters"),
                dict,
            ):
                raise ModelRegistryError("两阶段横线模型缺少残差参数。")
            try:
                CrosslineResidualConfig.from_mapping(residual["parameters"])
            except ValueError as error:
                raise ModelRegistryError(f"横线残差参数无效：{error}") from error

        from .temperature_pipeline import direction_models_from_parameters, compact_recipe_metadata
        recipe = {compact36.MODEL_NAME: compact36, seven.MODEL_NAME: seven, nine.MODEL_NAME: nine, tabicl_single.MODEL_NAME: tabicl_single, tabicl_16.MODEL_NAME: tabicl_16, tabm_single.MODEL_NAME: tabm_single}.get(direction_models_from_parameters(parameters)['crossline'])
        uses_compact36 = recipe is not None
        if uses_compact36 != (metadata.get('feature_version') in (compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION)):
            raise ModelRegistryError('36维组合算法与特征版本不一致。')
        if uses_compact36 and (metadata.get('feature_version') != recipe.FEATURE_VERSION
                or metadata.get('crossline_compact36') != compact_recipe_metadata(recipe, parameters['directions']['crossline'])):
            raise ModelRegistryError('36维组合缺少完整固定方案元数据。')
        uses_probability = direction_models_from_parameters(parameters)['crossline'] == probability.MODEL_NAME
        if uses_probability != (metadata.get('feature_version') == probability.FEATURE_VERSION):
            raise ModelRegistryError('横线概率融合算法与特征版本不一致。')
        if uses_probability and metadata.get('crossline_probability') != probability.recipe_metadata():
            raise ModelRegistryError('横线概率融合元数据不完整或与固定方案不一致。')

    def _version_from_metadata(
        self,
        version_id: str,
        metadata: dict[str, Any],
        note: str,
        model_sha256: str,
    ) -> ModelVersion:
        parameters = normalize_model_parameters(metadata["parameters"])
        inline = parameters["directions"]["inline"]
        crossline = parameters["directions"]["crossline"]
        ratio = metadata["split"]["requested_ratio"]["label"]
        return ModelVersion(
            version_id=version_id,
            created_at_utc=str(metadata.get("created_at_utc", self._now())),
            note=note,
            gateway=str(metadata["gateway"]),
            model=parameters["model"],
            epsilon=inline.get("epsilon"),
            alpha=inline.get("alpha"),
            inline_epsilon=inline.get("epsilon"),
            inline_alpha=inline.get("alpha"),
            crossline_epsilon=crossline.get("epsilon"),
            crossline_alpha=crossline.get("alpha"),
            max_iter=(inline.get("max_iter") if inline.get("max_iter") == crossline.get("max_iter") else None),
            split_ratio=str(ratio),
            dataset_version=str(metadata["dataset_version"]),
            source_sha256=str(metadata["source_sha256"]),
            status=str(metadata["status"]),
            model_sha256=model_sha256,
        )

    @staticmethod
    def _selected_parameters_document(metadata: dict[str, Any]) -> dict[str, Any]:
        document = {
            "schema_version": 3,
            "gateway": metadata["gateway"],
            "dataset_version": metadata["dataset_version"],
            "feature_version": metadata.get("feature_version"),
            "training_source": metadata.get("training_source"),
            "model": metadata["model"],
            "parameters": metadata["parameters"],
            "status": metadata["status"],
            "source_sha256": metadata["source_sha256"],
            "split_ratio": metadata["split"]["requested_ratio"],
        }
        if "direction_feature_columns" in metadata:
            document["direction_feature_columns"] = metadata[
                "direction_feature_columns"
            ]
        if "calibration_policy" in metadata:
            document["calibration_policy"] = metadata["calibration_policy"]
        if "cleaning_policy_sha256" in metadata:
            document["cleaning_policy_sha256"] = metadata[
                "cleaning_policy_sha256"
            ]
        for key in ("crossline_residual", "crossline_probability", "crossline_compact36", "derived_feature_columns", "library_versions"):
            if key in metadata:
                document[key] = metadata[key]
        return document

    def _read_versions(self) -> pd.DataFrame:
        if not self.index_path.is_file() or self.index_path.stat().st_size == 0:
            return pd.DataFrame(columns=VERSION_COLUMNS)
        try:
            frame = pd.read_csv(self.index_path, dtype=str, keep_default_na=False)
        except (OSError, pd.errors.ParserError) as error:
            raise ModelRegistryError(f"模型版本索引无法读取：{error}") from error
        for column, legacy_column in (
            ("inline_epsilon", "epsilon"),
            ("inline_alpha", "alpha"),
            ("crossline_epsilon", "epsilon"),
            ("crossline_alpha", "alpha"),
        ):
            if column not in frame and legacy_column in frame:
                frame[column] = frame[legacy_column]
        missing = [column for column in VERSION_COLUMNS if column not in frame]
        if missing:
            raise ModelRegistryError(f"模型版本索引缺少字段：{missing}")
        return frame.loc[:, VERSION_COLUMNS]

    def _persist_direction_columns_if_needed(self, versions: pd.DataFrame) -> None:
        """Atomically upgrade an old shared-parameter index exactly once."""

        try:
            stored_columns = set(pd.read_csv(self.index_path, nrows=0).columns)
        except (OSError, pd.errors.ParserError) as error:
            raise ModelRegistryError(f"模型版本索引表头无法读取：{error}") from error
        direction_columns = {
            "inline_epsilon",
            "inline_alpha",
            "crossline_epsilon",
            "crossline_alpha",
        }
        if not direction_columns.issubset(stored_columns):
            self._write_versions(versions)

    def _write_versions(self, versions: pd.DataFrame) -> None:
        ordered = versions.loc[:, VERSION_COLUMNS].copy()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv.tmp",
            prefix="versions-",
            dir=self.index_path.parent,
            encoding="utf-8-sig",
            newline="",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            ordered.to_csv(temporary, index=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, self.index_path)

    @staticmethod
    def _find_version(versions: pd.DataFrame, version_id: str) -> ModelVersion:
        match = versions.loc[versions["version_id"].astype(str) == version_id]
        if match.empty:
            raise ModelRegistryError(f"找不到模型版本：{version_id}")
        if len(match) != 1:
            raise ModelRegistryError(f"模型版本索引存在重复记录：{version_id}")
        return ModelVersion.from_row(match.iloc[0].to_dict())

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _copy_atomic(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{path.name}-",
            suffix=".tmp",
            dir=path.parent,
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)

    @staticmethod
    def _write_bytes_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)

    @staticmethod
    def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelRegistryError(f"JSON 文件无法读取：{path.name}：{error}") from error
        if not isinstance(payload, dict):
            raise ModelRegistryError(f"JSON 文件顶层必须是对象：{path.name}")
        return payload

    @staticmethod
    def _temporary_directory(parent: Path, prefix: str) -> Path:
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _new_identifier(prefix: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"

    @staticmethod
    def _next_version_id(versions: pd.DataFrame) -> str:
        """Return the next monotonic human-readable model version."""

        if versions.empty:
            return "v1"
        numbers: list[int] = []
        for value in versions["version_id"].astype(str):
            if len(value) < 2 or value[0] != "v" or not value[1:].isdigit():
                raise ModelRegistryError("模型版本索引包含尚未迁移的旧版本号。")
            number = int(value[1:])
            if number <= 0:
                raise ModelRegistryError("模型版本号必须从 v1 开始。")
            numbers.append(number)
        return f"v{max(numbers) + 1}"

    @staticmethod
    def _is_simple_version_id(identifier: str) -> bool:
        return (
            len(identifier) >= 2
            and identifier[0] == "v"
            and identifier[1:].isdigit()
            and int(identifier[1:]) > 0
        )

    @staticmethod
    def _validate_identifier(identifier: str, prefix: str) -> None:
        if prefix == "model":
            valid = ModelRegistry._is_simple_version_id(identifier)
        else:
            valid = identifier.startswith(f"{prefix}-") and not any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in identifier
            )
        if not valid:
            raise ModelRegistryError("模型版本或训练候选 ID 格式无效。")

    @staticmethod
    def _sanitize_note(note: str) -> str:
        normalized = " ".join(str(note).split())[:200]
        if normalized.startswith(("=", "+", "-", "@")):
            return f"'{normalized}"
        return normalized

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
