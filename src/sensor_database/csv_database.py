"""Recoverable, batch-aware persistence for the sensor CSV database."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from .errors import SensorDatabaseError
from .import_service import MEASUREMENT_COLUMNS, NormalizedUpload
from .cleaning import CleaningResult, SensorCleaningPipeline
from .processing import current_processing_version, retained_processed_rows


SCHEMA_VERSION = 2
GATEWAY = "CSG124"
ACTIVE_STATUS = "active"
DELETED_STATUS = "deleted"
DATABASE_COLUMNS = (
    "schema_version",
    "gateway",
    "timestamp",
    "source_row_count",
    *MEASUREMENT_COLUMNS,
    "source_filename",
    "source_sheet",
    "source_sha256",
    "import_id",
    "imported_at_utc",
)
INITIAL_EMPTY_COLUMNS = (
    "gateway",
    "timestamp",
    "tilt1_inline_model",
    "tilt1_crossline_model",
    "temperature_celsius",
    "tilt2_inline",
    "tilt2_crossline",
)
IMPORT_LOG_COLUMNS = (
    "schema_version",
    "import_id",
    "status",
    "imported_at_utc",
    "deleted_at_utc",
    "gateway",
    "source_filename",
    "source_sheet",
    "source_sha256",
    "source_rows",
    "accepted_rows",
    "rejected_rows",
    "time_series_rows",
    "duplicate_timestamp_rows_combined",
    "inserted_rows",
    "updated_rows",
    "unchanged_rows",
    "database_rows_after_import",
    "cleaning_policy_sha256",
    "cleaning_bucket_rows",
    "cleaning_usable_rows",
    "cleaning_imputed_rows",
    "cleaning_unresolved_rows",
    "cleaning_audit_path",
    "deletion_backup_path",
    "cleaning_options_json",
)


@dataclass(frozen=True)
class ImportResult:
    """Identity and counts for one committed import batch."""

    import_id: str
    inserted_rows: int
    updated_rows: int
    unchanged_rows: int
    database_rows: int
    rejected_rows: int
    duplicate: bool = False
    bucket_rows: int = 0
    usable_rows: int = 0
    imputed_rows: int = 0
    unresolved_rows: int = 0
    field_summary: tuple[dict[str, str | int], ...] = ()
    affected_rows: tuple[dict[str, object], ...] = ()
    cleaning_audit_path: str = ""
    temperature_jump_check_enabled: bool = False
    temperature_suspicious_rows: int = 0
    temperature_repaired_rows: int = 0
    temperature_isolated_rows: int = 0
    temperature_rule_version: str = "temperature_jump_v1"
    temperature_interpolated_rows: int = 0
    temperature_offset_corrected_rows: int = 0
    temperature_review_rows: int = 0
    temperature_hard_unresolved_rows: int = 0

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-compatible representation for the prediction UI."""

        return asdict(self)


@dataclass(frozen=True)
class DeleteResult:
    """Effects and backup location for one deleted import batch."""

    import_id: str
    batch_rows: int
    removed_current_rows: int
    restored_previous_rows: int
    database_rows: int
    backup_path: Path


@dataclass(frozen=True)
class DatabaseStatus:
    """Current row, batch, and time coverage summary."""

    measurement_rows: int
    active_imports: int
    deleted_imports: int
    timestamp_min: str | None
    timestamp_max: str | None
    cleaned_rows: int = 0
    usable_rows: int = 0
    imputed_rows: int = 0
    unresolved_rows: int = 0

    @property
    def import_count(self) -> int:
        """Return the total number of recorded import attempts."""

        return self.active_imports + self.deleted_imports


@dataclass(frozen=True)
class TrainingSnapshot:
    """Complete model rows and the exact database identity used for training."""

    frame: pd.DataFrame
    database_sha256: str
    database_rows: int
    excluded_missing_targets: int


class CsvSensorDatabase:
    """Maintain one chronological CSV table rebuilt from active import batches."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.database_dir = self.project_root / "data" / "database"
        self.measurement_path = self.database_dir / "CSG124_measurements.csv"
        self.import_log_path = self.database_dir / "import_log.csv"
        self.batch_dir = self.database_dir / "imports"
        self.cleaning_audit_dir = self.database_dir / "cleaning_audits"
        self.lock_path = self.database_dir / ".database.lock"
        self.deleted_backup_root = self.project_root / "backups" / "deleted_imports"

    @property
    def cleaned_path(self) -> Path:
        """Keep the new derived cache separate from older editions' materialization."""
        suffix = f"_{current_processing_version()}" if current_processing_version() in ("final_v2", "final_v3") else ""
        return self.database_dir / f"CSG124_cleaned_measurements{suffix}.csv"

    def import_upload(
        self,
        upload: NormalizedUpload,
        source_content: bytes,
    ) -> ImportResult:
        """Commit one upload and rebuild the main CSV in chronological order."""

        if not source_content:
            raise SensorDatabaseError("不能将空文件写入数据库。")
        incoming = self._prepare_batch(upload, source_content)
        import_id = str(incoming["import_id"].iloc[0])
        batch_path = self._batch_path(import_id)

        with self._exclusive_lock():
            current = self._read_measurements()
            import_log = self._read_import_log()
            duplicate = self._find_duplicate_import(import_log, incoming)
            if duplicate is not None:
                cleaned = self._read_cleaned_measurements()
                cleaning = SensorCleaningPipeline.for_project(self.project_root).clean(
                    current
                )
                return ImportResult(
                    import_id=str(duplicate["import_id"]),
                    inserted_rows=0,
                    updated_rows=0,
                    unchanged_rows=len(incoming),
                    database_rows=len(current),
                    rejected_rows=int(upload.audit["rejected_rows"]),
                    duplicate=True,
                    bucket_rows=len(cleaned),
                    usable_rows=int(cleaned.get("model_sample_usable", pd.Series(dtype=bool)).sum()),
                    imputed_rows=int(cleaned.get("model_input_imputed", pd.Series(dtype=bool)).sum()),
                    unresolved_rows=int(cleaned.get("model_input_unresolved", pd.Series(dtype=bool)).sum()),
                    field_summary=tuple(item.to_dict() for item in cleaning.field_summary),
                    affected_rows=tuple(cleaning.affected_rows.to_dict("records")),
                    cleaning_audit_path=str(duplicate.get("cleaning_audit_path", "")),
                )
            if import_id in set(import_log["import_id"].astype(str)):
                raise SensorDatabaseError("生成了重复的导入批次 ID，请重新导入。")
            inserted, updated, unchanged = self._compare_with_current(
                current,
                incoming,
            )
            log_row = self._build_log_row(
                upload,
                incoming,
                inserted,
                updated,
                unchanged,
                database_rows_after_import=0,
            )
            next_log = pd.concat([import_log, log_row], ignore_index=True)

            calibration_path = self.database_dir / "crossline_calibrations.csv"
            audit_path = self.cleaning_audit_dir / f"{import_id}.json"
            protected_paths = [
                self.measurement_path,
                self.cleaned_path,
                self.import_log_path,
                batch_path,
                calibration_path,
                audit_path,
            ]
            original_files = self._capture_files(protected_paths)
            try:
                self._atomic_write(incoming, batch_path)
                rebuilt = self._rebuild_from_log(next_log)
                cleaning = SensorCleaningPipeline.for_project(self.project_root).clean(
                    rebuilt
                )
                next_log.loc[
                    next_log["import_id"] == import_id,
                    "database_rows_after_import",
                ] = len(rebuilt)
                self._apply_cleaning_log_values(next_log, import_id, cleaning, audit_path)
                self._atomic_write(rebuilt, self.measurement_path)
                self._atomic_write(cleaning.frame, self.cleaned_path)
                self._atomic_write(next_log, self.import_log_path)
                self._write_cleaning_audit(audit_path, import_id, cleaning)
                from tilt_predictor.crossline_calibration import (
                    CrosslineCalibrationService,
                )

                CrosslineCalibrationService(self.project_root).reconcile(cleaning.frame)
            except Exception as error:
                self._restore_files(original_files)
                if isinstance(error, SensorDatabaseError):
                    raise
                raise SensorDatabaseError(f"入库失败，已回滚：{error}") from error

        return ImportResult(
            import_id=import_id,
            inserted_rows=inserted,
            updated_rows=updated,
            unchanged_rows=unchanged,
            database_rows=len(rebuilt),
            rejected_rows=int(upload.audit["rejected_rows"]),
            bucket_rows=cleaning.bucket_rows,
            usable_rows=cleaning.usable_rows,
            imputed_rows=cleaning.imputed_rows,
            unresolved_rows=cleaning.unresolved_rows,
            field_summary=tuple(item.to_dict() for item in cleaning.field_summary),
            affected_rows=tuple(cleaning.affected_rows.to_dict("records")),
            cleaning_audit_path=str(audit_path.relative_to(self.project_root)),
        )

    def _find_duplicate_import(
        self,
        import_log: pd.DataFrame,
        incoming: pd.DataFrame,
    ) -> pd.Series | None:
        """Return an active batch with the same source and normalized data."""

        source_sha256 = str(incoming["source_sha256"].iloc[0])
        candidates = import_log.loc[
            (import_log["status"] == ACTIVE_STATUS)
            & (import_log["source_sha256"].astype(str) == source_sha256)
        ]
        for _, candidate in candidates.sort_values(
            "imported_at_utc",
            kind="stable",
        ).iterrows():
            existing = self._read_batch(
                self._batch_path(str(candidate["import_id"]))
            )
            if self._batch_data_equal(existing, incoming):
                return candidate
        return None

    @staticmethod
    def _batch_data_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
        """Compare normalized measurements while ignoring batch metadata."""

        columns = ("gateway", "timestamp", "source_row_count", *MEASUREMENT_COLUMNS)
        if len(left) != len(right):
            return False
        left_values = left.loc[:, list(columns)].sort_values(
            ["gateway", "timestamp"],
            kind="stable",
        ).reset_index(drop=True)
        right_values = right.loc[:, list(columns)].sort_values(
            ["gateway", "timestamp"],
            kind="stable",
        ).reset_index(drop=True)
        # MySQL returns native ``datetime`` objects while upload normalization
        # uses pandas timestamps.  Compare their time values, not their Python
        # container types, so the same workbook cannot create another batch.
        left_values["timestamp"] = pd.to_datetime(
            left_values["timestamp"], errors="coerce"
        )
        right_values["timestamp"] = pd.to_datetime(
            right_values["timestamp"], errors="coerce"
        )
        if not left_values.loc[:, ["gateway", "timestamp"]].equals(
            right_values.loc[:, ["gateway", "timestamp"]]
        ):
            return False
        if not np.array_equal(
            left_values["source_row_count"].to_numpy(dtype=int),
            right_values["source_row_count"].to_numpy(dtype=int),
        ):
            return False
        return bool(
            np.allclose(
                left_values.loc[:, list(MEASUREMENT_COLUMNS)].to_numpy(dtype=float),
                right_values.loc[:, list(MEASUREMENT_COLUMNS)].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )

    def delete_import(self, import_id: str) -> DeleteResult:
        """Delete one active batch, rebuild current data, and retain a backup."""

        self._validate_import_id(import_id)
        batch_path = self._batch_path(import_id)
        with self._exclusive_lock():
            import_log = self._read_import_log()
            matching = import_log.loc[import_log["import_id"].astype(str) == import_id]
            if matching.empty:
                raise SensorDatabaseError("找不到指定的导入批次。")
            if str(matching.iloc[0]["status"]) != ACTIVE_STATUS:
                raise SensorDatabaseError("该导入批次已经删除，不能重复删除。")
            if not batch_path.exists():
                raise SensorDatabaseError("导入批次快照缺失，已停止删除以保护数据。")

            current = self._read_measurements()
            batch = self._read_batch(batch_path)
            deleted_at = datetime.now(timezone.utc)
            backup_dir = self._create_deletion_backup(
                import_id,
                deleted_at,
                batch_path,
            )
            next_log = import_log.copy()
            row_mask = next_log["import_id"].astype(str) == import_id
            next_log.loc[row_mask, "status"] = DELETED_STATUS
            next_log.loc[row_mask, "deleted_at_utc"] = deleted_at.isoformat()
            next_log.loc[row_mask, "deletion_backup_path"] = str(
                backup_dir.relative_to(self.project_root)
            )
            rebuilt = self._rebuild_from_log(next_log, excluded_import_id=import_id)
            cleaning = SensorCleaningPipeline.for_project(self.project_root).clean(rebuilt)
            removed, restored = self._deletion_effects(current, rebuilt)

            calibration_path = self.database_dir / "crossline_calibrations.csv"
            protected_paths = [
                self.measurement_path,
                self.cleaned_path,
                self.import_log_path,
                batch_path,
                calibration_path,
            ]
            original_files = self._capture_files(protected_paths)
            try:
                self._atomic_write(rebuilt, self.measurement_path)
                self._atomic_write(cleaning.frame, self.cleaned_path)
                self._atomic_write(next_log, self.import_log_path)
                archived_batch = backup_dir / "batch.csv"
                os.replace(batch_path, archived_batch)
                from tilt_predictor.crossline_calibration import (
                    CrosslineCalibrationService,
                )

                CrosslineCalibrationService(self.project_root).reconcile(cleaning.frame)
            except Exception as error:
                self._restore_files(original_files)
                if isinstance(error, SensorDatabaseError):
                    raise
                raise SensorDatabaseError(f"删除失败，已回滚：{error}") from error

        return DeleteResult(
            import_id=import_id,
            batch_rows=len(batch),
            removed_current_rows=removed,
            restored_previous_rows=restored,
            database_rows=len(rebuilt),
            backup_path=backup_dir,
        )

    def load_measurements(self) -> pd.DataFrame:
        """Return all current measurements in chronological order."""

        with self._exclusive_lock():
            return self._read_measurements().copy()

    def load_cleaned_measurements(self) -> pd.DataFrame:
        """Return the materialized ten-minute model table."""

        with self._exclusive_lock():
            return self._read_cleaned_measurements().copy()

    def load_import_log(self) -> pd.DataFrame:
        """Return active and deleted batch audit records."""

        with self._exclusive_lock():
            return self._read_import_log().copy()

    def active_imports(self) -> pd.DataFrame:
        """Return only batches currently contributing to the database."""

        import_log = self.load_import_log()
        return import_log.loc[import_log["status"] == ACTIVE_STATUS].copy()

    def status(self) -> DatabaseStatus:
        """Summarize current measurements and import batches."""

        measurements = self.load_measurements()
        cleaned = retained_processed_rows(self.load_cleaned_measurements())
        import_log = self.load_import_log()
        if measurements.empty:
            timestamp_min = None
            timestamp_max = None
        else:
            timestamps = pd.to_datetime(measurements["timestamp"], errors="raise")
            timestamp_min = timestamps.min().isoformat()
            timestamp_max = timestamps.max().isoformat()
        return DatabaseStatus(
            measurement_rows=len(measurements),
            active_imports=int((import_log["status"] == ACTIVE_STATUS).sum()),
            deleted_imports=int((import_log["status"] == DELETED_STATUS).sum()),
            timestamp_min=timestamp_min,
            timestamp_max=timestamp_max,
            cleaned_rows=len(retained_processed_rows(cleaned)),
            usable_rows=int(cleaned.get("model_sample_usable", pd.Series(dtype=bool)).sum()),
            imputed_rows=int(cleaned.get("model_input_imputed", pd.Series(dtype=bool)).sum()),
            unresolved_rows=int(cleaned.get("model_input_unresolved", pd.Series(dtype=bool)).sum()),
        )

    def training_snapshot(self) -> TrainingSnapshot:
        """Return complete feature/target rows sorted by time for model training."""

        with self._exclusive_lock():
            measurements = self._read_cleaned_measurements().copy()
            if measurements.empty:
                raise SensorDatabaseError("CSV 数据库尚无可训练数据。")
            database_sha256 = (
                self._sha256_file(self.cleaned_path)
                if self.cleaned_path.exists()
                else hashlib.sha256(
                    measurements.to_csv(index=False).encode("utf-8")
                ).hexdigest()
            )

        if "history_barrier" in measurements:
            measurements["history_segment"] = measurements["history_barrier"].fillna(False).astype(bool).cumsum()

        required = ["timestamp", *MEASUREMENT_COLUMNS]
        usable = measurements.get(
            "model_sample_usable", pd.Series(False, index=measurements.index)
        ).astype(bool)
        complete_mask = measurements.loc[:, required].notna().all(axis=1) & usable
        context_columns = [
            column
            for column in (
                "segment_id",
                "is_inserted_gap",
                "model_input_available",
                "model_input_imputed",
                "model_input_unresolved",
                "model_sample_usable",
                "quality_status",
                "processing_version",
                "history_barrier",
                "history_segment",
                "processed_row_excluded",
                "cleaning_policy_sha256",
                "import_id",
                "gateway",
            )
            if column in measurements
        ]
        complete = measurements.loc[complete_mask, [*required, *context_columns]].copy()
        numeric_values = complete.loc[:, list(MEASUREMENT_COLUMNS)].to_numpy(
            dtype=float
        )
        if len(complete) and not np.isfinite(numeric_values).all():
            raise SensorDatabaseError("数据库训练数据包含非有限数值。")
        if complete.empty:
            raise SensorDatabaseError("数据库没有同时包含特征和真值的记录。")
        complete["timestamp"] = pd.to_datetime(
            complete["timestamp"],
            errors="raise",
        )
        complete = complete.sort_values("timestamp", kind="stable").reset_index(
            drop=True
        )
        return TrainingSnapshot(
            frame=complete,
            database_sha256=database_sha256,
            database_rows=len(measurements),
            excluded_missing_targets=int((~complete_mask).sum()),
        )

    def _prepare_batch(
        self,
        upload: NormalizedUpload,
        source_content: bytes,
    ) -> pd.DataFrame:
        required = ["timestamp", "source_row_count", *MEASUREMENT_COLUMNS]
        missing = [column for column in required if column not in upload.frame]
        if missing:
            raise SensorDatabaseError(f"入库数据缺少字段：{missing}")
        if upload.frame.empty:
            raise SensorDatabaseError("入库数据为空。")

        incoming = upload.frame.loc[:, required].copy()
        timestamps = pd.to_datetime(incoming["timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise SensorDatabaseError("入库数据包含无效时间。")
        if timestamps.duplicated().any():
            raise SensorDatabaseError("规范化后的入库数据仍有重复时间戳。")
        feature_columns = (
            "tilt1_inline_model",
            "tilt1_crossline_model",
            "temperature_celsius",
        )
        feature_values = incoming.loc[:, list(feature_columns)].to_numpy(
            dtype=float
        )
        if not np.isfinite(feature_values).all():
            raise SensorDatabaseError("入库特征包含缺失或非有限数值。")
        for target in ("tilt2_inline", "tilt2_crossline"):
            finite_or_missing = np.isfinite(incoming[target]) | incoming[target].isna()
            if not finite_or_missing.all():
                raise SensorDatabaseError(f"入库目标字段 {target} 包含无效数值。")

        import_id = str(uuid4())
        imported_at = datetime.now(timezone.utc).isoformat()
        incoming["timestamp"] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
        incoming.insert(0, "gateway", GATEWAY)
        incoming.insert(0, "schema_version", SCHEMA_VERSION)
        incoming["source_filename"] = self._safe_csv_text(
            upload.audit["source_filename"]
        )
        incoming["source_sheet"] = self._safe_csv_text(
            upload.audit.get("source_sheet", "")
        )
        incoming["source_sha256"] = hashlib.sha256(source_content).hexdigest()
        incoming["import_id"] = import_id
        incoming["imported_at_utc"] = imported_at
        return incoming.loc[:, list(DATABASE_COLUMNS)].sort_values(
            ["timestamp", "gateway"],
            kind="stable",
        ).reset_index(drop=True)

    def _build_log_row(
        self,
        upload: NormalizedUpload,
        incoming: pd.DataFrame,
        inserted: int,
        updated: int,
        unchanged: int,
        database_rows_after_import: int,
    ) -> pd.DataFrame:
        row = {
            "schema_version": SCHEMA_VERSION,
            "import_id": str(incoming["import_id"].iloc[0]),
            "status": ACTIVE_STATUS,
            "imported_at_utc": str(incoming["imported_at_utc"].iloc[0]),
            "deleted_at_utc": "",
            "gateway": GATEWAY,
            "source_filename": str(incoming["source_filename"].iloc[0]),
            "source_sheet": str(incoming["source_sheet"].iloc[0]),
            "source_sha256": str(incoming["source_sha256"].iloc[0]),
            "source_rows": int(upload.audit["source_rows"]),
            "accepted_rows": int(upload.audit["accepted_rows"]),
            "rejected_rows": int(upload.audit["rejected_rows"]),
            "time_series_rows": int(
                upload.audit.get(
                    "time_series_rows",
                    upload.audit.get("prediction_rows", len(incoming)),
                )
            ),
            "duplicate_timestamp_rows_combined": int(
                upload.audit.get(
                    "duplicate_timestamp_rows_combined",
                    int(upload.audit["accepted_rows"]) - len(incoming),
                )
            ),
            "inserted_rows": inserted,
            "updated_rows": updated,
            "unchanged_rows": unchanged,
            "database_rows_after_import": database_rows_after_import,
            "deletion_backup_path": "",
            "cleaning_policy_sha256": "",
            "cleaning_bucket_rows": 0,
            "cleaning_usable_rows": 0,
            "cleaning_imputed_rows": 0,
            "cleaning_unresolved_rows": 0,
            "cleaning_audit_path": "",
            "cleaning_options_json": "",
        }
        return pd.DataFrame([row], columns=IMPORT_LOG_COLUMNS)

    def _rebuild_from_log(
        self,
        import_log: pd.DataFrame,
        excluded_import_id: str | None = None,
    ) -> pd.DataFrame:
        by_key: dict[tuple[str, str], pd.Series] = {}
        active = import_log.loc[import_log["status"] == ACTIVE_STATUS]
        active = active.sort_values("imported_at_utc", kind="stable")
        for _, log_row in active.iterrows():
            import_id = str(log_row["import_id"])
            if import_id == excluded_import_id:
                continue
            batch = self._read_batch(self._batch_path(import_id))
            for _, row in batch.iterrows():
                key = (str(row["gateway"]), str(row["timestamp"]))
                by_key[key] = row
        if not by_key:
            return pd.DataFrame(columns=DATABASE_COLUMNS)
        rebuilt = pd.DataFrame(by_key.values(), columns=DATABASE_COLUMNS)
        return rebuilt.sort_values(
            ["timestamp", "gateway"],
            kind="stable",
        ).reset_index(drop=True)

    def _read_measurements(self) -> pd.DataFrame:
        if not self.measurement_path.exists():
            return pd.DataFrame(columns=DATABASE_COLUMNS)
        try:
            frame = pd.read_csv(self.measurement_path, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError) as error:
            raise SensorDatabaseError(f"无法读取测量数据库：{error}") from error
        if frame.empty and tuple(frame.columns) == INITIAL_EMPTY_COLUMNS:
            return pd.DataFrame(columns=DATABASE_COLUMNS)
        self._validate_columns(frame, DATABASE_COLUMNS, "测量数据库")
        if frame.duplicated(subset=["gateway", "timestamp"]).any():
            raise SensorDatabaseError("测量数据库中存在重复主键。")
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise SensorDatabaseError("测量数据库中存在无效时间。")
        return frame.loc[:, list(DATABASE_COLUMNS)].sort_values(
            ["timestamp", "gateway"],
            kind="stable",
        ).reset_index(drop=True)

    def _read_cleaned_measurements(self) -> pd.DataFrame:
        if current_processing_version() in ("final_v2", "final_v3"):
            return self._read_versioned_cleaned_measurements()
        if not self.cleaned_path.exists():
            raw = self._read_measurements()
            return SensorCleaningPipeline.for_project(self.project_root).clean(raw).frame
        try:
            frame = pd.read_csv(self.cleaned_path, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError) as error:
            raise SensorDatabaseError(f"无法读取清洗数据库：{error}") from error
        if "timestamp" not in frame or "model_sample_usable" not in frame:
            raise SensorDatabaseError("清洗数据库字段不完整。")
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise SensorDatabaseError("清洗数据库包含无效时间。")
        return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

    def _read_versioned_cleaned_measurements(self) -> pd.DataFrame:
        """Validate a derived cache against source bytes, policy and cached bytes.

        The caller holds the database lock; rebuilding changes only the personal
        cache, never source measurements, import signatures or older caches.
        """
        pipeline = SensorCleaningPipeline.for_project(self.project_root)
        fingerprint = {
            "processing_sha256": pipeline.processing_sha256,
            "source_sha256": self._sha256_file(self.measurement_path) if self.measurement_path.exists() else None,
            "imports_sha256": self._sha256_file(self.import_log_path) if self.import_log_path.exists() else None,
        }
        metadata_path = self.cleaned_path.with_suffix(".json")
        if self.cleaned_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise SensorDatabaseError(f"无法读取清洗缓存签名：{error}") from error
            if metadata == {**fingerprint, "cache_sha256": self._sha256_file(self.cleaned_path)}:
                frame = pd.read_csv(self.cleaned_path, encoding="utf-8-sig")
                if frame.empty or frame["processing_version"].eq(current_processing_version()).all():
                    return frame
        raw = self._read_measurements()
        from .cleaning_options import parse_cleaning_options

        options = {}
        for _, row in self._read_import_log().iterrows():
            option = parse_cleaning_options(row["cleaning_options_json"])
            if option is not None and option.base_policy_sha256 != pipeline.policy.sha256:
                raise SensorDatabaseError("批次基础清洗策略已变更，不能静默重算。")
            options[str(row["import_id"])] = option
        if not raw.empty:
            missing = set(raw["import_id"].astype(str)) - options.keys()
            if missing:
                raise SensorDatabaseError("清洗数据缺少来源批次设置。")
            raw["temperature_jump_check_enabled"] = raw["import_id"].map(
                lambda key: options[str(key)].enabled if options[str(key)] else False)
            raw["temperature_rule_version"] = raw["import_id"].map(
                lambda key: options[str(key)].rule_version if options[str(key)] else "temperature_jump_v1")
        frame = pipeline.clean(raw).frame
        self._atomic_write(frame, self.cleaned_path)
        metadata = {**fingerprint, "cache_sha256": self._sha256_file(self.cleaned_path)}
        self._atomic_write_bytes(json.dumps(metadata, sort_keys=True).encode("utf-8"), metadata_path)
        return frame

    def _read_import_log(self) -> pd.DataFrame:
        if not self.import_log_path.exists():
            return pd.DataFrame(columns=IMPORT_LOG_COLUMNS)
        try:
            frame = pd.read_csv(
                self.import_log_path,
                encoding="utf-8-sig",
                keep_default_na=False,
            )
        except (OSError, pd.errors.ParserError) as error:
            raise SensorDatabaseError(f"无法读取导入日志：{error}") from error
        legacy_defaults: dict[str, object] = {
            "cleaning_options_json": "",
            "cleaning_policy_sha256": "",
            "cleaning_bucket_rows": 0,
            "cleaning_usable_rows": 0,
            "cleaning_imputed_rows": 0,
            "cleaning_unresolved_rows": 0,
            "cleaning_audit_path": "",
        }
        for column, default in legacy_defaults.items():
            if column not in frame:
                frame[column] = default
        frame = frame.reindex(columns=IMPORT_LOG_COLUMNS)
        self._validate_columns(frame, IMPORT_LOG_COLUMNS, "导入日志")
        if frame["import_id"].duplicated().any():
            raise SensorDatabaseError("导入日志中存在重复批次 ID。")
        invalid_statuses = set(frame["status"]) - {ACTIVE_STATUS, DELETED_STATUS}
        if invalid_statuses:
            raise SensorDatabaseError(f"导入日志包含无效状态：{invalid_statuses}")
        return frame.loc[:, list(IMPORT_LOG_COLUMNS)]

    def _read_batch(self, batch_path: Path) -> pd.DataFrame:
        if not batch_path.exists():
            raise SensorDatabaseError(f"缺少导入批次快照：{batch_path.name}")
        try:
            frame = pd.read_csv(batch_path, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError) as error:
            raise SensorDatabaseError(f"无法读取导入批次快照：{error}") from error
        self._validate_columns(frame, DATABASE_COLUMNS, "导入批次快照")
        if frame.duplicated(subset=["gateway", "timestamp"]).any():
            raise SensorDatabaseError("导入批次快照中存在重复主键。")
        return frame.loc[:, list(DATABASE_COLUMNS)]

    def _create_deletion_backup(
        self,
        import_id: str,
        deleted_at: datetime,
        batch_path: Path,
    ) -> Path:
        timestamp = deleted_at.strftime("%Y%m%dT%H%M%S%fZ")
        backup_dir = self.deleted_backup_root / f"{timestamp}_{import_id}"
        try:
            backup_dir.mkdir(parents=True, exist_ok=False)
            if self.measurement_path.exists():
                shutil.copy2(
                    self.measurement_path,
                    backup_dir / "measurements_before.csv",
                )
            if self.import_log_path.exists():
                shutil.copy2(
                    self.import_log_path,
                    backup_dir / "import_log_before.csv",
                )
            if self.cleaned_path.exists():
                shutil.copy2(self.cleaned_path, backup_dir / "cleaned_measurements_before.csv")
            shutil.copy2(batch_path, backup_dir / "batch.csv")
        except OSError as error:
            raise SensorDatabaseError(f"无法创建删除前备份：{error}") from error
        return backup_dir

    @staticmethod
    def _compare_with_current(
        current: pd.DataFrame,
        incoming: pd.DataFrame,
    ) -> tuple[int, int, int]:
        current_by_key = {
            (str(row["gateway"]), str(row["timestamp"])): row
            for _, row in current.iterrows()
        }
        inserted = 0
        updated = 0
        unchanged = 0
        for _, row in incoming.iterrows():
            key = (str(row["gateway"]), str(row["timestamp"]))
            previous = current_by_key.get(key)
            if previous is None:
                inserted += 1
            elif CsvSensorDatabase._measurements_equal(previous, row):
                unchanged += 1
            else:
                updated += 1
        return inserted, updated, unchanged

    @staticmethod
    def _measurements_equal(previous: pd.Series, incoming: pd.Series) -> bool:
        comparison_columns = ("source_row_count", *MEASUREMENT_COLUMNS)
        for column in comparison_columns:
            left = previous[column]
            right = incoming[column]
            if pd.isna(left) and pd.isna(right):
                continue
            if left != right:
                return False
        return True

    @staticmethod
    def _deletion_effects(
        current: pd.DataFrame,
        rebuilt: pd.DataFrame,
    ) -> tuple[int, int]:
        current_by_key = {
            (str(row["gateway"]), str(row["timestamp"])): row
            for _, row in current.iterrows()
        }
        rebuilt_by_key = {
            (str(row["gateway"]), str(row["timestamp"])): row
            for _, row in rebuilt.iterrows()
        }
        removed = len(set(current_by_key) - set(rebuilt_by_key))
        restored = sum(
            1
            for key in set(current_by_key) & set(rebuilt_by_key)
            if str(current_by_key[key]["import_id"])
            != str(rebuilt_by_key[key]["import_id"])
        )
        return removed, restored

    @staticmethod
    def _validate_columns(
        frame: pd.DataFrame,
        expected: tuple[str, ...],
        label: str,
    ) -> None:
        if tuple(frame.columns) != expected:
            raise SensorDatabaseError(
                f"{label}字段与 schema_version={SCHEMA_VERSION} 不匹配。"
            )

    @staticmethod
    def _safe_csv_text(value: object) -> str:
        text = str(value)
        if text.startswith(("=", "+", "-", "@")):
            return f"'{text}"
        return text

    def _apply_cleaning_log_values(
        self,
        import_log: pd.DataFrame,
        import_id: str,
        cleaning: CleaningResult,
        audit_path: Path,
    ) -> None:
        mask = import_log["import_id"].astype(str) == import_id
        values = {
            "cleaning_policy_sha256": cleaning.policy_sha256,
            "cleaning_bucket_rows": cleaning.bucket_rows,
            "cleaning_usable_rows": cleaning.usable_rows,
            "cleaning_imputed_rows": cleaning.imputed_rows,
            "cleaning_unresolved_rows": cleaning.unresolved_rows,
            "cleaning_audit_path": str(audit_path.relative_to(self.project_root)),
        }
        for column, value in values.items():
            import_log.loc[mask, column] = value

    def _write_cleaning_audit(
        self,
        destination: Path,
        import_id: str,
        cleaning: CleaningResult,
    ) -> None:
        payload = {
            "schema_version": 1,
            "import_id": import_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cleaning_policy_sha256": cleaning.policy_sha256,
            "bucket_rows": cleaning.bucket_rows,
            "usable_rows": cleaning.usable_rows,
            "imputed_rows": cleaning.imputed_rows,
            "unresolved_rows": cleaning.unresolved_rows,
            "field_summary": [item.to_dict() for item in cleaning.field_summary],
            "affected_rows": self._json_records(cleaning.affected_rows),
        }
        self._atomic_write_bytes(
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            destination,
        )

    @staticmethod
    def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
        if frame.empty:
            return []
        safe = frame.copy()
        if "timestamp" in safe:
            safe["timestamp"] = pd.to_datetime(safe["timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        safe = safe.replace({np.nan: None})
        return safe.to_dict("records")

    @staticmethod
    def _validate_import_id(import_id: str) -> None:
        try:
            parsed = UUID(import_id)
        except (ValueError, AttributeError) as error:
            raise SensorDatabaseError("导入批次 ID 格式无效。") from error
        if str(parsed) != import_id:
            raise SensorDatabaseError("导入批次 ID 格式无效。")

    def _batch_path(self, import_id: str) -> Path:
        return self.batch_dir / f"{import_id}.csv"

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.database_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _capture_files(paths: list[Path]) -> dict[Path, bytes | None]:
        return {path: path.read_bytes() if path.exists() else None for path in paths}

    @staticmethod
    def _restore_files(original_files: dict[Path, bytes | None]) -> None:
        for path, content in original_files.items():
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                CsvSensorDatabase._atomic_write_bytes(content, path)

    @staticmethod
    def _atomic_write(frame: pd.DataFrame, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as output:
                frame.to_csv(output, index=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

    @staticmethod
    def _atomic_write_bytes(content: bytes, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise
