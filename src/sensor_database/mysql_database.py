"""MySQL-backed sensor persistence with the CSV database contract preserved."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from sensor_app.mysql_config import MySQLConfig

from .csv_database import (
    DATABASE_COLUMNS,
    DELETED_STATUS,
    GATEWAY,
    IMPORT_LOG_COLUMNS,
    ACTIVE_STATUS,
    DatabaseStatus,
    DeleteResult,
    ImportResult,
    CsvSensorDatabase,
    TrainingSnapshot,
)
from .cleaning import CleaningResult, SensorCleaningPipeline
from .cleaning_cache import cleaned_frame_cache
from .cleaning_options import CleaningOptions, parse_cleaning_options
from .errors import SensorDatabaseError
from .import_service import MEASUREMENT_COLUMNS, NormalizedUpload
from .temperature_cusum import AUDIT_COLUMNS as CUSUM_AUDIT_COLUMNS
from .temperature_daily_trend import AUDIT_COLUMNS as DAILY_AUDIT_COLUMNS


MYSQL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MySQLSensorDatabase:
    """Persist measurements and import history in a dedicated MySQL database."""

    config: MySQLConfig
    project_root: Path = field(default_factory=Path.cwd)
    engine: Engine | None = None

    def __post_init__(self) -> None:
        if self.engine is None:
            object.__setattr__(self, "engine", self.config.engine())

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "MySQLSensorDatabase":
        return cls(
            MySQLConfig.from_env(),
            project_root=(project_root or Path.cwd()).expanduser().resolve(),
        )

    def ensure_schema(self) -> None:
        """Create only the CSG124 tables; never touch another database."""

        statements = (
            """
            CREATE TABLE IF NOT EXISTS csg124_schema_meta (
                schema_version INT NOT NULL PRIMARY KEY,
                applied_at_utc DATETIME(6) NOT NULL
            ) ENGINE=InnoDB
            """,
            """
            CREATE TABLE IF NOT EXISTS sensor_import_batches (
                import_id CHAR(36) NOT NULL PRIMARY KEY,
                schema_version INT NOT NULL,
                status VARCHAR(16) NOT NULL,
                imported_at_utc DATETIME(6) NOT NULL,
                deleted_at_utc DATETIME(6) NULL,
                gateway VARCHAR(128) NOT NULL,
                source_filename TEXT NOT NULL,
                source_sheet VARCHAR(255) NULL,
                source_sha256 CHAR(64) NOT NULL,
                source_rows INT NOT NULL,
                accepted_rows INT NOT NULL,
                rejected_rows INT NOT NULL,
                time_series_rows INT NOT NULL,
                duplicate_timestamp_rows_combined INT NOT NULL,
                inserted_rows INT NOT NULL,
                updated_rows INT NOT NULL,
                unchanged_rows INT NOT NULL,
                database_rows_after_import INT NOT NULL,
                cleaning_options_json TEXT NULL,
                deletion_backup_path TEXT NULL,
                INDEX idx_import_status (status),
                INDEX idx_imported_at (imported_at_utc)
            ) ENGINE=InnoDB
            """,
            """
            CREATE TABLE IF NOT EXISTS sensor_import_rows (
                import_id CHAR(36) NOT NULL,
                schema_version INT NOT NULL,
                gateway VARCHAR(128) NOT NULL,
                timestamp DATETIME(6) NOT NULL,
                source_row_count INT NOT NULL,
                tilt1_inline_model DOUBLE NOT NULL,
                tilt1_crossline_model DOUBLE NOT NULL,
                temperature_celsius DOUBLE NOT NULL,
                tilt2_inline DOUBLE NULL,
                tilt2_crossline DOUBLE NULL,
                source_filename TEXT NOT NULL,
                source_sheet VARCHAR(255) NULL,
                source_sha256 CHAR(64) NOT NULL,
                imported_at_utc DATETIME(6) NOT NULL,
                PRIMARY KEY (import_id, gateway, timestamp),
                CONSTRAINT fk_import_rows_batch FOREIGN KEY (import_id)
                    REFERENCES sensor_import_batches(import_id)
                    ON DELETE RESTRICT,
                INDEX idx_import_rows_time (gateway, timestamp)
            ) ENGINE=InnoDB
            """,
            """
            CREATE TABLE IF NOT EXISTS sensor_measurements (
                schema_version INT NOT NULL,
                gateway VARCHAR(128) NOT NULL,
                timestamp DATETIME(6) NOT NULL,
                source_row_count INT NOT NULL,
                tilt1_inline_model DOUBLE NOT NULL,
                tilt1_crossline_model DOUBLE NOT NULL,
                temperature_celsius DOUBLE NOT NULL,
                tilt2_inline DOUBLE NULL,
                tilt2_crossline DOUBLE NULL,
                source_filename TEXT NOT NULL,
                source_sheet VARCHAR(255) NULL,
                source_sha256 CHAR(64) NOT NULL,
                import_id CHAR(36) NOT NULL,
                imported_at_utc DATETIME(6) NOT NULL,
                PRIMARY KEY (gateway, timestamp),
                CONSTRAINT fk_measurements_batch FOREIGN KEY (import_id)
                    REFERENCES sensor_import_batches(import_id)
                    ON DELETE RESTRICT
            ) ENGINE=InnoDB
            """,
            """
            CREATE TABLE IF NOT EXISTS crossline_calibrations (
                calibration_id VARCHAR(128) NOT NULL PRIMARY KEY,
                gateway VARCHAR(128) NOT NULL,
                cycle_started_at DATETIME(6) NOT NULL,
                payload_json JSON NOT NULL,
                updated_at_utc DATETIME(6) NOT NULL,
                INDEX idx_calibration_gateway_cycle (gateway, cycle_started_at)
            ) ENGINE=InnoDB
            """,
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id VARCHAR(128) NOT NULL PRIMARY KEY,
                created_at_utc DATETIME(6) NOT NULL,
                note TEXT NOT NULL,
                gateway VARCHAR(128) NOT NULL,
                model VARCHAR(128) NOT NULL,
                epsilon DOUBLE NULL,
                alpha DOUBLE NULL,
                inline_epsilon DOUBLE NULL,
                inline_alpha DOUBLE NULL,
                crossline_epsilon DOUBLE NULL,
                crossline_alpha DOUBLE NULL,
                max_iter INT NULL,
                split_ratio VARCHAR(64) NOT NULL,
                dataset_version VARCHAR(128) NOT NULL,
                source_sha256 CHAR(64) NOT NULL,
                status VARCHAR(64) NOT NULL,
                model_sha256 CHAR(64) NOT NULL
            ) ENGINE=InnoDB
            """,
            """
            CREATE TABLE IF NOT EXISTS model_active_version (
                singleton_id TINYINT NOT NULL PRIMARY KEY,
                version_id VARCHAR(128) NOT NULL,
                activated_at_utc DATETIME(6) NOT NULL,
                CONSTRAINT fk_active_model_version FOREIGN KEY (version_id)
                    REFERENCES model_versions(version_id)
            ) ENGINE=InnoDB
            """,
            """
            CREATE TABLE IF NOT EXISTS model_drafts (
                run_id VARCHAR(128) NOT NULL PRIMARY KEY,
                created_at_utc DATETIME(6) NOT NULL,
                metadata_json JSON NOT NULL,
                status VARCHAR(64) NOT NULL
            ) ENGINE=InnoDB
            """,
            """
            CREATE TABLE IF NOT EXISTS model_artifacts (
                owner_type ENUM('version','draft') NOT NULL,
                owner_id VARCHAR(128) NOT NULL,
                artifact_name VARCHAR(255) NOT NULL,
                mime_type VARCHAR(128) NOT NULL,
                sha256 CHAR(64) NOT NULL,
                size_bytes BIGINT UNSIGNED NOT NULL,
                content LONGBLOB NOT NULL,
                created_at_utc DATETIME(6) NOT NULL,
                PRIMARY KEY (owner_type, owner_id, artifact_name),
                INDEX idx_artifact_hash (sha256)
            ) ENGINE=InnoDB
            """,
        )
        try:
            with self.engine.begin() as connection:
                for statement in statements:
                    connection.execute(text(statement))
                self.ensure_linear_model_schema(connection)
                connection.execute(
                    text(
                        "INSERT IGNORE INTO csg124_schema_meta "
                        "(schema_version, applied_at_utc) VALUES (:version, :now)"
                    ),
                    {"version": MYSQL_SCHEMA_VERSION, "now": datetime.now(timezone.utc)},
                )
        except SQLAlchemyError as error:
            raise SensorDatabaseError(f"MySQL schema 初始化失败：{error}") from error

    def ensure_linear_model_schema(
        self, connection: Connection | None = None, *, apply: bool = True
    ) -> tuple[str, ...]:
        """Idempotently allow inapplicable legacy model-index parameters to be NULL.

        Only column nullability changes; metadata artifacts retain complete parameters.
        MySQL DDL commits implicitly, so inspect first and never rewrite model rows.
        """

        if connection is None:
            with self.engine.begin() as opened:
                return self.ensure_linear_model_schema(opened, apply=apply)
        if connection.dialect.name != "mysql":
            return ()
        types = {
            "epsilon": "DOUBLE", "alpha": "DOUBLE", "inline_epsilon": "DOUBLE",
            "inline_alpha": "DOUBLE", "crossline_epsilon": "DOUBLE",
            "crossline_alpha": "DOUBLE", "max_iter": "INT",
        }
        rows = connection.execute(text(
            "SELECT COLUMN_NAME, IS_NULLABLE FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='model_versions'"
        )).mappings()
        changes = [
            f"MODIFY COLUMN `{row['COLUMN_NAME']}` {types[row['COLUMN_NAME']]} NULL"
            for row in rows
            if row["COLUMN_NAME"] in types and row["IS_NULLABLE"] == "NO"
        ]
        if changes and apply:
            connection.execute(text("ALTER TABLE model_versions " + ", ".join(changes)))
        return tuple(changes)

    def import_upload(
        self, upload: NormalizedUpload, source_content: bytes, *,
        temperature_jump_check_enabled: bool = False,
        temperature_rule_version: str = "temperature_jump_v1",
    ) -> ImportResult:
        """Import a normalized batch transactionally and rebuild current rows."""

        if not source_content:
            raise SensorDatabaseError("不能将空文件写入数据库。")
        options = CleaningOptions(
            temperature_jump_check_enabled,
            SensorCleaningPipeline.for_project(self.project_root).policy.sha256,
            temperature_rule_version,
        )
        file_backend = CsvSensorDatabase(Path.cwd())
        incoming = file_backend._prepare_batch(upload, source_content)  # pure normalization
        import_id = str(incoming["import_id"].iloc[0])
        try:
            with self.engine.begin() as connection:
                supports_options = self._supports_cleaning_options(connection)
                if (options.enabled or options.rule_version != "temperature_jump_v1") and not supports_options:
                    raise SensorDatabaseError(
                        "温度突变检查需要先升级数据库：运行 "
                        "scripts/migrate_temperature_options.py --apply。"
                    )
                current = self._read_measurements(connection)
                source_sha256 = str(incoming["source_sha256"].iloc[0])
                candidates = connection.execute(
                    text(
                        "SELECT import_id FROM sensor_import_batches "
                        "WHERE status=:status AND source_sha256=:source_sha256 "
                        "ORDER BY imported_at_utc DESC, import_id DESC"
                        + self._for_update_suffix(connection)
                    ),
                    {
                        "status": ACTIVE_STATUS,
                        "source_sha256": source_sha256,
                    },
                ).scalars().all()
                for candidate_id in candidates:
                    rows = connection.execute(
                        text(
                            "SELECT * FROM sensor_import_rows "
                            "WHERE import_id=:import_id ORDER BY gateway, timestamp"
                        ),
                        {"import_id": str(candidate_id)},
                    ).mappings().all()
                    existing_batch = pd.DataFrame(rows)
                    if CsvSensorDatabase._batch_data_equal(
                        existing_batch,
                        incoming,
                    ):
                        previous_options = self._read_batch_options(connection, str(candidate_id))
                        previous_enabled = previous_options.enabled if previous_options else False
                        previous_version = previous_options.rule_version if previous_options else "temperature_jump_v1"
                        if previous_enabled != options.enabled or previous_version != options.rule_version:
                            # 最新的同内容批次设置不同：新建批次，不能回头复用更早同设置批次。
                            break
                        cleaning = self._clean_measurements(connection, current)
                        return ImportResult(
                            import_id=str(candidate_id),
                            inserted_rows=0,
                            updated_rows=0,
                            unchanged_rows=len(incoming),
                            database_rows=len(current),
                            rejected_rows=int(upload.audit["rejected_rows"]),
                            duplicate=True,
                            bucket_rows=cleaning.bucket_rows,
                            usable_rows=cleaning.usable_rows,
                            imputed_rows=cleaning.imputed_rows,
                            unresolved_rows=cleaning.unresolved_rows,
                            field_summary=tuple(
                                item.to_dict() for item in cleaning.field_summary
                            ),
                            affected_rows=tuple(
                                cleaning.affected_rows.to_dict("records")
                            ),
                            temperature_jump_check_enabled=options.enabled,
                            temperature_rule_version=options.rule_version,
                            **self._temperature_counts(cleaning.frame, incoming),
                        )
                inserted, updated, unchanged = CsvSensorDatabase._compare_with_current(
                    current, incoming
                )
                existing = connection.execute(
                    text("SELECT 1 FROM sensor_import_batches WHERE import_id=:id"),
                    {"id": import_id},
                ).first()
                if existing:
                    raise SensorDatabaseError("生成了重复的导入批次 ID，请重新导入。")
                audit = upload.audit
                now = pd.Timestamp(str(incoming["imported_at_utc"].iloc[0])).to_pydatetime()
                connection.execute(
                    text(
                        """INSERT INTO sensor_import_batches
                        (import_id,schema_version,status,imported_at_utc,gateway,source_filename,
                         source_sheet,source_sha256,source_rows,accepted_rows,rejected_rows,
                         time_series_rows,duplicate_timestamp_rows_combined,inserted_rows,
                         updated_rows,unchanged_rows,database_rows_after_import)
                        VALUES (:import_id,:schema_version,:status,:imported_at_utc,:gateway,
                         :source_filename,:source_sheet,:source_sha256,:source_rows,:accepted_rows,
                         :rejected_rows,:time_series_rows,:duplicates,:inserted_rows,:updated_rows,
                         :unchanged_rows,0)"""
                    ),
                    self._batch_params(incoming, audit, inserted, updated, unchanged, now),
                )
                self._insert_rows(connection, incoming)
                if supports_options:
                    connection.execute(
                        text("UPDATE sensor_import_batches SET cleaning_options_json=:options WHERE import_id=:id"),
                        {"options": options.to_json(), "id": import_id},
                    )
                rebuilt = self._rebuild(connection)
                cleaning = self._clean_measurements(connection, rebuilt)
                temperature_counts = self._temperature_counts(cleaning.frame, incoming)
                if supports_options:
                    audit_counts = {
                        **temperature_counts,
                        "cleaning_bucket_rows": cleaning.bucket_rows,
                        "cleaning_usable_rows": cleaning.usable_rows,
                        "cleaning_imputed_rows": cleaning.imputed_rows,
                        "cleaning_unresolved_rows": cleaning.unresolved_rows,
                    }
                    connection.execute(
                        text("UPDATE sensor_import_batches SET cleaning_options_json=:options WHERE import_id=:id"),
                        {"options": options.to_json(audit_counts), "id": import_id},
                    )
                self._replace_measurements(connection, rebuilt)
                connection.execute(
                    text("UPDATE sensor_import_batches SET database_rows_after_import=:count WHERE import_id=:id"),
                    {"count": len(rebuilt), "id": import_id},
                )
        except SensorDatabaseError:
            raise
        except SQLAlchemyError as error:
            raise SensorDatabaseError(f"MySQL 入库失败，事务已回滚：{error}") from error
        return ImportResult(
            import_id,
            inserted,
            updated,
            unchanged,
            len(rebuilt),
            int(upload.audit["rejected_rows"]),
            bucket_rows=cleaning.bucket_rows,
            usable_rows=cleaning.usable_rows,
            imputed_rows=cleaning.imputed_rows,
            unresolved_rows=cleaning.unresolved_rows,
            field_summary=tuple(item.to_dict() for item in cleaning.field_summary),
            affected_rows=tuple(cleaning.affected_rows.to_dict("records")),
            temperature_jump_check_enabled=options.enabled,
            temperature_rule_version=options.rule_version,
            **temperature_counts,
        )

    def delete_import(self, import_id: str) -> DeleteResult:
        """Mark one active batch deleted and rebuild current rows transactionally."""

        try:
            UUID(import_id)
        except ValueError as error:
            raise SensorDatabaseError("导入批次 ID 格式无效。") from error
        try:
            with self.engine.begin() as connection:
                batch = connection.execute(
                    text(
                        "SELECT * FROM sensor_import_batches WHERE import_id=:id"
                        + self._for_update_suffix(connection)
                    ),
                    {"id": import_id},
                ).mappings().first()
                if not batch:
                    raise SensorDatabaseError("找不到指定的导入批次。")
                if batch["status"] != ACTIVE_STATUS:
                    raise SensorDatabaseError("该导入批次已经删除，不能重复删除。")
                current = self._read_measurements(connection)
                deleted_at = datetime.now(timezone.utc)
                connection.execute(
                    text("UPDATE sensor_import_batches SET status=:status, deleted_at_utc=:deleted WHERE import_id=:id"),
                    {"status": DELETED_STATUS, "deleted": deleted_at, "id": import_id},
                )
                rebuilt = self._rebuild(connection)
                self._replace_measurements(connection, rebuilt)
                removed, restored = CsvSensorDatabase._deletion_effects(current, rebuilt)
        except SensorDatabaseError:
            raise
        except SQLAlchemyError as error:
            raise SensorDatabaseError(f"MySQL 删除失败，事务已回滚：{error}") from error
        return DeleteResult(import_id, 0, removed, restored, len(rebuilt), Path("MySQL transaction"))

    def load_measurements(self) -> pd.DataFrame:
        with self.engine.connect() as connection:
            return self._read_measurements(connection)

    @staticmethod
    def _for_update_suffix(connection) -> str:
        """Keep MySQL row locking while allowing the SQLite contract tests to run."""

        return " FOR UPDATE" if connection.dialect.name == "mysql" else ""

    def load_cleaned_measurements(self) -> pd.DataFrame:
        """Build the model table from MySQL measurements without a CSV store."""

        from .processing import current_processing_version

        with self.engine.begin() as connection:
            measurements = self._read_measurements(connection)
            if current_processing_version() == "legacy":
                return self._clean_measurements(connection, measurements).frame.copy()
            prepared = self._prepare_cleaning_measurements(connection, measurements)
            pipeline = SensorCleaningPipeline.for_project(self.project_root)
        return cleaned_frame_cache.load(str(self.project_root.resolve()), pipeline, prepared)

    @staticmethod
    def _supports_cleaning_options(connection) -> bool:
        return any(
            column["name"] == "cleaning_options_json"
            for column in inspect(connection).get_columns("sensor_import_batches")
        )

    def migrate_cleaning_options(self, *, apply: bool = False) -> bool:
        """预检或幂等新增批次元数据列；返回迁移前是否需要升级。

        MySQL DDL 会隐式提交，因此这里只执行单条可重入的加列操作，绝不伴随数据重建。
        """

        try:
            with self.engine.begin() as connection:
                needed = not self._supports_cleaning_options(connection)
                if needed and apply:
                    connection.execute(text(
                        "ALTER TABLE sensor_import_batches ADD COLUMN cleaning_options_json TEXT NULL"
                    ))
                return needed
        except SQLAlchemyError as error:
            raise SensorDatabaseError(f"批次清洗设置结构升级失败：{error}") from error

    def _read_batch_options(self, connection, import_id: str) -> CleaningOptions | None:
        if not self._supports_cleaning_options(connection):
            return None
        value = connection.execute(
            text("SELECT cleaning_options_json FROM sensor_import_batches WHERE import_id=:id"),
            {"id": import_id},
        ).scalar_one()
        return parse_cleaning_options(value)

    def _clean_measurements(self, connection: Connection, measurements: pd.DataFrame) -> CleaningResult:
        """同一事务快照内读取原值和批次设置，供导入及后续查询共用。"""

        prepared = self._prepare_cleaning_measurements(connection, measurements)
        return SensorCleaningPipeline.for_project(self.project_root).clean(prepared)

    def _prepare_cleaning_measurements(self, connection: Connection, measurements: pd.DataFrame) -> pd.DataFrame:
        """Validate source batch signatures and attach their original settings."""

        prepared = measurements.copy()
        if self._supports_cleaning_options(connection) and not prepared.empty:
            rows = connection.execute(text(
                "SELECT import_id, cleaning_options_json FROM sensor_import_batches WHERE status=:status"
            ), {"status": ACTIVE_STATUS}).mappings().all()
            modes = {}
            versions = {}
            policy_sha = SensorCleaningPipeline.for_project(self.project_root).policy.sha256
            for row in rows:
                options = parse_cleaning_options(row["cleaning_options_json"])
                if options is not None and options.base_policy_sha256 != policy_sha:
                    raise SensorDatabaseError("批次基础清洗策略已变更，不能静默使用新策略重算。")
                modes[str(row["import_id"])] = options.enabled if options else False
                versions[str(row["import_id"])] = options.rule_version if options else "temperature_jump_v1"
            missing = set(prepared["import_id"].astype(str)) - modes.keys()
            if missing:
                raise SensorDatabaseError("当前测量值缺少有效来源批次设置。")
            prepared["temperature_jump_check_enabled"] = prepared["import_id"].astype(str).map(modes)
            prepared["temperature_rule_version"] = prepared["import_id"].astype(str).map(versions)
        else:
            prepared["temperature_jump_check_enabled"] = False
            prepared["temperature_rule_version"] = "temperature_jump_v1"
        return prepared

    def clean_prediction_upload(
        self, incoming: pd.DataFrame, *, temperature_jump_check_enabled: bool = False
    ) -> CleaningResult:
        """Clean a transient upload without writes; v3 simulates an import.

        Earlier profiles retain their existing prediction behavior.
        """

        from .cleaning_options import CURRENT_REPAIR_RULE_VERSION, PERSONAL_FIXED_TEMPERATURE_RULE_VERSION
        from .processing import current_processing_version, is_final_processing

        with self.engine.connect() as connection:
            history = self._prepare_cleaning_measurements(
                connection, self._read_measurements(connection)
            )
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="raise")
        proposed = incoming.copy()
        proposed["timestamp"] = pd.to_datetime(proposed["timestamp"], errors="raise")
        if current_processing_version() == "final_v3":
            if not isinstance(temperature_jump_check_enabled, bool):
                raise SensorDatabaseError("温度异常检测设置必须为布尔值。")
            proposed["temperature_jump_check_enabled"] = temperature_jump_check_enabled
            proposed["temperature_rule_version"] = PERSONAL_FIXED_TEMPERATURE_RULE_VERSION
            proposed["imported_at_utc"] = pd.Timestamp.now(tz="UTC")
            proposed["import_id"] = "prediction-transient"
            # Import replaces exact timestamps, not every observation in a bucket.
            untouched = history.loc[~history["timestamp"].isin(set(proposed["timestamp"]))]
            combined = pd.concat([untouched, proposed], ignore_index=True, sort=False)
            return SensorCleaningPipeline.for_project(self.project_root).clean(combined)
        if is_final_processing():
            # Upload prediction is a fresh transient processing request, not a
            # replay of the import policy that happened to own the same bucket.
            proposed["temperature_jump_check_enabled"] = True
            proposed["temperature_rule_version"] = CURRENT_REPAIR_RULE_VERSION
            history["temperature_jump_check_enabled"] = True
            history["temperature_rule_version"] = CURRENT_REPAIR_RULE_VERSION
        elif history.empty:
            proposed["temperature_jump_check_enabled"] = True
            proposed["temperature_rule_version"] = "temperature_jump_v3"
        else:
            context_columns = ["temperature_jump_check_enabled", "temperature_rule_version"]
            history = history.sort_values("timestamp", kind="stable")
            # Match on the same rounded bucket used by import and display, even
            # when the original observation arrived a few seconds after it.
            context = history[["timestamp", *context_columns]].copy()
            context["timestamp"] = context["timestamp"].dt.round("10min")
            context = context.drop_duplicates("timestamp", keep="last")
            proposed = pd.merge_asof(
                proposed.drop(columns=context_columns, errors="ignore").sort_values("timestamp"),
                context, on="timestamp", direction="backward",
            )
            for name in context_columns:
                proposed[name] = proposed[name].fillna(context[name].iloc[0])
        # Incoming input wins for its bucket; all untouched buckets keep history.
        incoming_times = set(proposed["timestamp"].dt.round("10min"))
        untouched = history.loc[~history["timestamp"].dt.round("10min").isin(incoming_times)]
        combined = pd.concat([untouched, proposed], ignore_index=True, sort=False)
        return SensorCleaningPipeline.for_project(self.project_root).clean(combined)

    @staticmethod
    def _temperature_counts(cleaned: pd.DataFrame, incoming: pd.DataFrame) -> dict[str, int]:
        """只计本批次观测时间涉及的桶；不是整个数据库的合计。"""

        timestamps = pd.to_datetime(incoming["timestamp"]).dt.round("10min")
        selected = cleaned.loc[cleaned["timestamp"].isin(timestamps)]
        suspect = selected["temperature_jump_outlier"].astype(bool)
        if "temperature_cusum_outlier" in selected:
            suspect |= selected["temperature_cusum_outlier"].astype(bool)
        if "temperature_daily_corrected" in selected:
            suspect |= selected["temperature_daily_corrected"].astype(bool)
        return {
            "temperature_suspicious_rows": int(suspect.sum()),
            "temperature_repaired_rows": int((suspect & selected["temperature_repaired"].astype(bool)).sum()),
            "temperature_interpolated_rows": int((suspect & selected["temperature_imputed"].astype(bool)).sum()),
            "temperature_offset_corrected_rows": int(selected["temperature_offset_corrected"].sum()),
            "temperature_isolated_rows": int((suspect & selected["temperature_unresolved"].astype(bool)).sum()),
            "temperature_review_rows": int(selected["temperature_review_required"].sum()),
            "temperature_hard_unresolved_rows": int((
                selected["temperature_physical_outlier"] | selected["temperature_bucket_conflict"]
            ).sum()),
        }

    def load_import_log(self) -> pd.DataFrame:
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        "SELECT * FROM sensor_import_batches "
                        "ORDER BY imported_at_utc"
                    )
                )
                .mappings()
                .all()
            )
        if not rows:
            return pd.DataFrame(columns=IMPORT_LOG_COLUMNS)
        frame = pd.DataFrame(rows)
        for column in ("imported_at_utc", "deleted_at_utc"):
            if column in frame:
                frame[column] = frame[column].fillna("").astype(str)
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
        # pandas 将混合字符串列中的 SQL NULL 转为 NaN；还原历史空设置语义后再严格校验。
        frame["cleaning_options_json"] = frame["cleaning_options_json"].fillna("")
        for index, value in frame["cleaning_options_json"].items():
            options = parse_cleaning_options(value)
            if options is not None:
                document = json.loads(value)
                frame.loc[index, "cleaning_policy_sha256"] = options.base_policy_sha256
                for column in (
                    "cleaning_bucket_rows", "cleaning_usable_rows",
                    "cleaning_imputed_rows", "cleaning_unresolved_rows",
                ):
                    frame.loc[index, column] = document.get("counts", {}).get(column, 0)
        return frame.loc[:, list(IMPORT_LOG_COLUMNS)]

    def active_imports(self) -> pd.DataFrame:
        return self.load_import_log().loc[lambda frame: frame["status"] == ACTIVE_STATUS].copy()

    def status(self) -> DatabaseStatus:
        from .processing import retained_processed_rows

        measurements = self.load_measurements()
        cleaned = retained_processed_rows(self.load_cleaned_measurements())
        imports = self.load_import_log()
        timestamps = pd.to_datetime(measurements["timestamp"], errors="coerce") if not measurements.empty else pd.Series(dtype="datetime64[ns]")
        return DatabaseStatus(
            len(measurements),
            int((imports["status"] == ACTIVE_STATUS).sum()),
            int((imports["status"] == DELETED_STATUS).sum()),
            timestamps.min().isoformat() if not timestamps.empty else None,
            timestamps.max().isoformat() if not timestamps.empty else None,
            cleaned_rows=len(cleaned),
            usable_rows=int(cleaned.get("model_sample_usable", pd.Series(dtype=bool)).sum()),
            imputed_rows=int(cleaned.get("model_input_imputed", pd.Series(dtype=bool)).sum()),
            unresolved_rows=int(cleaned.get("model_input_unresolved", pd.Series(dtype=bool)).sum()),
        )

    def training_snapshot(self) -> TrainingSnapshot:
        measurements = self.load_cleaned_measurements()
        if "history_barrier" in measurements:
            measurements["history_segment"] = measurements["history_barrier"].fillna(False).astype(bool).cumsum()
        if measurements.empty:
            raise SensorDatabaseError("MySQL 数据库尚无可训练数据。")
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
                "cleaning_policy_sha256",
                "cleaning_profile_sha256",
                "base_cleaning_policy_sha256",
                "processing_version",
                "tilt1_inline_gap_imputed",
                "tilt1_crossline_gap_imputed",
                "temperature_gap_imputed",
                "temperature_jump_check_enabled",
                "temperature_rule_version",
                "temperature_repair_method",
                "temperature_offset_celsius",
                "temperature_event_id",
                "temperature_repaired",
                "temperature_review_required",
                "temperature_review_reason",
                *CUSUM_AUDIT_COLUMNS,
                *DAILY_AUDIT_COLUMNS,
                "temperature_zero_invalid",
                "temperature_zero_interpolated",
                "processed_row_excluded",
                "history_barrier",
                "history_segment",
                "import_id",
                "gateway",
            )
            if column in measurements
        ]
        complete = measurements.loc[
            complete_mask, [*required, *context_columns]
        ].copy()
        numeric = complete.loc[:, list(MEASUREMENT_COLUMNS)].to_numpy(dtype=float)
        if len(complete) and not np.isfinite(numeric).all():
            raise SensorDatabaseError("MySQL 训练数据包含非有限数值。")
        if complete.empty:
            raise SensorDatabaseError("MySQL 没有同时包含特征和真值的记录。")
        complete["timestamp"] = pd.to_datetime(complete["timestamp"], errors="raise")
        complete = complete.sort_values("timestamp", kind="stable").reset_index(drop=True)
        digest = hashlib.sha256(
            measurements.to_csv(index=False).encode("utf-8")
        ).hexdigest()

        return TrainingSnapshot(complete, digest, len(measurements), int((~complete_mask).sum()))

    def _read_measurements(self, connection) -> pd.DataFrame:
        rows = connection.execute(text("SELECT * FROM sensor_measurements ORDER BY timestamp, gateway")).mappings().all()
        if not rows:
            return pd.DataFrame(columns=DATABASE_COLUMNS)
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise").dt.strftime("%Y-%m-%d %H:%M:%S")
        frame["imported_at_utc"] = pd.to_datetime(frame["imported_at_utc"], errors="raise").astype(str)
        return frame.loc[:, list(DATABASE_COLUMNS)]

    def _insert_rows(self, connection, frame: pd.DataFrame) -> None:
        query = text("""INSERT INTO sensor_import_rows
            (import_id,schema_version,gateway,timestamp,source_row_count,tilt1_inline_model,
             tilt1_crossline_model,temperature_celsius,tilt2_inline,tilt2_crossline,
             source_filename,source_sheet,source_sha256,imported_at_utc)
            VALUES (:import_id,:schema_version,:gateway,:timestamp,:source_row_count,:tilt1_inline_model,
             :tilt1_crossline_model,:temperature_celsius,:tilt2_inline,:tilt2_crossline,
             :source_filename,:source_sheet,:source_sha256,:imported_at_utc)""")
        connection.execute(query, [self._row_params(row) for _, row in frame.iterrows()])

    def _rebuild(self, connection) -> pd.DataFrame:
        query = text("""SELECT r.* FROM sensor_import_rows r
            JOIN sensor_import_batches b ON b.import_id=r.import_id
            WHERE b.status='active' ORDER BY b.imported_at_utc, b.import_id, r.timestamp, r.gateway""")
        rows = connection.execute(query).mappings().all()
        if not rows:
            return pd.DataFrame(columns=DATABASE_COLUMNS)
        frame = pd.DataFrame(rows)
        frame = frame.drop_duplicates(subset=["gateway", "timestamp"], keep="last")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        frame["imported_at_utc"] = pd.to_datetime(frame["imported_at_utc"]).astype(str)
        return frame.loc[:, list(DATABASE_COLUMNS)].sort_values(["timestamp", "gateway"], kind="stable").reset_index(drop=True)

    def _replace_measurements(self, connection, frame: pd.DataFrame) -> None:
        connection.execute(text("DELETE FROM sensor_measurements"))
        if frame.empty:
            return
        query = text("""INSERT INTO sensor_measurements
            (schema_version,gateway,timestamp,source_row_count,tilt1_inline_model,
             tilt1_crossline_model,temperature_celsius,tilt2_inline,tilt2_crossline,
             source_filename,source_sheet,source_sha256,import_id,imported_at_utc)
            VALUES (:schema_version,:gateway,:timestamp,:source_row_count,:tilt1_inline_model,
             :tilt1_crossline_model,:temperature_celsius,:tilt2_inline,:tilt2_crossline,
             :source_filename,:source_sheet,:source_sha256,:import_id,:imported_at_utc)""")
        connection.execute(query, [self._row_params(row, include_schema=True) for _, row in frame.iterrows()])

    @staticmethod
    def _row_params(row: pd.Series, include_schema: bool = False) -> dict[str, Any]:
        result = {column: row[column] for column in DATABASE_COLUMNS if column in row}
        if include_schema:
            result["schema_version"] = int(row["schema_version"])
        result["timestamp"] = pd.Timestamp(row["timestamp"]).to_pydatetime()
        result["imported_at_utc"] = pd.Timestamp(row["imported_at_utc"]).to_pydatetime()
        for key in ("tilt2_inline", "tilt2_crossline"):
            if pd.isna(result.get(key)):
                result[key] = None
        return result

    @staticmethod
    def _batch_params(frame, audit, inserted, updated, unchanged, imported_at):
        first = frame.iloc[0]
        return {
            "import_id": str(first["import_id"]), "schema_version": int(first["schema_version"]),
            "status": ACTIVE_STATUS, "imported_at_utc": imported_at, "gateway": GATEWAY,
            "source_filename": str(first["source_filename"]), "source_sheet": str(first["source_sheet"]),
            "source_sha256": str(first["source_sha256"]), "source_rows": int(audit["source_rows"]),
            "accepted_rows": int(audit["accepted_rows"]), "rejected_rows": int(audit["rejected_rows"]),
            "time_series_rows": int(audit.get("time_series_rows", len(frame))),
            "duplicates": int(audit.get("duplicate_timestamp_rows_combined", 0)),
            "inserted_rows": inserted, "updated_rows": updated, "unchanged_rows": unchanged,
        }
