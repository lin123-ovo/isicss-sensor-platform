"""MySQL model registry with a disposable artifact cache."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import mimetypes
import os
import shutil
import tempfile
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from sensor_app.mysql_config import MySQLConfig
from sensor_database.mysql_database import MySQLSensorDatabase

from .model_registry import DraftRun, ModelRegistry, ModelRegistryError, ModelVersion


class MySQLModelRegistry(ModelRegistry):
    """Keep the existing registry API while making MySQL the source of truth."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.expanduser().resolve()
        self.database = MySQLSensorDatabase.from_env(self.project_root)
        self.cache_root = Path(tempfile.gettempdir()) / "csg124-model-cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.registry_dir = self.cache_root
        self.drafts_dir = self.cache_root / "drafts"
        self.index_path = self.cache_root / "versions.csv"
        self.active_path = self.cache_root / "active.json"
        self.lock_path = self.cache_root / ".registry.lock"
        self.legacy_bootstrap_disabled_path = (
            self.cache_root / ".mysql-cleaning-v2-required"
        )
        self.legacy_bootstrap_disabled_path.touch(exist_ok=True)

    def bootstrap_current_version(self) -> ModelVersion | None:
        versions = self.list_versions()
        if not versions:
            return None
        return self.get_active_version()

    def list_versions(self) -> tuple[ModelVersion, ...]:
        with self.database.engine.connect() as connection:
            rows = connection.execute(text("SELECT * FROM model_versions ORDER BY CAST(SUBSTRING(version_id, 2) AS UNSIGNED) DESC")).mappings().all()
        return tuple(ModelVersion.from_row(dict(row)) for row in rows)

    def get_version(self, version_id: str) -> ModelVersion:
        with self.database.engine.connect() as connection:
            row = connection.execute(text("SELECT * FROM model_versions WHERE version_id=:id"), {"id": version_id}).mappings().first()
        if not row:
            raise ModelRegistryError(f"找不到模型版本：{version_id}")
        self.version_directory(version_id)
        return ModelVersion.from_row(dict(row))

    def get_active_version(self) -> ModelVersion:
        with self.database.engine.connect() as connection:
            row = connection.execute(text("SELECT version_id FROM model_active_version WHERE singleton_id=1")).scalar()
        if not row:
            raise ModelRegistryError("MySQL 尚未配置活动模型版本。")
        return self.get_version(str(row))

    def get_metadata(self, version_id: str) -> dict[str, Any]:
        return json.loads((self.version_directory(version_id) / "metadata.json").read_text(encoding="utf-8-sig"))

    def get_parameters(self, version_id: str) -> dict[str, Any]:
        return self.get_metadata(version_id)["parameters"]

    def version_directory(self, version_id: str) -> Path:
        directory = self.cache_root / "versions" / version_id
        directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "model.joblib", "metadata.json", "metrics.csv", "parameter_candidates.csv",
            "parameter_search.csv", "test_predictions.csv", "split_plan.json", "selected_parameters.json",
        ):
            self._materialize(version_id, "version", directory, name)
        self._materialize_optional(
            version_id,
            "version",
            directory,
            "crossline_residual_search.csv",
        )
        return directory

    def draft_directory(self, run_id: str) -> Path:
        directory = self.drafts_dir / run_id
        directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "model.joblib", "metadata.json", "metrics.csv", "parameter_candidates.csv",
            "parameter_search.csv", "test_predictions.csv", "split_plan.json", "selected_parameters.json",
        ):
            self._materialize(run_id, "draft", directory, name)
        self._materialize_optional(
            run_id,
            "draft",
            directory,
            "crossline_residual_search.csv",
        )
        return directory

    def list_drafts(self) -> tuple[DraftRun, ...]:
        with self.database.engine.connect() as connection:
            rows = connection.execute(text("SELECT run_id, created_at_utc FROM model_drafts ORDER BY created_at_utc DESC")).mappings().all()
        return tuple(DraftRun(str(row["run_id"]), str(row["created_at_utc"]), self.draft_directory(str(row["run_id"]))) for row in rows)

    def delete_draft(self, run_id: str) -> None:
        """Permanently delete one training candidate and all of its artifacts."""

        self._validate_identifier(run_id, "run")
        try:
            with self.database.engine.begin() as connection:
                exists = connection.execute(
                    text("SELECT 1 FROM model_drafts WHERE run_id=:id"),
                    {"id": run_id},
                ).scalar()
                if exists is None:
                    raise ModelRegistryError(f"找不到训练候选：{run_id}")
                connection.execute(
                    text(
                        "DELETE FROM model_artifacts "
                        "WHERE owner_type='draft' AND owner_id=:id"
                    ),
                    {"id": run_id},
                )
                deleted = connection.execute(
                    text("DELETE FROM model_drafts WHERE run_id=:id"),
                    {"id": run_id},
                )
                if deleted.rowcount != 1:
                    raise ModelRegistryError(f"删除训练候选失败：{run_id}")
        except SQLAlchemyError as error:
            raise ModelRegistryError(f"删除训练候选失败：{error}") from error

        shutil.rmtree(self.drafts_dir / run_id, ignore_errors=True)

    def create_training_draft(self, **kwargs) -> DraftRun:
        """Stage a candidate through the existing serializer, then upload it transactionally."""

        draft = super().create_training_draft(**kwargs)
        metadata = json.loads((draft.directory / "metadata.json").read_text(encoding="utf-8-sig"))
        with self.database.engine.begin() as connection:
            connection.execute(
                text("INSERT INTO model_drafts (run_id,created_at_utc,metadata_json,status) VALUES (:id,:at,:metadata,:status)"),
                {"id": draft.run_id, "at": pd.Timestamp(draft.created_at_utc).to_pydatetime(), "metadata": json.dumps(metadata, ensure_ascii=False), "status": str(metadata.get("status", "candidate"))},
            )
            self._upload_artifacts(connection, draft.directory, "draft", draft.run_id)
        return draft

    def save_draft(
        self,
        run_id: str,
        note: str = "",
        *,
        activate: bool = False,
        delete_source: bool = False,
    ) -> ModelVersion:
        """Publish a draft and optionally consume it in the same transaction."""

        draft_dir = self.draft_directory(run_id)
        metadata = json.loads((draft_dir / "metadata.json").read_text(encoding="utf-8-sig"))
        self._validate_artifact_directory(draft_dir)
        try:
            self.database.ensure_linear_model_schema()
        except SQLAlchemyError as error:
            raise ModelRegistryError("模型参数索引兼容扩展失败，请检查数据库权限。") from error
        versions = self.list_versions()
        version_id = f"v{max((int(item.version_id[1:]) for item in versions), default=0) + 1}"
        version_dir = self.cache_root / "versions" / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)
        shutil.copytree(draft_dir, version_dir)
        predictions = version_dir / "test_predictions.csv"
        if predictions.is_file():
            frame = pd.read_csv(predictions)
            frame["model_version"] = version_id
            frame.to_csv(predictions, index=False, encoding="utf-8-sig")
        safe_note = note[:200] or self._default_version_note(metadata)
        metadata["version_id"] = version_id
        metadata["note"] = safe_note
        (version_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        version = self._version_from_metadata(version_id, metadata, safe_note, self._sha256(version_dir / "model.joblib"))
        try:
            with self.database.engine.begin() as connection:
                values = version.to_row()
                values["created_at_utc"] = pd.Timestamp(version.created_at_utc).to_pydatetime()
                connection.execute(
                    text("""INSERT INTO model_versions
                    (version_id,created_at_utc,note,gateway,model,epsilon,alpha,inline_epsilon,
                     inline_alpha,crossline_epsilon,crossline_alpha,max_iter,split_ratio,
                     dataset_version,source_sha256,status,model_sha256)
                    VALUES (:version_id,:created_at_utc,:note,:gateway,:model,:epsilon,:alpha,
                     :inline_epsilon,:inline_alpha,:crossline_epsilon,:crossline_alpha,:max_iter,
                     :split_ratio,:dataset_version,:source_sha256,:status,:model_sha256)"""), values,
                )
                self._upload_artifacts(connection, version_dir, "version", version_id)
                if activate:
                    self._set_active_version(connection, version_id)
                if delete_source:
                    connection.execute(
                        text(
                            "DELETE FROM model_artifacts "
                            "WHERE owner_type='draft' AND owner_id=:id"
                        ),
                        {"id": run_id},
                    )
                    deleted = connection.execute(
                        text("DELETE FROM model_drafts WHERE run_id=:id"),
                        {"id": run_id},
                    )
                    if deleted.rowcount != 1:
                        raise ModelRegistryError(f"保存后清理训练候选失败：{run_id}")
        except SQLAlchemyError as error:
            raise ModelRegistryError("模型保存事务失败，原活动模型保持不变。") from error
        if delete_source:
            shutil.rmtree(self.drafts_dir / run_id, ignore_errors=True)
        return version

    def update_note(self, version_id: str, note: str) -> ModelVersion:
        with self.database.engine.begin() as connection:
            result = connection.execute(text("UPDATE model_versions SET note=:note WHERE version_id=:id"), {"note": note[:200], "id": version_id})
            if result.rowcount != 1:
                raise ModelRegistryError(f"找不到模型版本：{version_id}")
            path = self.version_directory(version_id) / "metadata.json"
            metadata = json.loads(path.read_text(encoding="utf-8-sig"))
            metadata["note"] = note[:200]
            path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self.get_version(version_id)

    def activate_version(self, version_id: str) -> ModelVersion:
        version = self.get_version(version_id)
        self._validate_version_artifact(version)
        with self.database.engine.begin() as connection:
            self._set_active_version(connection, version_id)
        return version

    def delete_version(self, version_id: str) -> str | None:
        """Delete one saved MySQL model and return its active replacement, if any.

        Model artifacts belong to the version and are deleted in the same
        transaction.  If the deleted version is active, the newest remaining
        version becomes active; removing the final version deliberately leaves
        the registry without an active model.
        """

        self._validate_identifier(version_id, "model")
        try:
            with self.database.engine.begin() as connection:
                exists = connection.execute(
                    text("SELECT 1 FROM model_versions WHERE version_id=:id"),
                    {"id": version_id},
                ).scalar()
                if exists is None:
                    raise ModelRegistryError(f"找不到模型版本：{version_id}")

                active_id = connection.execute(
                    text(
                        "SELECT version_id FROM model_active_version "
                        "WHERE singleton_id=1"
                    )
                ).scalar()
                replacement_id: str | None = None
                if active_id == version_id:
                    remaining_ids = [
                        str(item)
                        for item in connection.execute(
                            text(
                                "SELECT version_id FROM model_versions "
                                "WHERE version_id<>:id"
                            ),
                            {"id": version_id},
                        ).scalars()
                    ]
                    if remaining_ids:
                        replacement_id = max(
                            remaining_ids,
                            key=lambda item: int(item[1:]),
                        )
                        self._set_active_version(connection, replacement_id)
                    else:
                        connection.execute(
                            text(
                                "DELETE FROM model_active_version "
                                "WHERE singleton_id=1"
                            )
                        )

                connection.execute(
                    text(
                        "DELETE FROM model_artifacts "
                        "WHERE owner_type='version' AND owner_id=:id"
                    ),
                    {"id": version_id},
                )
                deleted = connection.execute(
                    text("DELETE FROM model_versions WHERE version_id=:id"),
                    {"id": version_id},
                )
                if deleted.rowcount != 1:
                    raise ModelRegistryError(f"删除模型版本失败：{version_id}")
        except SQLAlchemyError as error:
            raise ModelRegistryError(f"删除模型版本失败：{error}") from error

        # This is only a disposable local cache.  The database transaction has
        # already completed, so a stale cache must not make deletion fail.
        shutil.rmtree(self.cache_root / "versions" / version_id, ignore_errors=True)
        return replacement_id

    @staticmethod
    def _set_active_version(
        connection: Any,
        version_id: str,
        *,
        activated_at: datetime | None = None,
    ) -> None:
        """Create or update the singleton active-model pointer.

        A newly initialized database has an empty ``model_active_version``
        table, so an UPDATE-only implementation silently affects zero rows.
        """

        timestamp = activated_at or datetime.now(timezone.utc)
        updated = connection.execute(
            text(
                "UPDATE model_active_version "
                "SET version_id=:id, activated_at_utc=:at WHERE singleton_id=1"
            ),
            {"id": version_id, "at": timestamp},
        )
        if updated.rowcount == 0:
            connection.execute(
                text(
                    "INSERT INTO model_active_version "
                    "(singleton_id, version_id, activated_at_utc) "
                    "VALUES (1, :id, :at)"
                ),
                {"id": version_id, "at": timestamp},
            )

    def _artifact_path(self, owner_id: str, name: str) -> Path:
        directory = self.cache_root / "versions" / owner_id
        directory.mkdir(parents=True, exist_ok=True)
        self._materialize(owner_id, "version", directory, name)
        return directory / name

    def _materialize(self, owner_id: str, owner_type: str, directory: Path, name: str) -> None:
        path = directory / name
        with self.database.engine.connect() as connection:
            row = connection.execute(text("SELECT sha256, content FROM model_artifacts WHERE owner_type=:type AND owner_id=:id AND artifact_name=:name"), {"type": owner_type, "id": owner_id, "name": name}).mappings().first()
        if not row:
            raise ModelRegistryError(f"模型 artifact 缺失：{owner_type}/{owner_id}/{name}")
        if path.is_file() and self._sha256(path) == row["sha256"]:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(bytes(row["content"]))
        os.replace(temporary, path)

    def _materialize_optional(
        self,
        owner_id: str,
        owner_type: str,
        directory: Path,
        name: str,
    ) -> None:
        """Materialize a new artifact without breaking legacy model versions."""

        path = directory / name
        with self.database.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT sha256, content FROM model_artifacts "
                    "WHERE owner_type=:type AND owner_id=:id AND artifact_name=:name"
                ),
                {"type": owner_type, "id": owner_id, "name": name},
            ).mappings().first()
        if not row:
            return
        if path.is_file() and self._sha256(path) == row["sha256"]:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(bytes(row["content"]))
        os.replace(temporary, path)

    @staticmethod
    def _upload_artifacts(connection, directory: Path, owner_type: str, owner_id: str) -> None:
        names = (
            "model.joblib", "metadata.json", "metrics.csv", "parameter_candidates.csv",
            "parameter_search.csv", "test_predictions.csv", "split_plan.json", "selected_parameters.json",
            "crossline_residual_search.csv",
        )
        import hashlib
        from datetime import datetime, timezone

        for name in names:
            path = directory / name
            if not path.is_file():
                continue
            payload = path.read_bytes()
            connection.execute(
                text("""INSERT INTO model_artifacts
                (owner_type,owner_id,artifact_name,mime_type,sha256,size_bytes,content,created_at_utc)
                VALUES (:owner_type,:owner_id,:artifact_name,:mime_type,:sha256,:size_bytes,:content,:created_at_utc)"""),
                {"owner_type": owner_type, "owner_id": owner_id, "artifact_name": name,
                 "mime_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
                 "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
                 "content": payload, "created_at_utc": datetime.now(timezone.utc)},
            )

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()
