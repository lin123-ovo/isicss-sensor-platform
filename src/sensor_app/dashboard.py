"""Read-only dashboard view model for the CSG124 workbench."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from sensor_database import SensorDatabaseError, get_sensor_database
from tilt_predictor.model_registry import ModelRegistryError
from tilt_predictor.registry_factory import get_model_registry


@dataclass(frozen=True)
class DashboardSnapshot:
    """Small read-only projection used by the system overview."""

    gateway: str = "CSG124"
    database_rows: int | None = None
    cleaned_rows: int | None = None
    active_imports: int | None = None
    deleted_imports: int | None = None
    timestamp_min: str | None = None
    timestamp_max: str | None = None
    active_model_version: str | None = None
    model_status: str | None = None
    latest_import_time: str | None = None
    active_model_created_at: str | None = None
    measurements: pd.DataFrame = field(
        default_factory=pd.DataFrame,
        repr=False,
        compare=False,
    )
    recent_imports: pd.DataFrame = field(
        default_factory=pd.DataFrame,
        repr=False,
        compare=False,
    )
    warnings: tuple[str, ...] = ()


def load_dashboard_snapshot(project_root: Path) -> DashboardSnapshot:
    """Load dashboard data without bootstrapping, training, publishing, or writing."""

    warnings: list[str] = []
    database_rows: int | None = None
    cleaned_rows: int | None = None
    active_imports: int | None = None
    deleted_imports: int | None = None
    timestamp_min: str | None = None
    timestamp_max: str | None = None
    latest_import_time: str | None = None
    measurements = pd.DataFrame()
    recent_imports = pd.DataFrame()

    try:
        database = get_sensor_database(project_root)
        status = database.status()
        load_cleaned = getattr(database, "load_cleaned_measurements", None)
        measurements = (
            load_cleaned() if callable(load_cleaned) else database.load_measurements()
        )
        from sensor_database.processing import retained_processed_rows

        measurements = retained_processed_rows(measurements)
        import_log = database.load_import_log()
    except SensorDatabaseError as error:
        warnings.append(f"数据库状态不可用：{error}")
    else:
        database_rows = status.measurement_rows
        cleaned_rows = len(measurements)
        active_imports = status.active_imports
        deleted_imports = status.deleted_imports
        timestamp_min = status.timestamp_min
        timestamp_max = status.timestamp_max
        if not import_log.empty and "imported_at_utc" in import_log.columns:
            ordered = import_log.assign(
                _dashboard_imported_at=pd.to_datetime(
                    import_log["imported_at_utc"],
                    errors="coerce",
                    utc=True,
                )
            ).sort_values("_dashboard_imported_at", ascending=False)
            recent_imports = ordered.head(6).drop(
                columns=["_dashboard_imported_at"],
                errors="ignore",
            )
            latest = ordered["_dashboard_imported_at"].dropna()
            if not latest.empty:
                latest_import_time = latest.iloc[0].tz_convert("Asia/Shanghai").strftime(
                    "%Y-%m-%d %H:%M"
                )

    model = load_workspace_snapshot(project_root)
    return replace(
        model,
        database_rows=database_rows,
        cleaned_rows=cleaned_rows,
        active_imports=active_imports,
        deleted_imports=deleted_imports,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        latest_import_time=latest_import_time,
        measurements=measurements,
        recent_imports=recent_imports,
        warnings=tuple(warnings) + model.warnings,
    )


def load_workspace_snapshot(project_root: Path) -> DashboardSnapshot:
    """Read shell model status without loading or cleaning measurement data."""
    warnings: list[str] = []
    active_model_version: str | None = None
    model_status: str | None = None
    active_model_created_at: str | None = None
    try:
        registry = get_model_registry(project_root)
        bootstrap = getattr(registry, "bootstrap_current_version", None)
        active_model = (
            bootstrap() if callable(bootstrap) else registry.get_active_version()
        )
    except ModelRegistryError as error:
        warnings.append(f"模型状态不可用：{error}")
    else:
        if active_model is not None:
            active_model_version = active_model.version_id
            model_status = active_model.status
            active_model_created_at = active_model.created_at_utc

    return DashboardSnapshot(
        active_model_version=active_model_version,
        model_status=model_status,
        active_model_created_at=active_model_created_at,
        warnings=tuple(warnings),
    )
