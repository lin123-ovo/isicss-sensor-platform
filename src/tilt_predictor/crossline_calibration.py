"""Twelve-hour local calibration for the CSG124 crossline model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .temperature_pipeline import GATEWAY, TemperaturePipelineError


CALIBRATION_FEATURE_VERSION = "tilt_hybrid_12h_v2"
CROSSLINE_FEATURE_COLUMNS = (
    "tilt1_inline_delta",
    "tilt1_crossline_delta",
    "temperature_delta",
    "temperature_change_10m",
    "temperature_change_1h",
    "temperature_rolling_mean_1h_delta",
)

# These cleaned-table fields determine whether a row may contribute a
# calibration target and whether its time history is continuous enough for the
# crossline temperature features.  They must travel with transient inference
# uploads as well as database-backed prediction requests.
CALIBRATION_CONTEXT_COLUMNS = (
    "segment_id",
    "is_inserted_gap",
    "model_input_available",
    "model_input_unresolved",
    "quality_status",
)
CALIBRATION_INPUT_COLUMNS = (
    "tilt1_inline_model",
    "tilt1_crossline_model",
    "temperature_celsius",
    "tilt2_crossline",
)


@dataclass(frozen=True)
class CalibrationPolicy:
    """Validated policy for one recurring local-calibration cycle."""

    period_hours: int = 12
    window_minutes: int = 60
    bucket_minutes: int = 10
    minimum_samples: int = 6
    maximum_gap_minutes: int = 20

    def validate(self) -> None:
        if self.period_hours <= 0 or self.window_minutes <= 0:
            raise TemperaturePipelineError("校准周期和窗口必须大于0。")
        if self.window_minutes >= self.period_hours * 60:
            raise TemperaturePipelineError("校准窗口必须小于总周期。")
        if self.bucket_minutes <= 0 or self.minimum_samples <= 0:
            raise TemperaturePipelineError("校准时间桶和最小样本数必须大于0。")
        if self.maximum_gap_minutes < self.bucket_minutes:
            raise TemperaturePipelineError("允许的最大缺口不能小于时间桶。")

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class CalibrationSnapshot:
    """One auditable calibration baseline and its validity interval."""

    calibration_id: str
    gateway: str
    cycle_started_at: pd.Timestamp
    calibration_ended_at: pd.Timestamp
    next_calibration_at: pd.Timestamp
    sample_count: int
    tilt1_inline_baseline: float
    tilt1_crossline_baseline: float
    temperature_baseline: float
    tilt2_crossline_baseline: float
    source_import_ids: tuple[str, ...]
    status: str = "valid"

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        for name in (
            "cycle_started_at",
            "calibration_ended_at",
            "next_calibration_at",
        ):
            record[name] = pd.Timestamp(record[name]).isoformat()
        record["source_import_ids"] = json.dumps(
            list(self.source_import_ids), ensure_ascii=False
        )
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "CalibrationSnapshot":
        raw_ids = record.get("source_import_ids", "[]")
        ids = json.loads(raw_ids) if isinstance(raw_ids, str) else list(raw_ids)
        return cls(
            calibration_id=str(record["calibration_id"]),
            gateway=str(record["gateway"]),
            cycle_started_at=pd.Timestamp(record["cycle_started_at"]),
            calibration_ended_at=pd.Timestamp(record["calibration_ended_at"]),
            next_calibration_at=pd.Timestamp(record["next_calibration_at"]),
            sample_count=int(record["sample_count"]),
            tilt1_inline_baseline=float(record["tilt1_inline_baseline"]),
            tilt1_crossline_baseline=float(record["tilt1_crossline_baseline"]),
            temperature_baseline=float(record["temperature_baseline"]),
            tilt2_crossline_baseline=float(record["tilt2_crossline_baseline"]),
            source_import_ids=tuple(str(value) for value in ids),
            status=str(record.get("status", "valid")),
        )


@dataclass(frozen=True)
class CalibratedFrame:
    """Rows enriched with calibration state and crossline delta features."""

    frame: pd.DataFrame
    snapshots: tuple[CalibrationSnapshot, ...]
    invalid_cycles: tuple[dict[str, Any], ...]

    @property
    def prediction_rows(self) -> pd.DataFrame:
        return self.frame.loc[self.frame["calibration_status"] == "ready"].copy()


def build_calibrated_frame(
    frame: pd.DataFrame,
    policy: CalibrationPolicy | None = None,
    *,
    require_crossline_target: bool = True,
) -> CalibratedFrame:
    """Apply recurring calibration and return all rows with explicit state.

    Every cycle is anchored at its first row.  The first ``window_minutes`` are
    calibration-only; the remainder of the twelve-hour cycle is prediction time.
    """

    selected_policy = policy or CalibrationPolicy()
    selected_policy.validate()
    required = ["timestamp", *CALIBRATION_INPUT_COLUMNS[:-1]]
    if require_crossline_target:
        required.append("tilt2_crossline")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise TemperaturePipelineError(f"校准数据缺少字段：{missing}")
    if frame.empty:
        raise TemperaturePipelineError("没有可用于校准的数据。")

    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    ordered = ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if ordered["timestamp"].duplicated().any():
        raise TemperaturePipelineError("校准数据存在重复时间戳。")
    numeric_columns = list(CALIBRATION_INPUT_COLUMNS[:-1])
    quality_invalid = pd.Series(False, index=ordered.index)
    if "model_input_available" in ordered:
        quality_invalid |= ~ordered["model_input_available"].astype(bool)
    if "model_input_unresolved" in ordered:
        quality_invalid |= ordered["model_input_unresolved"].astype(bool)
    if "is_inserted_gap" in ordered:
        quality_invalid |= ordered["is_inserted_gap"].astype(bool)
    finite_rows = ordered.loc[~quality_invalid, numeric_columns]
    if len(finite_rows) and not np.isfinite(finite_rows.to_numpy(dtype=float)).all():
        raise TemperaturePipelineError("可用校准输入包含缺失或非有限值。")
    if "tilt2_crossline" not in ordered:
        ordered["tilt2_crossline"] = np.nan

    for column in (
        "calibration_id",
        "cycle_started_at",
        "calibration_ended_at",
        "next_calibration_at",
        "calibration_status",
    ):
        ordered[column] = pd.NA
    for column in (
        "tilt1_inline_baseline",
        "tilt1_crossline_baseline",
        "temperature_baseline",
        "tilt2_crossline_baseline",
        *CROSSLINE_FEATURE_COLUMNS,
        "tilt2_crossline_delta",
    ):
        ordered[column] = np.nan

    snapshots: list[CalibrationSnapshot] = []
    invalid_cycles: list[dict[str, Any]] = []
    period = pd.Timedelta(hours=selected_policy.period_hours)
    window = pd.Timedelta(minutes=selected_policy.window_minutes)
    cycle_start = ordered["timestamp"].iloc[0]
    maximum_timestamp = ordered["timestamp"].iloc[-1]
    while cycle_start <= maximum_timestamp:
        cycle_end = cycle_start + period
        calibration_end = cycle_start + window
        cycle_mask = (ordered["timestamp"] >= cycle_start) & (
            ordered["timestamp"] < cycle_end
        )
        calibration_mask = cycle_mask & (ordered["timestamp"] < calibration_end)
        prediction_mask = cycle_mask & ~calibration_mask
        window_frame = ordered.loc[calibration_mask].copy()
        valid_target = window_frame["tilt2_crossline"].notna()
        finite_target = pd.Series(False, index=window_frame.index)
        if valid_target.any():
            finite_target.loc[valid_target] = np.isfinite(
                window_frame.loc[valid_target, "tilt2_crossline"].to_numpy(dtype=float)
            )
        window_quality = ~quality_invalid.reindex(window_frame.index, fill_value=True)
        usable = window_frame.loc[valid_target & finite_target & window_quality]
        gaps = usable["timestamp"].diff().dropna()
        has_large_gap = bool(
            gaps.gt(pd.Timedelta(minutes=selected_policy.maximum_gap_minutes)).any()
        )
        is_valid = len(usable) >= selected_policy.minimum_samples and not has_large_gap
        if not is_valid:
            status = "insufficient_data"
            ordered.loc[cycle_mask, "calibration_status"] = status
            ordered.loc[cycle_mask, "cycle_started_at"] = cycle_start
            ordered.loc[cycle_mask, "calibration_ended_at"] = calibration_end
            ordered.loc[cycle_mask, "next_calibration_at"] = cycle_end
            invalid_cycles.append(
                {
                    "cycle_started_at": cycle_start.isoformat(),
                    "sample_count": int(len(usable)),
                    "has_large_gap": has_large_gap,
                    "reason": (
                        "calibration_gap_too_large"
                        if has_large_gap
                        else "calibration_samples_insufficient"
                    ),
                }
            )
            cycle_start = cycle_end
            continue

        identity = f"{GATEWAY}|{cycle_start.isoformat()}|{selected_policy.period_hours}"
        calibration_id = (
            f"cal-{cycle_start.strftime('%Y%m%dT%H%M%S')}-"
            f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:8]}"
        )
        source_ids = (
            tuple(sorted(set(usable["import_id"].dropna().astype(str))))
            if "import_id" in usable
            else ()
        )
        snapshot = CalibrationSnapshot(
            calibration_id=calibration_id,
            gateway=(
                str(usable["gateway"].iloc[0]) if "gateway" in usable else GATEWAY
            ),
            cycle_started_at=cycle_start,
            calibration_ended_at=calibration_end,
            next_calibration_at=cycle_end,
            sample_count=len(usable),
            tilt1_inline_baseline=float(usable["tilt1_inline_model"].median()),
            tilt1_crossline_baseline=float(usable["tilt1_crossline_model"].median()),
            temperature_baseline=float(usable["temperature_celsius"].median()),
            tilt2_crossline_baseline=float(usable["tilt2_crossline"].median()),
            source_import_ids=source_ids,
        )
        snapshots.append(snapshot)
        for name, value in snapshot.to_record().items():
            if name in ordered.columns and name != "source_import_ids":
                ordered.loc[cycle_mask, name] = value
        ordered.loc[calibration_mask, "calibration_status"] = "calibrating"
        ordered.loc[prediction_mask, "calibration_status"] = "ready"
        ordered.loc[cycle_mask, "tilt1_inline_baseline"] = snapshot.tilt1_inline_baseline
        ordered.loc[cycle_mask, "tilt1_crossline_baseline"] = snapshot.tilt1_crossline_baseline
        ordered.loc[cycle_mask, "temperature_baseline"] = snapshot.temperature_baseline
        ordered.loc[cycle_mask, "tilt2_crossline_baseline"] = snapshot.tilt2_crossline_baseline
        ordered.loc[cycle_mask, "tilt1_inline_delta"] = (
            ordered.loc[cycle_mask, "tilt1_inline_model"] - snapshot.tilt1_inline_baseline
        )
        ordered.loc[cycle_mask, "tilt1_crossline_delta"] = (
            ordered.loc[cycle_mask, "tilt1_crossline_model"]
            - snapshot.tilt1_crossline_baseline
        )
        ordered.loc[cycle_mask, "temperature_delta"] = (
            ordered.loc[cycle_mask, "temperature_celsius"] - snapshot.temperature_baseline
        )
        cycle_index = ordered.index[cycle_mask]
        cycle_temperature = ordered.loc[cycle_index, "temperature_celsius"].mask(
            quality_invalid.loc[cycle_index]
        )
        cycle_time = ordered.loc[cycle_index, "timestamp"]
        segment = (
            ordered.loc[cycle_index, "segment_id"]
            if "segment_id" in ordered
            else pd.Series(0, index=cycle_index)
        )
        same_10m = cycle_time.diff().eq(pd.Timedelta(minutes=10)) & segment.eq(
            segment.shift()
        )
        change_10m = cycle_temperature.diff().where(same_10m)
        one_hour_continuous = (
            cycle_time.sub(cycle_time.shift(6)).eq(pd.Timedelta(hours=1))
            & segment.eq(segment.shift(6))
            & cycle_temperature.notna().rolling(7, min_periods=7).sum().eq(7)
        )
        change_1h = cycle_temperature.diff(6).where(one_hour_continuous)
        rolling_continuous = (
            cycle_time.sub(cycle_time.shift(5)).eq(pd.Timedelta(minutes=50))
            & segment.eq(segment.shift(5))
            & cycle_temperature.notna().rolling(6, min_periods=6).sum().eq(6)
        )
        rolling_mean = cycle_temperature.rolling(6, min_periods=6).mean().where(
            rolling_continuous
        )
        ordered.loc[cycle_index, "temperature_change_10m"] = change_10m
        ordered.loc[cycle_index, "temperature_change_1h"] = change_1h
        ordered.loc[cycle_index, "temperature_rolling_mean_1h_delta"] = (
            rolling_mean - snapshot.temperature_baseline
        )
        ordered.loc[cycle_mask, "tilt2_crossline_delta"] = (
            ordered.loc[cycle_mask, "tilt2_crossline"]
            - snapshot.tilt2_crossline_baseline
        )
        cycle_start = cycle_end

    ready = ordered["calibration_status"].eq("ready")
    ordered.loc[ready & quality_invalid, "calibration_status"] = "input_unusable"
    ready = ordered["calibration_status"].eq("ready")
    finite_features = pd.Series(False, index=ordered.index)
    if ready.any():
        finite_features.loc[ready] = np.isfinite(
            ordered.loc[ready, list(CROSSLINE_FEATURE_COLUMNS)].to_numpy(dtype=float)
        ).all(axis=1)
    ordered.loc[ready & ~finite_features, "calibration_status"] = "feature_warmup"
    return CalibratedFrame(ordered, tuple(snapshots), tuple(invalid_cycles))


class CrosslineCalibrationService:
    """Persist and reconcile calibration snapshots behind one small interface."""

    COLUMNS = (
        "calibration_id",
        "gateway",
        "cycle_started_at",
        "calibration_ended_at",
        "next_calibration_at",
        "sample_count",
        "tilt1_inline_baseline",
        "tilt1_crossline_baseline",
        "temperature_baseline",
        "tilt2_crossline_baseline",
        "source_import_ids",
        "status",
    )

    def __init__(self, project_root: Path, policy: CalibrationPolicy | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.policy = policy or CalibrationPolicy()
        self.path = self.project_root / "data" / "database" / "crossline_calibrations.csv"
        self.lock_path = self.project_root / "data" / "database" / ".calibration.lock"

    def reconcile(self, measurements: pd.DataFrame) -> CalibratedFrame:
        """Rebuild persisted snapshots from the active measurement database."""

        if measurements.empty:
            calibrated = CalibratedFrame(measurements.copy(), (), ())
        else:
            calibrated = build_calibrated_frame(measurements, self.policy)
        records = pd.DataFrame(
            [snapshot.to_record() for snapshot in calibrated.snapshots],
            columns=self.COLUMNS,
        )
        with self._exclusive_lock():
            self._atomic_write(records)
        return calibrated

    def list_snapshots(self) -> tuple[CalibrationSnapshot, ...]:
        with self._exclusive_lock():
            if not self.path.is_file() or self.path.stat().st_size == 0:
                return ()
            frame = pd.read_csv(self.path, encoding="utf-8-sig", keep_default_na=False)
        missing = [column for column in self.COLUMNS if column not in frame]
        if missing:
            raise TemperaturePipelineError(f"校准记录缺少字段：{missing}")
        return tuple(
            CalibrationSnapshot.from_record(row)
            for row in frame.loc[:, self.COLUMNS].to_dict("records")
        )

    def current(self, at: pd.Timestamp | None = None) -> CalibrationSnapshot | None:
        moment = pd.Timestamp(at if at is not None else datetime.now(timezone.utc))
        if moment.tzinfo is not None:
            moment = moment.tz_localize(None)
        valid = [
            snapshot
            for snapshot in self.list_snapshots()
            if snapshot.status == "valid"
            and snapshot.calibration_ended_at <= moment < snapshot.next_calibration_at
        ]
        return max(valid, key=lambda item: item.cycle_started_at) if valid else None

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _atomic_write(self, frame: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="crossline-calibration-",
            suffix=".csv.tmp",
            dir=self.path.parent,
            encoding="utf-8-sig",
            newline="",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, self.path)
