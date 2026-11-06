"""Frozen, auditable cleaning for CSG124 model inputs.

The pipeline keeps bucket-level raw values separate from model values.  Only
sensor-1 inline/crossline and temperature are repaired; sensor-2 values remain
unmodified targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import SensorDatabaseError
from .temperature_quality import add_temperature_jump_flags, repair_enabled_temperature
from .cleaning_options import CleaningOptions, RULE_VERSION, SUPPORTED_RULE_VERSIONS
from .temperature_quality import _enabled_runs
from .temperature_repair import REPAIR_COLUMNS
from .temperature_repair_v3 import AUDIT_COLUMNS
from .temperature_cusum import (
    AUDIT_COLUMNS as CUSUM_AUDIT_COLUMNS, BOOL_COLUMNS as CUSUM_BOOL_COLUMNS,
    RULE_VERSIONS as CUSUM_VERSIONS,
)
from .temperature_daily_trend import RULE_VERSION as DAILY_VERSION, AUDIT_COLUMNS as DAILY_AUDIT_COLUMNS
from .processing import (
    COMPARISON_ATOL,
    MINIMUM_TILT_TOLERANCE_DEGREES,
    ProcessingVersion,
    current_processing_version,
    is_final_processing,
    final_processing_settings,
    round_final_measurement,
)


POLICY_SCHEMA_VERSION = 1
CLEANED_SCHEMA_VERSION = 1
DATASET_VERSION = "csg124_cleaned_database_v2"
REFERENCE_START = pd.Timestamp("2000-01-01 00:00:00")
REFERENCE_END = pd.Timestamp("2000-01-06 23:59:59")
BUCKET_FREQUENCY = "10min"
SEGMENT_GAP_MINUTES = 20
HAMPEL_WINDOW_SIZE = 61
HAMPEL_MIN_PERIODS = 30
HAMPEL_N_SIGMAS = 3.5
MAD_TO_SIGMA = 1.4826
QUANTILE = 0.99
MULTIPLIER = 3.0
MAX_INTERPOLATED_RUN = 2
MINIMUM_LIMIT = 1e-6


@dataclass(frozen=True)
class FieldSpec:
    """One raw input and its public audit-column prefix."""

    value_column: str
    raw_column: str
    prefix: str
    minimum: float
    maximum: float
    label: str


FIELD_SPECS = (
    FieldSpec(
        "tilt1_inline_model",
        "tilt1_inline_raw",
        "tilt1_inline",
        -10.0,
        10.0,
        "1号顺线",
    ),
    FieldSpec(
        "tilt1_crossline_model",
        "tilt1_crossline_raw",
        "tilt1_crossline",
        -10.0,
        10.0,
        "1号横线",
    ),
    FieldSpec(
        "temperature_celsius",
        "temperature_raw_celsius",
        "temperature",
        -40.0,
        50.0,
        "温度",
    ),
)
TARGET_COLUMNS = ("tilt2_inline", "tilt2_crossline")


@dataclass(frozen=True)
class CleaningPolicy:
    """Frozen thresholds and their provenance."""

    payload: dict[str, Any]

    @property
    def sha256(self) -> str:
        serialized = json.dumps(
            self.payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def threshold(self, group: str, value_column: str) -> float:
        try:
            value = float(self.payload[group][value_column])
        except (KeyError, TypeError, ValueError) as error:
            raise SensorDatabaseError(
                f"清洗策略缺少 {group}.{value_column}。"
            ) from error
        if not np.isfinite(value) or value <= 0:
            raise SensorDatabaseError(
                f"清洗策略 {group}.{value_column} 必须是正数。"
            )
        return value


@dataclass(frozen=True)
class FieldCleaningSummary:
    """Counts shown in import and database UIs."""

    field: str
    label: str
    physical_outliers: int
    bucket_conflicts: int
    hampel_outliers: int
    outliers: int
    imputed: int
    unresolved: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "field": self.field,
            "label": self.label,
            "physical_outliers": self.physical_outliers,
            "bucket_conflicts": self.bucket_conflicts,
            "hampel_outliers": self.hampel_outliers,
            "outliers": self.outliers,
            "imputed": self.imputed,
            "unresolved": self.unresolved,
        }


@dataclass(frozen=True)
class CleaningResult:
    """Materialized cleaned table plus compact and row-level audit data."""

    frame: pd.DataFrame
    field_summary: tuple[FieldCleaningSummary, ...]
    affected_rows: pd.DataFrame
    policy_sha256: str

    @property
    def bucket_rows(self) -> int:
        if self.frame.empty:
            return 0
        return int(self.frame["source_row_count"].gt(0).sum())

    @property
    def usable_rows(self) -> int:
        return int(self.frame["model_sample_usable"].sum())

    @property
    def imputed_rows(self) -> int:
        return int(self.frame["model_input_imputed"].sum())

    @property
    def unresolved_rows(self) -> int:
        return int(self.frame["model_input_unresolved"].sum())


class SensorCleaningPipeline:
    """Apply one reviewed policy to every model-input path."""

    def __init__(
        self, policy: CleaningPolicy, *, processing_version: ProcessingVersion | None = None
    ) -> None:
        self.policy = self._validate_policy(policy)
        self.processing_version = processing_version or current_processing_version()
        if self.processing_version not in ("legacy", "final_v1", "final_v2", "final_v3"):
            raise SensorDatabaseError("不支持的数据处理版本。")

    @property
    def processing_sha256(self) -> str:
        """Bind derived results to their base policy and processing revision."""

        if self.processing_version == "legacy":
            return self.policy.sha256
        payload = {
            "base_policy_sha256": self.policy.sha256,
            "processing": final_processing_settings(self.processing_version),
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    @classmethod
    def for_project(cls, project_root: Path) -> "SensorCleaningPipeline":
        path = Path(project_root) / "configs" / "cleaning_policy.json"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as error:
                raise SensorDatabaseError(f"无法读取清洗策略：{error}") from error
            return cls(CleaningPolicy(payload))
        return cls(cls.default_policy())

    @staticmethod
    def default_policy() -> CleaningPolicy:
        """Return a deterministic bootstrap policy for empty/test databases."""

        return CleaningPolicy(
            {
                "schema_version": POLICY_SCHEMA_VERSION,
                "gateway": "CSG124",
                "dataset_version": DATASET_VERSION,
                "cleaning_mode": "offline_centered_hampel",
                "reference_period": {
                    "start": REFERENCE_START.isoformat(),
                    "end": REFERENCE_END.isoformat(),
                },
                "reference_source_sha256": "bootstrap-no-reference",
                "bucket_conflict_limits": {
                    spec.value_column: MINIMUM_LIMIT for spec in FIELD_SPECS
                },
                "noise_floors": {
                    spec.value_column: MINIMUM_LIMIT for spec in FIELD_SPECS
                },
                "settings": SensorCleaningPipeline.settings(),
            }
        )

    @staticmethod
    def settings() -> dict[str, Any]:
        return {
            "bucket_frequency": BUCKET_FREQUENCY,
            "segment_gap_minutes": SEGMENT_GAP_MINUTES,
            "hampel_window_size": HAMPEL_WINDOW_SIZE,
            "hampel_min_periods": HAMPEL_MIN_PERIODS,
            "hampel_n_sigmas": HAMPEL_N_SIGMAS,
            "mad_to_sigma": MAD_TO_SIGMA,
            "threshold_quantile": QUANTILE,
            "threshold_multiplier": MULTIPLIER,
            "max_interpolated_outlier_run": MAX_INTERPOLATED_RUN,
            "physical_ranges": {
                spec.value_column: [spec.minimum, spec.maximum]
                for spec in FIELD_SPECS
            },
        }

    @classmethod
    def fit_policy(cls, raw_frame: pd.DataFrame) -> CleaningPolicy:
        """Fit thresholds once from the example reference period."""

        prepared = cls._prepare_raw(raw_frame)
        reference = prepared.loc[
            prepared["bucket_timestamp"].between(REFERENCE_START, REFERENCE_END)
        ].copy()
        if reference.empty:
            raise SensorDatabaseError("当前数据没有覆盖 2000-01-01 至 2000-01-06 基准期。")

        conflict_limits: dict[str, float] = {}
        noise_floors: dict[str, float] = {}
        for spec in FIELD_SPECS:
            valid = reference.loc[
                reference[spec.value_column].between(spec.minimum, spec.maximum),
                ["bucket_timestamp", spec.value_column],
            ]
            stats = valid.groupby("bucket_timestamp")[spec.value_column].agg(
                ["count", "min", "max", "median"]
            )
            spreads = (stats["max"] - stats["min"])[stats["count"].ge(2)].dropna()
            conflict_limits[spec.value_column] = max(
                float(spreads.quantile(QUANTILE) * MULTIPLIER)
                if not spreads.empty
                else MINIMUM_LIMIT,
                MINIMUM_LIMIT,
            )
            changes = stats["median"].sort_index().diff().abs().dropna()
            noise_floors[spec.value_column] = max(
                float(changes.quantile(QUANTILE) * MULTIPLIER)
                if not changes.empty
                else MINIMUM_LIMIT,
                MINIMUM_LIMIT,
            )

        canonical = reference.loc[
            :, ["timestamp", *(spec.value_column for spec in FIELD_SPECS)]
        ].to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode("utf-8")
        payload = {
            "schema_version": POLICY_SCHEMA_VERSION,
            "gateway": "CSG124",
            "dataset_version": DATASET_VERSION,
            "cleaning_mode": "offline_centered_hampel",
            "reference_period": {
                "start": REFERENCE_START.isoformat(),
                "end": REFERENCE_END.isoformat(),
            },
            "reference_source_sha256": hashlib.sha256(canonical).hexdigest(),
            "reference_row_count": len(reference),
            "bucket_conflict_limits": conflict_limits,
            "noise_floors": noise_floors,
            "settings": cls.settings(),
        }
        return CleaningPolicy(payload)

    def clean(self, raw_frame: pd.DataFrame) -> CleaningResult:
        """Create the complete ten-minute cleaned/model table."""

        if raw_frame.empty:
            return self._empty_result()
        prepared = self._prepare_raw(raw_frame)
        bucketed = self._aggregate(prepared)
        gap_minutes = bucketed["timestamp"].diff().dt.total_seconds().div(60)
        bucketed["segment_id"] = (gap_minutes.isna() | gap_minutes.gt(20)).cumsum()

        if self.processing_version in ("final_v2", "final_v3"):
            from .zero_temperature import prepare_zero_temperature

            prepare_zero_temperature(bucketed)

        for spec in FIELD_SPECS:
            self._add_hampel(bucketed, spec)
        add_temperature_jump_flags(bucketed)

        completed_parts: list[pd.DataFrame] = []
        for segment_id, segment in bucketed.groupby("segment_id", sort=False):
            full_index = pd.date_range(
                segment["timestamp"].min(),
                segment["timestamp"].max(),
                freq=BUCKET_FREQUENCY,
                name="timestamp",
            )
            completed = segment.set_index("timestamp").reindex(full_index)
            completed["segment_id"] = int(segment_id)
            completed["gateway"] = "CSG124"
            completed["source_row_count"] = completed["source_row_count"].fillna(0).astype(int)
            completed["is_inserted_gap"] = completed["source_row_count"].eq(0)
            for spec in FIELD_SPECS:
                self._repair_field(completed, spec)
            completed["temperature_rule_version"] = completed["temperature_rule_version"].fillna(RULE_VERSION)
            repair_enabled_temperature(completed)
            self._finalize_temperature_repairs(completed)
            self._finalize_temperature_v3(completed)
            # 开启新规则的桶保守使用：任一模型输入硬异常都不能被旧插值逻辑放行。
            # 关闭桶完全保留旧行为；原始列不被覆盖。
            enabled = completed["temperature_jump_check_enabled"]
            for spec in FIELD_SPECS:
                hard = enabled & (
                    completed[f"{spec.prefix}_physical_outlier"]
                    | completed[f"{spec.prefix}_bucket_conflict"]
                )
                completed.loc[hard, spec.value_column] = np.nan
                completed.loc[hard, f"{spec.prefix}_imputed"] = False
                completed.loc[hard, f"{spec.prefix}_unresolved"] = True
            completed_parts.append(completed.reset_index())

        cleaned = pd.concat(completed_parts, ignore_index=True)
        if is_final_processing(self.processing_version):
            from .short_gaps import finalize_short_gaps

            cleaned = finalize_short_gaps(cleaned)
            if self.processing_version in ("final_v2", "final_v3"):
                from .zero_temperature import finalize_zero_temperature

                cleaned = finalize_zero_temperature(cleaned)
                from .target_gaps import interpolate_target_gaps

                cleaned = interpolate_target_gaps(cleaned)
                if self.processing_version == "final_v3":
                    from .whole_gaps import interpolate_whole_row_gaps

                    cleaned = interpolate_whole_row_gaps(cleaned)
            # Quantize actual outputs once, after all repairs; raw audit values
            # and the anomaly decisions above retain their original precision.
            for column in (*[spec.value_column for spec in FIELD_SPECS], *TARGET_COLUMNS):
                cleaned[column] = cleaned[column].map(round_final_measurement, na_action="ignore")
        cleaned["model_input_imputed"] = cleaned[
            [f"{spec.prefix}_imputed" for spec in FIELD_SPECS]
        ].any(axis=1)
        cleaned["model_input_unresolved"] = cleaned[
            [f"{spec.prefix}_unresolved" for spec in FIELD_SPECS]
        ].any(axis=1)
        cleaned["input_outlier"] = cleaned[
            [f"{spec.prefix}_outlier" for spec in FIELD_SPECS]
        ].any(axis=1)
        target_imputed = (cleaned[["tilt2_inline_gap_imputed", "tilt2_crossline_gap_imputed"]].any(axis=1)
                          if self.processing_version in ("final_v2", "final_v3")
                          else pd.Series(False, index=cleaned.index))
        cleaned["quality_status"] = np.select(
            [
                cleaned["model_input_unresolved"],
                cleaned["temperature_review_required"],
                cleaned["temperature_offset_corrected"],
                cleaned["model_input_imputed"] | target_imputed,
                cleaned["is_inserted_gap"],
            ],
            ["unresolved", "review_required", "corrected", "imputed", "inserted_gap"],
            default="clean",
        )
        required_model = [spec.value_column for spec in FIELD_SPECS]
        cleaned["model_input_available"] = (
            cleaned[required_model].notna().all(axis=1) & ~cleaned["temperature_review_required"]
        )
        cleaned["model_sample_usable"] = (
            cleaned["model_input_available"]
            & cleaned[list(TARGET_COLUMNS)].notna().all(axis=1)
            & ~cleaned["is_inserted_gap"]
        )
        if self.processing_version in ("final_v2", "final_v3"):
            cleaned["model_sample_usable"] &= ~cleaned[[
                "tilt2_inline_gap_imputed", "tilt2_crossline_gap_imputed"
            ]].any(axis=1)
        cleaned.insert(0, "cleaned_schema_version", CLEANED_SCHEMA_VERSION)
        cleaned["cleaning_policy_sha256"] = self.processing_sha256
        if is_final_processing(self.processing_version):
            cleaned["base_cleaning_policy_sha256"] = self.policy.sha256
            cleaned["processing_version"] = self.processing_version
        # 基础冻结策略不变；额外配置单独签名，让训练数据快照可追溯开关选择。
        profiles = {
            (enabled, version): CleaningOptions(enabled, self.policy.sha256, version).sha256
            for enabled in (False, True) for version in SUPPORTED_RULE_VERSIONS
        }
        if is_final_processing(self.processing_version):
            profiles = {
                key: hashlib.sha256(
                    f"{value}:{self.processing_sha256}".encode("utf-8")
                ).hexdigest()
                for key, value in profiles.items()
            }
        cleaned["cleaning_profile_sha256"] = [
            profiles[(enabled, version)] for enabled, version in zip(
                cleaned["temperature_jump_check_enabled"], cleaned["temperature_rule_version"], strict=True
            )
        ]
        cleaned = cleaned.drop(columns=["_temperature_candidate_celsius"])
        cleaned = cleaned.sort_values("timestamp", kind="stable").reset_index(drop=True)
        summary = self._summaries(cleaned)
        affected = self._affected_rows(cleaned)
        return CleaningResult(cleaned, summary, affected, self.processing_sha256)

    @staticmethod
    def _prepare_raw(raw_frame: pd.DataFrame) -> pd.DataFrame:
        required = ["timestamp", *(spec.value_column for spec in FIELD_SPECS), *TARGET_COLUMNS]
        missing = [column for column in required if column not in raw_frame]
        if missing:
            raise SensorDatabaseError(f"清洗输入缺少字段：{missing}")
        prepared = raw_frame.copy()
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        if prepared["timestamp"].isna().any():
            raise SensorDatabaseError("清洗输入包含无效时间。")
        prepared["bucket_timestamp"] = prepared["timestamp"].dt.round(BUCKET_FREQUENCY)
        for column in [*(spec.value_column for spec in FIELD_SPECS), *TARGET_COLUMNS]:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        return prepared.sort_values("timestamp", kind="stable").reset_index(drop=True)

    def _aggregate(self, prepared: pd.DataFrame) -> pd.DataFrame:
        aggregations: dict[str, tuple[str, str]] = {
            "timestamp": ("bucket_timestamp", "first"),
            "observation_timestamp": ("timestamp", "median"),
            "source_row_count": ("timestamp", "size"),
            "tilt2_inline": ("tilt2_inline", "median"),
            "tilt2_crossline": ("tilt2_crossline", "median"),
        }
        for spec in FIELD_SPECS:
            aggregations.update(
                {
                    spec.raw_column: (spec.value_column, "median"),
                    f"{spec.prefix}_source_count": (spec.value_column, "count"),
                    f"{spec.prefix}_bucket_min": (spec.value_column, "min"),
                    f"{spec.prefix}_bucket_max": (spec.value_column, "max"),
                }
            )
        for provenance in (
            "source_filename",
            "source_sheet",
            "source_sha256",
            "import_id",
            "imported_at_utc",
        ):
            if provenance in prepared:
                aggregations[provenance] = (provenance, "last")
        bucketed = (
            prepared.groupby("bucket_timestamp", as_index=False, sort=True)
            .agg(**aggregations)
            .drop(columns=["bucket_timestamp"], errors="ignore")
        )
        # 桶内可包含多个批次：由入库时间最新者决定开关，而不是由观测时间最后者决定。
        options = prepared.copy()
        modes = options.get("temperature_jump_check_enabled", pd.Series(False, index=options.index))
        if not modes.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise SensorDatabaseError("温度突变检查设置必须为布尔值。")
        options["temperature_jump_check_enabled"] = modes.astype(bool)
        versions = options.get("temperature_rule_version", pd.Series(RULE_VERSION, index=options.index))
        if not versions.isin(SUPPORTED_RULE_VERSIONS).all():
            raise SensorDatabaseError("不支持的温度规则版本。")
        options["temperature_rule_version"] = versions
        options["_batch_time"] = pd.to_datetime(
            options.get("imported_at_utc", pd.Series(pd.NaT, index=options.index)), utc=True
        )
        options["_batch_id"] = options.get("import_id", pd.Series("", index=options.index)).fillna("").astype(str)
        newest = options.sort_values(
            ["_batch_time", "_batch_id", "timestamp"], kind="stable", na_position="first"
        ).drop_duplicates("bucket_timestamp", keep="last").set_index("bucket_timestamp")
        bucketed["temperature_jump_check_enabled"] = bucketed["timestamp"].map(
            newest["temperature_jump_check_enabled"]
        ).astype(bool)
        bucketed["temperature_rule_version"] = bucketed["timestamp"].map(newest["temperature_rule_version"])
        for spec in FIELD_SPECS:
            prefix = spec.prefix
            bucketed[f"{prefix}_bucket_spread"] = (
                bucketed[f"{prefix}_bucket_max"] - bucketed[f"{prefix}_bucket_min"]
            )
            bucketed[f"{prefix}_physical_outlier"] = (
                bucketed[f"{prefix}_bucket_min"].lt(spec.minimum)
                | bucketed[f"{prefix}_bucket_max"].gt(spec.maximum)
            )
            bucketed[f"{prefix}_bucket_conflict"] = (
                bucketed[f"{prefix}_source_count"].ge(2)
                & bucketed[f"{prefix}_bucket_spread"].gt(
                    self.policy.threshold("bucket_conflict_limits", spec.value_column)
                )
            )
        return bucketed.sort_values("timestamp", kind="stable").reset_index(drop=True)

    def _add_hampel(self, bucketed: pd.DataFrame, spec: FieldSpec) -> None:
        prefix = spec.prefix
        values = bucketed[spec.raw_column].to_numpy(dtype=float)
        physical = bucketed[f"{prefix}_physical_outlier"].to_numpy(dtype=bool)
        conflict = bucketed[f"{prefix}_bucket_conflict"].to_numpy(dtype=bool)
        scores = np.full(len(bucketed), np.nan)
        medians = np.full(len(bucketed), np.nan)
        mads = np.full(len(bucketed), np.nan)
        counts = np.zeros(len(bucketed), dtype=int)
        limits = np.full(len(bucketed), np.nan)
        noise_floor = self.policy.threshold("noise_floors", spec.value_column)
        derived_tilt = is_final_processing(self.processing_version) and spec.prefix != "temperature"
        if derived_tilt:
            noise_floor = max(noise_floor, MINIMUM_TILT_TOLERANCE_DEGREES)

        for _, segment in bucketed.groupby("segment_id", sort=False):
            indexes = segment.index.to_numpy()
            segment_values = values[indexes]
            references = segment_values.copy()
            references[physical[indexes] | conflict[indexes]] = np.nan
            window_size = min(HAMPEL_WINDOW_SIZE, len(indexes))
            half = HAMPEL_WINDOW_SIZE // 2
            for local_position, global_position in enumerate(indexes):
                start = max(0, local_position - half)
                end = min(len(indexes), local_position + half + 1)
                if end - start < window_size:
                    if start == 0:
                        end = window_size
                    elif end == len(indexes):
                        start = len(indexes) - window_size
                reliable = references[start:end]
                reliable = reliable[np.isfinite(reliable)]
                counts[global_position] = len(reliable)
                if len(reliable) < HAMPEL_MIN_PERIODS:
                    continue
                median = float(np.median(reliable))
                mad = float(np.median(np.abs(reliable - median)))
                limit = max(HAMPEL_N_SIGMAS * MAD_TO_SIGMA * mad, noise_floor)
                medians[global_position] = median
                mads[global_position] = mad
                limits[global_position] = limit
                if np.isfinite(values[global_position]):
                    scores[global_position] = abs(values[global_position] - median) / limit

        bucketed[f"{prefix}_hampel_reference_count"] = counts
        bucketed[f"{prefix}_hampel_local_median"] = medians
        bucketed[f"{prefix}_hampel_local_mad"] = mads
        bucketed[f"{prefix}_hampel_effective_limit"] = limits
        bucketed[f"{prefix}_hampel_score"] = scores
        outliers = scores > 1.0
        if derived_tilt:
            outliers &= np.abs(values - medians) > limits + COMPARISON_ATOL
        bucketed[f"{prefix}_hampel_outlier"] = outliers
        bucketed[f"{prefix}_outlier"] = (
            bucketed[f"{prefix}_physical_outlier"]
            | bucketed[f"{prefix}_bucket_conflict"]
            | bucketed[f"{prefix}_hampel_outlier"]
        )

    def _finalize_temperature_repairs(self, completed: pd.DataFrame) -> None:
        """用冻结策略复查v2候选；整事件失败时不保留部分修复。"""
        for column in ("temperature_repair_method", "temperature_event_id"):
            completed[column] = completed[column].fillna("none" if column.endswith("method") else "")
        completed["temperature_offset_celsius"] = completed["temperature_offset_celsius"].fillna(0.0)
        for column in (
            "temperature_offset_corrected", "temperature_repaired", "temperature_post_repair_hampel_outlier",
        ):
            completed[column] = completed[column].astype("boolean").fillna(False).astype(bool)
        runs = _enabled_runs(completed)
        enabled = completed["temperature_jump_check_enabled"] & completed["temperature_rule_version"].eq(
            "temperature_jump_v2"
        )
        spec = next(item for item in FIELD_SPECS if item.prefix == "temperature")
        for _, part in completed.loc[enabled].groupby(runs.loc[enabled], sort=False):
            part = part.copy()
            proposed = part["temperature_repair_method"].isin(("interpolation", "offset"))
            candidate = part["_temperature_candidate_celsius"]
            suspicious = part["temperature_jump_outlier"].astype(bool)
            verification = part["temperature_celsius"].mask(suspicious)
            verification.loc[proposed] = candidate.loc[proposed]
            # 拒绝一个事件后，它不应继续作为其他事件复查时的参考数据。
            # 每轮至少拒绝一个事件，否则退出，因此循环有界且结果可重复。
            while proposed.any():
                scratch = part.reset_index(drop=True).copy()
                scratch[spec.raw_column] = verification.to_numpy()
                self._add_hampel(scratch, spec)
                post_hampel = pd.Series(scratch["temperature_hampel_outlier"].to_numpy(), index=part.index)
                invalid = post_hampel | ~candidate.between(spec.minimum, spec.maximum)
                part["temperature_post_repair_hampel_outlier"] |= post_hampel
                rejected = False
                for event_id in part.loc[proposed, "temperature_event_id"].unique():
                    event = part["temperature_event_id"].eq(event_id)
                    if invalid.loc[event].any():
                        proposed.loc[event] = False
                        verification.loc[event] = np.nan
                        part.loc[event, "temperature_repair_method"] = "unresolved"
                        part.loc[event, "temperature_offset_celsius"] = 0.0
                        part.loc[event, "temperature_jump_isolated"] = True
                        rejected = True
                if not rejected:
                    break
            part.loc[suspicious, "temperature_celsius"] = np.nan
            part.loc[suspicious, "temperature_imputed"] = False
            part.loc[proposed, "temperature_celsius"] = candidate.loc[proposed]
            part.loc[proposed, "temperature_imputed"] = part.loc[
                proposed, "temperature_repair_method"
            ].eq("interpolation")
            part["temperature_offset_corrected"] = proposed & part["temperature_repair_method"].eq("offset")
            part["temperature_repaired"] = proposed | part["temperature_imputed"]
            unresolved = suspicious & ~proposed
            part.loc[unresolved, "temperature_repair_method"] = "unresolved"
            part["temperature_outlier"] = part["temperature_outlier"] | suspicious
            part["temperature_unresolved"] = part["temperature_outlier"] & part["temperature_celsius"].isna()
            groups = part["temperature_outlier"].ne(part["temperature_outlier"].shift(fill_value=False)).cumsum()
            part["temperature_outlier_run_length"] = part["temperature_outlier"].groupby(groups).transform(
                "sum"
            ).where(part["temperature_outlier"], 0).astype(int)
            for mask, reason in (
                (suspicious, "temperature_jump"),
                (part["temperature_offset_corrected"], "temperature_offset_corrected"),
                (unresolved, "temperature_jump_isolated"),
            ):
                part.loc[mask, "temperature_reason"] = part.loc[mask, "temperature_reason"].map(
                    lambda value: f"{value}|{reason}" if value else reason
                )
            for column in (
                *REPAIR_COLUMNS, "temperature_celsius", "temperature_imputed", "temperature_outlier",
                "temperature_unresolved", "temperature_outlier_run_length", "temperature_reason",
                "temperature_jump_isolated",
            ):
                completed.loc[part.index, column] = part[column]
        # 旧版和基础Hampel的插值也计入总修复，但不改变其原有质量判定。
        imputed = completed["temperature_imputed"] & ~completed["temperature_rule_version"].isin(
            ("temperature_jump_v3", *CUSUM_VERSIONS, DAILY_VERSION))
        completed.loc[imputed, "temperature_repaired"] = True
        completed.loc[imputed, "temperature_repair_method"] = "interpolation"

    @staticmethod
    def _finalize_temperature_v3(completed: pd.DataFrame) -> None:
        """v3数值只来自新流程；旧Hampel标记仅作诊断，待判断原值禁止建模。"""
        completed["temperature_review_required"] = completed["temperature_review_required"].astype(
            "boolean"
        ).fillna(False).astype(bool)
        completed["temperature_review_reason"] = completed["temperature_review_reason"].fillna("")
        enabled = completed["temperature_jump_check_enabled"] & completed["temperature_rule_version"].isin(
            ("temperature_jump_v3", *CUSUM_VERSIONS, DAILY_VERSION))
        for column in CUSUM_BOOL_COLUMNS:
            completed[column] = completed[column].astype("boolean").fillna(False).astype(bool)
        completed["temperature_cusum_status"] = completed["temperature_cusum_status"].fillna("not_observed")
        completed["temperature_cusum_event_id"] = completed["temperature_cusum_event_id"].fillna("")
        completed["temperature_cusum_reference_count"] = completed["temperature_cusum_reference_count"].fillna(0).astype(int)
        for column in ("temperature_daily_checked", "temperature_daily_corrected"):
            completed[column] = completed[column].astype("boolean").fillna(False).astype(bool)
        completed["temperature_daily_status"] = completed["temperature_daily_status"].fillna("not_observed")
        runs = _enabled_runs(completed)
        for _, part in completed.loc[enabled].groupby(runs.loc[enabled], sort=False):
            part = part.copy()
            hard = part["temperature_physical_outlier"] | part["temperature_bucket_conflict"]
            part["temperature_celsius"] = part["_temperature_candidate_celsius"].mask(hard)
            part["temperature_imputed"] = part["temperature_repair_method"].isin(
                ("interpolation", "interpolation_offset")
            ) & ~hard
            part["temperature_offset_corrected"] = part["temperature_repair_method"].isin(
                ("offset", "interpolation_offset", "cusum_offset", "daily_trend")
            ) & ~hard
            part["temperature_repaired"] = part["temperature_imputed"] | part["temperature_offset_corrected"]
            part["temperature_unresolved"] = part["temperature_celsius"].isna()
            part["temperature_outlier"] = (part["temperature_unresolved"] | part["temperature_jump_outlier"]
                                           | part["temperature_cusum_outlier"] | part["temperature_daily_corrected"])
            groups = part["temperature_outlier"].ne(part["temperature_outlier"].shift(fill_value=False)).cumsum()
            part["temperature_outlier_run_length"] = part["temperature_outlier"].groupby(groups).transform(
                "sum"
            ).where(part["temperature_outlier"], 0).astype(int)
            reasons = pd.Series("", index=part.index, dtype=object)
            for mask, reason in (
                (part["temperature_physical_outlier"], "physical_range"),
                (part["temperature_bucket_conflict"], "bucket_conflict"),
                (part["temperature_jump_outlier"], "temperature_jump"),
                (part["temperature_cusum_outlier"], "temperature_cusum"),
                (part["temperature_daily_corrected"], "temperature_daily_trend"),
                (part["temperature_imputed"], "temperature_interpolated"),
                (part["temperature_offset_corrected"], "temperature_offset_corrected"),
                (part["temperature_review_required"], "review_required"),
            ):
                reasons.loc[mask] = reasons.loc[mask].map(lambda value: f"{value}|{reason}" if value else reason)
            part["temperature_reason"] = reasons
            for column in (
                "temperature_celsius", "temperature_imputed", "temperature_offset_corrected",
                "temperature_repaired", "temperature_unresolved", "temperature_outlier",
                "temperature_outlier_run_length", "temperature_reason",
            ):
                completed.loc[part.index, column] = part[column]

    @staticmethod
    def _repair_field(completed: pd.DataFrame, spec: FieldSpec) -> None:
        prefix = spec.prefix
        for suffix in (
            "physical_outlier",
            "bucket_conflict",
            "hampel_outlier",
            "outlier",
        ):
            column = f"{prefix}_{suffix}"
            completed[column] = completed[column].astype("boolean").fillna(False).astype(bool)
        # Inserted time buckets are audited separately.  They are not counted as
        # a physical/Hampel anomaly for every sensor field and are never model
        # samples because no sensor-2 truth exists for them.
        anomaly = completed[f"{prefix}_outlier"]
        run_id = anomaly.ne(anomaly.shift(fill_value=False)).cumsum()
        run_length = anomaly.groupby(run_id).transform("sum").where(anomaly, 0).astype(int)
        completed[f"{prefix}_outlier_run_length"] = run_length
        completed[spec.value_column] = completed[spec.raw_column].mask(anomaly)
        completed[f"{prefix}_imputed"] = False

        for current_run in run_id[anomaly & run_length.le(MAX_INTERPOLATED_RUN)].unique():
            positions = np.flatnonzero((run_id == current_run).to_numpy())
            before = int(positions[0]) - 1
            after = int(positions[-1]) + 1
            if before < 0 or after >= len(completed):
                continue
            if anomaly.iloc[before] or anomaly.iloc[after]:
                continue
            left = completed[spec.raw_column].iloc[before]
            right = completed[spec.raw_column].iloc[after]
            if pd.isna(left) or pd.isna(right):
                continue
            left_time = completed.index[before]
            right_time = completed.index[after]
            duration = right_time - left_time
            if duration == pd.Timedelta(0):
                continue
            for position in positions:
                fraction = (completed.index[position] - left_time) / duration
                completed.iloc[
                    position, completed.columns.get_loc(spec.value_column)
                ] = left + (right - left) * fraction
                completed.iloc[
                    position, completed.columns.get_loc(f"{prefix}_imputed")
                ] = True

        completed[f"{prefix}_unresolved"] = anomaly & completed[spec.value_column].isna()
        reasons = pd.Series("", index=completed.index, dtype="object")
        reason_flags = (
            (completed[f"{prefix}_physical_outlier"], "physical_range"),
            (completed[f"{prefix}_bucket_conflict"], "bucket_conflict"),
            (completed[f"{prefix}_hampel_outlier"], "hampel"),
        )
        for mask, label in reason_flags:
            reasons.loc[mask] = reasons.loc[mask].map(
                lambda value: f"{value}|{label}" if value else label
            )
        completed[f"{prefix}_reason"] = reasons

    @staticmethod
    def _summaries(cleaned: pd.DataFrame) -> tuple[FieldCleaningSummary, ...]:
        return tuple(
            FieldCleaningSummary(
                field=spec.prefix,
                label=spec.label,
                physical_outliers=int(cleaned[f"{spec.prefix}_physical_outlier"].sum()),
                bucket_conflicts=int(cleaned[f"{spec.prefix}_bucket_conflict"].sum()),
                hampel_outliers=int(cleaned[f"{spec.prefix}_hampel_outlier"].sum()),
                outliers=int(cleaned[f"{spec.prefix}_outlier"].sum()),
                imputed=int(cleaned[f"{spec.prefix}_imputed"].sum()),
                unresolved=int(cleaned[f"{spec.prefix}_unresolved"].sum()),
            )
            for spec in FIELD_SPECS
        )

    @staticmethod
    def _affected_rows(cleaned: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for spec in FIELD_SPECS:
            mask = (
                cleaned[f"{spec.prefix}_outlier"]
                | cleaned[f"{spec.prefix}_imputed"]
                | cleaned[f"{spec.prefix}_unresolved"]
                | cleaned["is_inserted_gap"]
            )
            for _, row in cleaned.loc[mask].iterrows():
                rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "field": spec.prefix,
                        "field_label": spec.label,
                        "raw_value": row.get(spec.raw_column),
                        "cleaned_value": row.get(spec.value_column),
                        "hampel_score": row.get(f"{spec.prefix}_hampel_score"),
                        "reason": (
                            "inserted_gap"
                            if row.get("is_inserted_gap", False)
                            else row.get(f"{spec.prefix}_reason", "")
                        ),
                        "status": (
                            "imputed"
                            if row.get(f"{spec.prefix}_gap_imputed", False)
                            else "inserted_gap"
                            if row.get("is_inserted_gap", False)
                            else "unresolved"
                            if row.get(f"{spec.prefix}_unresolved", False)
                            else "review_required"
                            if spec.prefix == "temperature" and row.get("temperature_review_required", False)
                            else "imputed"
                            if row.get(f"{spec.prefix}_imputed", False)
                            else "corrected"
                            if spec.prefix == "temperature" and row.get("temperature_offset_corrected", False)
                            else "outlier"
                        ),
                        **({column: row.get(column) for column in (
                            "temperature_event_id", "temperature_repair_method", "temperature_offset_celsius",
                            *AUDIT_COLUMNS, *CUSUM_AUDIT_COLUMNS, *DAILY_AUDIT_COLUMNS,
                        )} if spec.prefix == "temperature" else {}),
                    }
                )
        # Preserve provenance of estimated target values in the row-level audit.
        if "whole_row_gap_imputed" in cleaned:
            for target, label in (("tilt2_inline", "2号顺线"), ("tilt2_crossline", "2号横线")):
                mask = cleaned["whole_row_gap_imputed"].fillna(False) & cleaned[f"{target}_gap_imputed"].fillna(False)
                for _, row in cleaned.loc[mask].iterrows():
                    rows.append({"timestamp": row["timestamp"], "field": target, "field_label": label,
                                 "raw_value": row.get(f"{target}_observed"), "cleaned_value": row.get(target),
                                 "hampel_score": np.nan, "reason": "whole_row_gap_interpolation",
                                 "status": "imputed"})
        columns = (
            "timestamp",
            "field",
            "field_label",
            "raw_value",
            "cleaned_value",
            "hampel_score",
            "reason",
            "status",
            "temperature_event_id", "temperature_repair_method", "temperature_offset_celsius", *AUDIT_COLUMNS,
            *CUSUM_AUDIT_COLUMNS, *DAILY_AUDIT_COLUMNS,
        )
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _validate_policy(policy: CleaningPolicy) -> CleaningPolicy:
        payload = policy.payload
        if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise SensorDatabaseError("清洗策略结构版本不兼容。")
        if payload.get("gateway") != "CSG124":
            raise SensorDatabaseError("清洗策略不属于 CSG124。")
        if payload.get("settings") != SensorCleaningPipeline.settings():
            raise SensorDatabaseError("清洗策略与当前算法设置不一致。")
        for spec in FIELD_SPECS:
            policy.threshold("bucket_conflict_limits", spec.value_column)
            policy.threshold("noise_floors", spec.value_column)
        return policy

    def _empty_result(self) -> CleaningResult:
        columns = [
            "cleaned_schema_version",
            "gateway",
            "timestamp",
            "observation_timestamp",
            "segment_id",
            "source_row_count",
            *(spec.raw_column for spec in FIELD_SPECS),
            *(spec.value_column for spec in FIELD_SPECS),
            *TARGET_COLUMNS,
            "is_inserted_gap",
            "model_input_imputed",
            "model_input_unresolved",
            "input_outlier",
            "quality_status",
            "model_input_available",
            "model_sample_usable",
            "cleaning_policy_sha256",
            "cleaning_profile_sha256",
            "temperature_jump_check_enabled",
            "temperature_rate_celsius_per_10min",
            "temperature_jump_trigger",
            "temperature_jump_outlier",
            "temperature_jump_isolated",
            "temperature_rule_version",
            *(column for column in REPAIR_COLUMNS if not column.startswith("_")),
            *AUDIT_COLUMNS,
            *CUSUM_AUDIT_COLUMNS, *DAILY_AUDIT_COLUMNS,
        ]
        summary = tuple(
            FieldCleaningSummary(spec.prefix, spec.label, 0, 0, 0, 0, 0, 0)
            for spec in FIELD_SPECS
        )
        if is_final_processing(self.processing_version):
            columns.extend([
                "base_cleaning_policy_sha256", "processing_version",
                *(f"{spec.prefix}_gap_imputed" for spec in FIELD_SPECS),
            ])
        if self.processing_version in ("final_v2", "final_v3"):
            from .zero_temperature import ZERO_COLUMNS

            columns.extend([*ZERO_COLUMNS, "processed_exclusion_reason", "temperature_zero_run_length"])
        if self.processing_version == "final_v3":
            columns.extend(["whole_row_gap_imputed", "whole_gap_reference_sufficient",
                            "tilt2_inline_observed", "tilt2_crossline_observed",
                            "tilt2_inline_gap_imputed", "tilt2_crossline_gap_imputed"])
        return CleaningResult(
            pd.DataFrame(columns=columns),
            summary,
            pd.DataFrame(
                columns=(
                    "timestamp",
                    "field",
                    "field_label",
                    "raw_value",
                    "cleaned_value",
                    "hampel_score",
                    "reason",
                    "status",
                    "temperature_event_id", "temperature_repair_method", "temperature_offset_celsius", *AUDIT_COLUMNS,
                )
            ),
            self.processing_sha256,
        )
