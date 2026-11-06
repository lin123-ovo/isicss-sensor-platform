"""Build a bounded inference request from the chronological CSV database."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .interactive_prediction import NormalizedUpload
from .crossline_compact36 import exogenous_history
from .temperature_pipeline import FEATURE_COLUMNS, GATEWAY, TARGET_COLUMNS, TemperaturePipelineError


DATABASE_PREDICTION_COLUMNS = (
    "timestamp",
    "source_row_count",
    *FEATURE_COLUMNS,
    *TARGET_COLUMNS,
)


def _prediction_view(measurements: pd.DataFrame, include_temperature_review: bool) -> pd.DataFrame:
    """Relax only the review gate on an independent inference/history copy."""
    view = measurements.copy()
    if not include_temperature_review:
        return view

    def flag(name: str) -> pd.Series:
        return view.get(name, pd.Series(False, index=view.index)).fillna(False).astype(bool)

    review = flag("temperature_review_required")
    original_temperature = pd.to_numeric(
        view.get("temperature_raw_celsius", view["temperature_celsius"]), errors="coerce",
    )
    view.loc[review, "temperature_celsius"] = original_temperature.loc[review]
    finite = np.isfinite(view.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    blocked = pd.Series(False, index=view.index)
    for name in (
        "processed_row_excluded", "history_barrier", "model_input_unresolved",
        "tilt1_inline_unresolved", "tilt1_crossline_unresolved", "temperature_unresolved",
        "temperature_physical_outlier", "temperature_bucket_conflict", "temperature_zero_invalid",
    ):
        blocked |= flag(name)
    included = review & finite & original_temperature.between(-40., 50.) & ~blocked
    # Keep this separate from the original quality flag: feature builders mask
    # review-required inputs, so only the transient history copy may relax it.
    view["temperature_review_included"] = included
    if "temperature_review_required" in view:
        view.loc[included, "temperature_review_required"] = False
    if "model_input_available" in view:
        view.loc[included, "model_input_available"] = True
    return view


def build_database_prediction_upload(
    measurements: pd.DataFrame,
    percentage: int,
    *,
    include_temperature_review: bool = False,
) -> NormalizedUpload:
    """Select the latest percentage of database rows for inference only.

    The selected rows remain chronological. Existing target values are retained
    solely for optional error evaluation and are never used to fit a model.
    Explicitly including temperature reviews uses retained original temperatures
    for inference and causal history only; hard errors and training remain excluded.
    The percentage applies after selecting eligible rows. Inputs are never mutated.
    """

    if isinstance(percentage, bool) or not isinstance(percentage, (int, np.integer)):
        raise TemperaturePipelineError("预测比例必须是 1 到 100 的整数。")
    if percentage < 1 or percentage > 100:
        raise TemperaturePipelineError("预测比例必须在 1% 到 100% 之间。")
    if not isinstance(include_temperature_review, bool):
        raise TemperaturePipelineError("包含温度待判断数据必须为布尔值。")
    if measurements.empty:
        raise TemperaturePipelineError("传感器数据库没有可用于预测的数据。")

    required = {"timestamp", *FEATURE_COLUMNS}
    missing = sorted(required - set(measurements.columns))
    if missing:
        raise TemperaturePipelineError(f"传感器数据库缺少预测字段：{missing}")

    prediction_view = _prediction_view(measurements, include_temperature_review)
    ordered = prediction_view.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if ordered["timestamp"].isna().any():
        raise TemperaturePipelineError("传感器数据库包含无法识别的时间。")
    ordered = ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if "processed_row_excluded" in ordered:
        ordered = ordered.loc[~ordered["processed_row_excluded"].fillna(False).astype(bool)].copy()
    if "model_input_available" in ordered:
        ordered = ordered.loc[ordered["model_input_available"].astype(bool)].copy()
    if "model_input_unresolved" in ordered:
        ordered = ordered.loc[~ordered["model_input_unresolved"].astype(bool)].copy()
    if "temperature_review_required" in ordered:
        ordered = ordered.loc[~ordered["temperature_review_required"].fillna(False).astype(bool)].copy()
    if "is_inserted_gap" in ordered:
        final_rows = ordered.get("processing_version", pd.Series("legacy", index=ordered.index)).isin(("final_v1", "final_v2", "final_v3"))
        ordered = ordered.loc[~ordered["is_inserted_gap"].astype(bool) | final_rows].copy()
    finite_input = np.isfinite(
        ordered.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    ordered = ordered.loc[finite_input].copy()
    if ordered.empty:
        raise TemperaturePipelineError("清洗数据库没有可用于预测的输入记录。")

    row_count = max(1, math.ceil(len(ordered) * percentage / 100))
    selected = ordered.tail(row_count).copy().reset_index(drop=True)
    selected["source_row_count"] = pd.to_numeric(
        selected.get("source_row_count", pd.Series(1, index=selected.index)), errors="coerce"
    ).fillna(1).astype(int)
    for target in TARGET_COLUMNS:
        if target not in selected:
            selected[target] = np.nan
        flag = f"{target}_gap_imputed"
        if flag in selected:
            selected.loc[selected[flag].fillna(False).astype(bool), target] = np.nan

    frame = selected.loc[:, DATABASE_PREDICTION_COLUMNS].copy()
    if include_temperature_review:
        frame["temperature_review_included"] = selected["temperature_review_included"].astype(bool)
    feature_values = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise TemperaturePipelineError("所选数据库记录包含缺失或非有限的预测特征。")

    audit = {
        "gateway": GATEWAY,
        "source": "csv_database",
        "database_rows": len(ordered),
        "prediction_percentage": int(percentage),
        "prediction_rows": len(frame),
        "selection_policy": "latest_rows_chronological",
        "timestamp_min": frame["timestamp"].min().isoformat(),
        "timestamp_max": frame["timestamp"].max().isoformat(),
        "cleaning_applied": include_temperature_review or "cleaning_policy_sha256" in measurements.columns,
        "include_temperature_review": include_temperature_review,
        "temperature_review_prediction_rows": int(selected["temperature_review_included"].sum())
        if include_temperature_review else 0,
    }
    context = (exogenous_history(prediction_view) if "history_barrier" in prediction_view
               else ordered.loc[:, ["timestamp", *FEATURE_COLUMNS]].copy())
    return NormalizedUpload(
        frame=frame,
        rejectedRows=pd.DataFrame(),
        audit=audit,
        featureContext=context,
        inputHistory=exogenous_history(prediction_view),
    )
