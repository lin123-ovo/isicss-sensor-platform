"""Build auditable row-level prediction reports for display and download."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .temperature_pipeline import (
    FEATURE_COLUMNS,
    PREDICTION_VALUE_COLUMNS,
    TARGET_COLUMNS,
    TemperaturePipelineError,
    absolute_relative_error_percent,
    regression_metrics,
    round_prediction_values,
)


IDENTITY_COLUMNS = ("model_version", "gateway", "timestamp")
ROW_METRIC_COLUMNS = (
    "error_inline",
    "absolute_error_inline",
    "squared_error_inline",
    "absolute_relative_error_inline_percent",
    "error_crossline",
    "absolute_error_crossline",
    "squared_error_crossline",
    "absolute_relative_error_crossline_percent",
)
AGGREGATE_METRIC_COLUMNS = (
    "mae",
    "rmse",
    "max_abs_error",
    "mean_abs_relative_error_percent",
    "max_abs_relative_error_percent",
)


def remove_retired_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Hide retired percentile and weighted-percentage metrics from old artifacts."""

    retired = [
        column
        for column in frame.columns
        if "p95" in str(column).lower() or "wape" in str(column).lower()
    ]
    return frame.drop(columns=retired, errors="ignore").copy()


def build_detailed_prediction_report(
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
    *,
    source_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one traceable row per test sample with row and test-set metrics.

    ``source_frame`` is only needed to upgrade an older persisted report that did
    not retain the three model inputs. New reports already contain those values.
    """

    report = normalize_prediction_values(remove_retired_metric_columns(predictions))
    if "timestamp" not in report.columns:
        raise TemperaturePipelineError("测试集逐条预测结果缺少 timestamp。")
    report["timestamp"] = pd.to_datetime(report["timestamp"], errors="raise")

    model_version = str(
        metadata.get("version_id") or metadata.get("run_id") or "未记录"
    )
    report["model_version"] = model_version
    report["gateway"] = str(metadata.get("gateway", "未记录"))
    report = _attach_missing_inputs(report, source_frame)
    recalculated_metrics = recalculate_test_metrics(report, metadata, metrics)
    report = _attach_test_metrics(report, metadata, recalculated_metrics)

    preferred = [
        *IDENTITY_COLUMNS,
        "tilt1_crossline_model",
        "tilt1_inline_model",
        "temperature_celsius",
        "tilt2_crossline",
        "tilt2_inline",
        "predicted_crossline",
        "predicted_inline",
        *ROW_METRIC_COLUMNS,
    ]
    preferred = [column for column in preferred if column in report.columns]
    for direction in ("crossline", "inline"):
        preferred.extend(
            f"test_{metric}_{direction}" for metric in AGGREGATE_METRIC_COLUMNS
        )
    remaining = [column for column in report.columns if column not in preferred]
    return remove_retired_metric_columns(
        report.loc[:, [*preferred, *remaining]]
    )


def normalize_prediction_values(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return rounded predictions with row errors rebuilt from those values."""

    report = predictions.copy()
    for column in PREDICTION_VALUE_COLUMNS:
        if column in report:
            report[column] = round_prediction_values(
                pd.to_numeric(report[column], errors="coerce").to_numpy(dtype=float)
            )
    return _attach_row_metrics(report)


def recalculate_test_metrics(
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild selected-model test metrics without rewriting old artifacts."""

    report = normalize_prediction_values(predictions)
    recalculated = remove_retired_metric_columns(metrics)
    selected_model = str(metadata.get("model", ""))
    for direction in ("inline", "crossline"):
        target = pd.to_numeric(report[f"tilt2_{direction}"], errors="coerce")
        predicted = pd.to_numeric(
            report[f"predicted_{direction}"], errors="coerce"
        )
        evaluable = np.isfinite(target.to_numpy(dtype=float)) & np.isfinite(
            predicted.to_numpy(dtype=float)
        )
        if not evaluable.any():
            raise TemperaturePipelineError(
                f"测试集没有可评价的 {direction} 预测。"
            )
        actual_values = target.loc[evaluable].to_numpy(dtype=float)
        predicted_values = predicted.loc[evaluable].to_numpy(dtype=float)
        timestamps = report.loc[evaluable, "timestamp"]
        calculated = regression_metrics(
            actual_values,
            predicted_values,
            timestamps,
        )
        row_mask = (
            (recalculated["split"] == "test")
            & (recalculated["model"] == selected_model)
            & (recalculated["direction"] == direction)
        )
        if int(row_mask.sum()) != 1:
            raise TemperaturePipelineError(
                f"测试集缺少 {direction} 方向的唯一汇总指标。"
            )
        recalculated.loc[row_mask, "sample_count"] = int(evaluable.sum())
        for name, value in calculated.items():
            if name.endswith("_timestamp"):
                value = pd.NA if pd.isna(value) else str(pd.Timestamp(value))
            recalculated.loc[row_mask, name] = value
    return recalculated


def _attach_missing_inputs(
    report: pd.DataFrame,
    source_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in report.columns]
    if not missing:
        return report
    if source_frame is None:
        raise TemperaturePipelineError(
            f"旧版测试结果缺少模型输入字段：{missing}，且没有可用于补齐的数据库快照。"
        )

    required = {"timestamp", *FEATURE_COLUMNS}
    absent = sorted(required - set(source_frame.columns))
    if absent:
        raise TemperaturePipelineError(f"数据库快照缺少字段：{absent}。")
    lookup = source_frame.loc[:, ["timestamp", *FEATURE_COLUMNS]].copy()
    lookup["timestamp"] = pd.to_datetime(lookup["timestamp"], errors="raise")
    if lookup["timestamp"].duplicated().any():
        raise TemperaturePipelineError("数据库快照时间戳不唯一，无法可靠补齐旧版测试结果。")
    merged = report.merge(
        lookup,
        on="timestamp",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged.loc[:, list(FEATURE_COLUMNS)].isna().any().any():
        raise TemperaturePipelineError("旧版测试结果无法与当前数据库时间戳完整对应。")
    return merged


def _attach_row_metrics(report: pd.DataFrame) -> pd.DataFrame:
    required = {
        *TARGET_COLUMNS,
        "predicted_inline",
        "predicted_crossline",
    }
    missing = sorted(required - set(report.columns))
    if missing:
        raise TemperaturePipelineError(f"测试集逐条预测结果缺少字段：{missing}。")
    for direction in ("inline", "crossline"):
        error = report[f"predicted_{direction}"] - report[f"tilt2_{direction}"]
        report[f"error_{direction}"] = error
        report[f"absolute_error_{direction}"] = np.abs(error)
        report[f"squared_error_{direction}"] = np.square(error)
        report[f"absolute_relative_error_{direction}_percent"] = (
            absolute_relative_error_percent(
                report[f"tilt2_{direction}"],
                report[f"predicted_{direction}"],
            )
        )
    return report


def _attach_test_metrics(
    report: pd.DataFrame,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    selected = metrics.loc[
        (metrics["split"] == "test")
        & (metrics["model"] == str(metadata.get("model", "")))
    ]
    for direction in ("crossline", "inline"):
        row = selected.loc[selected["direction"] == direction]
        if len(row) != 1:
            raise TemperaturePipelineError(f"测试集缺少 {direction} 方向的汇总指标。")
        values = row.iloc[0]
        missing = [
            metric for metric in AGGREGATE_METRIC_COLUMNS if metric not in values.index
        ]
        calculated: dict[str, Any] = {}
        if missing:
            try:
                calculated = regression_metrics(
                    report[f"tilt2_{direction}"].to_numpy(dtype=float),
                    report[f"predicted_{direction}"].to_numpy(dtype=float),
                    report["timestamp"],
                )
            except TemperaturePipelineError:
                calculated = {metric: np.nan for metric in missing}
        for metric in AGGREGATE_METRIC_COLUMNS:
            value = values[metric] if metric in values.index else calculated[metric]
            report[f"test_{metric}_{direction}"] = value
    return report
