"""Read-only diagnostics for information available to crossline prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .temperature_pipeline import TemperaturePipelineError


DIAGNOSTIC_FIELDS = (
    "tilt1_inline_model",
    "tilt1_crossline_model",
    "temperature_celsius",
    "tilt2_crossline",
)
FEATURE_LABELS = {
    "tilt1_inline_model": "1号顺线",
    "tilt1_crossline_model": "1号横线",
    "temperature_celsius": "温度",
    "tilt2_crossline": "2号横线",
}


@dataclass(frozen=True)
class CrosslineInformationReport:
    """Tables and findings from one immutable database snapshot."""

    summary: dict[str, Any]
    field_quality: pd.DataFrame
    lag_analysis: pd.DataFrame
    best_lags: pd.DataFrame
    findings: tuple[str, ...]


def analyze_crossline_information(
    measurements: pd.DataFrame,
    *,
    bucket_minutes: int = 10,
    maximum_lag_minutes: int = 60,
) -> CrosslineInformationReport:
    """Measure crossline precision, time coverage, and lag relationships.

    Positive lag means the source feature changes before the 2号横线 target.
    The input frame is copied and is never cleaned or changed in place.
    """

    if isinstance(bucket_minutes, bool) or bucket_minutes <= 0:
        raise TemperaturePipelineError("诊断时间桶必须为正整数分钟。")
    if (
        isinstance(maximum_lag_minutes, bool)
        or maximum_lag_minutes < 0
        or maximum_lag_minutes % bucket_minutes != 0
    ):
        raise TemperaturePipelineError("最大滞后时间必须是诊断时间桶的非负整数倍。")
    required = {"timestamp", *DIAGNOSTIC_FIELDS}
    missing = sorted(required - set(measurements.columns))
    if missing:
        raise TemperaturePipelineError(f"横线数据诊断缺少字段：{missing}")
    if measurements.empty:
        raise TemperaturePipelineError("横线数据诊断没有可用记录。")

    source = measurements.loc[:, ["timestamp", *DIAGNOSTIC_FIELDS]].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    for field in DIAGNOSTIC_FIELDS:
        source[field] = pd.to_numeric(source[field], errors="coerce")
        source.loc[~np.isfinite(source[field]), field] = np.nan

    recognizable_time_rows = int(source["timestamp"].notna().sum())
    if recognizable_time_rows == 0:
        raise TemperaturePipelineError("横线数据诊断没有可识别时间。")
    complete = source.dropna(subset=["timestamp", *DIAGNOSTIC_FIELDS]).copy()
    if len(complete) < 3:
        raise TemperaturePipelineError("横线数据诊断至少需要3条完整记录。")
    complete = complete.sort_values("timestamp", kind="stable").reset_index(drop=True)

    duplicate_timestamps = int(complete["timestamp"].duplicated().sum())
    frequency = f"{bucket_minutes}min"
    complete["bucket"] = complete["timestamp"].dt.round(frequency)
    bucketed = (
        complete.groupby("bucket", sort=True)[list(DIAGNOSTIC_FIELDS)]
        .median()
        .sort_index()
    )
    if bucketed.empty:
        raise TemperaturePipelineError("横线数据诊断无法建立时间桶。")
    expected_index = pd.date_range(
        bucketed.index.min(),
        bucketed.index.max(),
        freq=frequency,
    )
    regular = bucketed.reindex(expected_index)

    raw_intervals = complete["timestamp"].diff().dt.total_seconds().div(60).dropna()
    present_bucket_intervals = (
        bucketed.index.to_series().diff().dt.total_seconds().div(60).dropna()
    )
    field_quality = _build_field_quality(complete, regular)
    lag_analysis = _build_lag_analysis(
        regular,
        bucket_minutes=bucket_minutes,
        maximum_lag_minutes=maximum_lag_minutes,
    )
    best_lags = _select_best_lags(lag_analysis)
    target_quality = field_quality.set_index("field").loc["tilt2_crossline"]
    crossline_best = best_lags.set_index("feature").loc["tilt1_crossline_model"]

    summary = {
        "source_rows": int(len(measurements)),
        "complete_rows": int(len(complete)),
        "invalid_rows": int(len(measurements) - len(complete)),
        "duplicate_timestamps": duplicate_timestamps,
        "effective_bucket_rows": int(len(bucketed)),
        "collapsed_rows": int(len(complete) - len(bucketed)),
        "expected_bucket_rows": int(len(expected_index)),
        "missing_bucket_rows": int(len(expected_index) - len(bucketed)),
        "bucket_coverage_percent": float(len(bucketed) / len(expected_index) * 100.0),
        "median_raw_interval_minutes": (
            float(raw_intervals.median()) if not raw_intervals.empty else np.nan
        ),
        "maximum_bucket_gap_minutes": (
            float(present_bucket_intervals.max())
            if not present_bucket_intervals.empty
            else np.nan
        ),
        "target_source_unique_values": int(target_quality["source_unique_values"]),
        "target_source_resolution_degrees": float(target_quality["source_resolution"]),
        "target_unchanged_fraction": float(target_quality["unchanged_fraction"]),
        "best_crossline_lag_minutes": int(crossline_best["lag_minutes"]),
        "best_crossline_lag_correlation": float(
            crossline_best["pearson_correlation"]
        ),
        "best_crossline_lag_pairs": int(crossline_best["pair_count"]),
    }
    findings = _build_findings(summary)
    return CrosslineInformationReport(
        summary=summary,
        field_quality=field_quality,
        lag_analysis=lag_analysis,
        best_lags=best_lags,
        findings=findings,
    )


def _build_field_quality(
    source: pd.DataFrame,
    regular: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize observed source precision without inventing extra decimals."""

    rows = []
    for field in DIAGNOSTIC_FIELDS:
        values = source[field].dropna().to_numpy(dtype=float)
        unique = np.unique(values)
        differences = np.diff(unique)
        positive = differences[differences > 1e-12]
        resolution = float(positive.min()) if positive.size else np.nan
        changes = regular[field].diff().dropna()
        unchanged_fraction = float(changes.eq(0.0).mean()) if len(changes) else np.nan
        rows.append(
            {
                "field": field,
                "label": FEATURE_LABELS[field],
                "source_unique_values": int(len(unique)),
                "source_resolution": resolution,
                "unchanged_fraction": unchanged_fraction,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def _build_lag_analysis(
    regular: pd.DataFrame,
    *,
    bucket_minutes: int,
    maximum_lag_minutes: int,
) -> pd.DataFrame:
    """Compare feature changes with target changes at bounded time offsets."""

    target_change = regular["tilt2_crossline"].diff()
    maximum_steps = maximum_lag_minutes // bucket_minutes
    rows = []
    for feature in DIAGNOSTIC_FIELDS[:-1]:
        feature_change = regular[feature].diff()
        for lag_steps in range(-maximum_steps, maximum_steps + 1):
            lag_minutes = lag_steps * bucket_minutes
            paired = pd.concat(
                [feature_change, target_change.shift(-lag_steps)],
                axis=1,
                keys=["feature_change", "target_change"],
            ).dropna()
            pearson = _safe_correlation(paired, rank=False)
            spearman = _safe_correlation(paired, rank=True)
            rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "lag_minutes": lag_minutes,
                    "relationship": _lag_relationship(lag_minutes),
                    "pair_count": int(len(paired)),
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                }
            )
    return pd.DataFrame(rows)


def _safe_correlation(paired: pd.DataFrame, *, rank: bool) -> float:
    if len(paired) < 3:
        return np.nan
    values = paired.rank(method="average") if rank else paired
    if values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(values.iloc[:, 0].corr(values.iloc[:, 1]))


def _select_best_lags(lag_analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in lag_analysis.groupby("feature", sort=False):
        valid = group.dropna(subset=["pearson_correlation"]).copy()
        if valid.empty:
            rows.append(group.iloc[0].to_dict())
            continue
        selected = valid.loc[valid["pearson_correlation"].abs().idxmax()]
        rows.append(selected.to_dict())
    return pd.DataFrame(rows).reset_index(drop=True)


def _lag_relationship(lag_minutes: int) -> str:
    if lag_minutes > 0:
        return "1号领先2号"
    if lag_minutes < 0:
        return "2号领先1号"
    return "同步"


def _build_findings(summary: dict[str, Any]) -> tuple[str, ...]:
    findings = []
    if summary["collapsed_rows"] > 0:
        findings.append(
            f"{summary['source_rows']}条源记录在10分钟粒度下形成"
            f"{summary['effective_bucket_rows']}个有效桶；"
            f"{summary['collapsed_rows']}条记录与同桶记录合并，不会增加独立时间信息。"
        )
    if summary["target_source_unique_values"] < 20:
        findings.append(
            "2号横线原始取值层级较少："
            f"仅{summary['target_source_unique_values']}个不同值，最小步进约"
            f"{summary['target_source_resolution_degrees']:.2f}°；"
            "应优先确认采集或导出环节是否保留了设备原始小数位。"
        )
    correlation = abs(summary["best_crossline_lag_correlation"])
    if np.isfinite(correlation) and correlation < 0.1:
        findings.append(
            "在±60分钟范围内未发现稳定的1号横线领先/滞后关系："
            f"最佳变化相关系数仅{summary['best_crossline_lag_correlation']:.2f}。"
        )
    if summary["missing_bucket_rows"] > 0:
        findings.append(
            f"时间轴缺少{summary['missing_bucket_rows']}个10分钟桶；"
            "建模时应继续保留缺口，不插值伪造观测。"
        )
    if not findings:
        findings.append("当前精度、时间覆盖和滞后关系未触发信息量预警。")
    return tuple(findings)
