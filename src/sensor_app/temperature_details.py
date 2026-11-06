"""Read-only temperature detail tables using the same masks as the summary cards."""

from __future__ import annotations

import pandas as pd


TEMPERATURE_CATEGORIES = (
    "温度插值修复", "温度偏移修正", "温度未解决异常", "温度待判断", "温度硬异常",
)

DETAIL_COLUMNS = {
    "timestamp": "时间",
    "temperature_raw_celsius": "原始温度（℃）",
    "temperature_celsius": "处理后温度（℃）",
    "temperature_repair_method": "处理方式",
    "temperature_offset_celsius": "偏移修正量（℃）",
    "temperature_reason": "温度处理原因",
    "temperature_review_reason": "待判断原因",
    "temperature_physical_outlier": "物理越界",
    "temperature_bucket_conflict": "桶内冲突",
    "temperature_event_start": "事件开始",
    "temperature_event_end": "事件结束",
    "quality_status": "整行质量状态",
    "model_sample_usable": "可用于训练",
    "temperature_rule_version": "温度规则版本",
}


def temperature_category_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return independent, index-aligned masks; absent/NULL flags count as false."""

    def flag(column: str) -> pd.Series:
        return frame.get(column, pd.Series(False, index=frame.index)).fillna(False).astype(bool)

    return dict(zip(TEMPERATURE_CATEGORIES, (
        flag("temperature_imputed"),
        flag("temperature_offset_corrected"),
        flag("temperature_unresolved"),
        flag("temperature_review_required"),
        flag("temperature_physical_outlier") | flag("temperature_bucket_conflict"),
    ), strict=True))


def temperature_detail_table(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    """Copy all matching records in time order without changing values or flags.

    Categories may overlap. Missing optional audit columns are omitted; required
    time/value columns and unknown categories raise ValueError. No rows are capped.
    """
    if category not in TEMPERATURE_CATEGORIES:
        raise ValueError(f"未知温度明细类别：{category}")
    required = ("timestamp", "temperature_raw_celsius", "temperature_celsius")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"缺少温度明细字段：{'、'.join(missing)}")
    columns = [column for column in DETAIL_COLUMNS if column in frame]
    return (
        frame.loc[temperature_category_masks(frame)[category], columns]
        .sort_values("timestamp", kind="stable")
        .rename(columns=DETAIL_COLUMNS)
        .reset_index(drop=True)
        .copy()
    )
