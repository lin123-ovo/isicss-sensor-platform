"""Interpolate bounded missing inputs after all event/temperature repairs."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from .processing import MAX_ENDPOINT_MINUTES, MAX_MISSING_BUCKETS


_FIELDS = (
    ("tilt1_inline_model", "tilt1_inline"),
    ("tilt1_crossline_model", "tilt1_crossline"),
    ("temperature_celsius", "temperature"),
)
_CONFIG_COLUMNS = ("temperature_jump_check_enabled", "temperature_rule_version")


def _matching_context(left: pd.Series, right: pd.Series) -> bool:
    return (
        left["timestamp"].normalize() == right["timestamp"].normalize()
        and all(pd.notna(left[name]) and pd.notna(right[name])
                and left[name] == right[name] for name in _CONFIG_COLUMNS)
    )


def finalize_short_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Return final inputs with at most two bounded missing buckets interpolated.

    Detection segments are never recomputed. Only short gaps between matching
    observed contexts receive new rows; neither source values nor targets change.
    """

    result = frame.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    boolean_columns = [name for name in result if is_bool_dtype(result[name].dtype)]
    inserted_times: list[pd.Timestamp] = []
    for position in range(1, len(result)):
        left, right = result.iloc[position - 1], result.iloc[position]
        elapsed = (right["timestamp"] - left["timestamp"]).total_seconds() / 60
        if 10 < elapsed <= MAX_ENDPOINT_MINUTES and _matching_context(left, right):
            inserted_times.extend(pd.date_range(
                left["timestamp"] + pd.Timedelta(minutes=10),
                right["timestamp"] - pd.Timedelta(minutes=10), freq="10min",
            ))
    if inserted_times:
        index = pd.DatetimeIndex(result["timestamp"]).union(pd.DatetimeIndex(inserted_times)).sort_values()
        result = result.set_index("timestamp").reindex(index).rename_axis("timestamp").reset_index()

    result["source_row_count"] = result["source_row_count"].fillna(0).astype(int)
    result["is_inserted_gap"] = result["source_row_count"].eq(0)
    for name in boolean_columns:
        result[name] = result[name].astype("boolean").fillna(False).astype(bool)
    for name in result:
        if name.endswith(("_source_count", "_outlier_run_length", "_hampel_reference_count")):
            result[name] = result[name].fillna(0)
    for _, prefix in _FIELDS:
        result[f"{prefix}_reason"] = result[f"{prefix}_reason"].fillna("")
        result[f"{prefix}_gap_imputed"] = False
    result["temperature_repair_method"] = result["temperature_repair_method"].fillna("none")
    result["temperature_review_reason"] = result["temperature_review_reason"].fillna("")
    result["temperature_event_id"] = result["temperature_event_id"].fillna("")
    result["temperature_offset_celsius"] = result["temperature_offset_celsius"].fillna(0.)

    # Fix endpoint eligibility before interpolating, preventing chained estimates.
    reliable = result["source_row_count"].gt(0) & ~result["temperature_review_required"]
    for _, prefix in _FIELDS:
        reliable &= ~result[f"{prefix}_unresolved"] & ~result[f"{prefix}_imputed"]

    for value_column, prefix in _FIELDS:
        missing = result[value_column].isna()
        groups = missing.ne(missing.shift(fill_value=False)).cumsum()
        for _, indexes in result.loc[missing].groupby(groups.loc[missing], sort=False).groups.items():
            positions = list(indexes)
            if len(positions) > MAX_MISSING_BUCKETS:
                continue
            before, after = positions[0] - 1, positions[-1] + 1
            if before < 0 or after >= len(result) or not reliable.iloc[before] or not reliable.iloc[after]:
                continue
            left, right = result.iloc[before], result.iloc[after]
            duration = right["timestamp"] - left["timestamp"]
            if not pd.Timedelta(0) < duration <= pd.Timedelta(minutes=MAX_ENDPOINT_MINUTES):
                continue
            if not _matching_context(left, right):
                continue
            if not np.isfinite(left[value_column]) or not np.isfinite(right[value_column]):
                continue
            gap = result.loc[positions]
            if prefix == "temperature" and gap.get(
                "temperature_zero_invalid", pd.Series(False, index=gap.index)
            ).any():
                continue
            if (gap[f"{prefix}_physical_outlier"] | gap[f"{prefix}_bucket_conflict"]
                    | gap[f"{prefix}_outlier"] | gap["temperature_review_required"]).any():
                continue
            observed_gap = gap.loc[~gap["is_inserted_gap"]]
            if any(observed_gap[name].ne(left[name]).any() for name in _CONFIG_COLUMNS):
                continue
            fractions = (result.loc[positions, "timestamp"] - left["timestamp"]) / duration
            result.loc[positions, value_column] = (
                left[value_column] + (right[value_column] - left[value_column]) * fractions
            )
            result.loc[positions, f"{prefix}_imputed"] = True
            result.loc[positions, f"{prefix}_gap_imputed"] = True
            result.loc[positions, f"{prefix}_unresolved"] = False
            result.loc[positions, f"{prefix}_reason"] = "short_gap_interpolation"
            for name in _CONFIG_COLUMNS:
                result.loc[positions, name] = left[name]
            result.loc[positions, "segment_id"] = result.loc[positions, "segment_id"].fillna(left["segment_id"])
            result.loc[positions, "gateway"] = result.loc[positions, "gateway"].fillna(left["gateway"])
            if prefix == "temperature":
                result.loc[positions, "temperature_repaired"] = True
                result.loc[positions, "temperature_repair_method"] = "interpolation"

    # Unfilled inserted rows retain explicit missing-data audit, never anomaly flags.
    for value_column, prefix in _FIELDS:
        unresolved = result["is_inserted_gap"] & result[value_column].isna()
        result.loc[unresolved, f"{prefix}_unresolved"] = True
        result.loc[unresolved, f"{prefix}_reason"] = "missing_observation"
    result["temperature_jump_check_enabled"] = result["temperature_jump_check_enabled"].astype("boolean").fillna(False).astype(bool)
    result["temperature_rule_version"] = result["temperature_rule_version"].fillna("temperature_jump_v1")
    result["segment_id"] = result["segment_id"].ffill().astype(int)
    result["gateway"] = result["gateway"].fillna("CSG124")
    return result
