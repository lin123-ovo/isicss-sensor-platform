"""Personal-edition zero sentinel repair, independent of signed jump policies."""

import numpy as np
import pandas as pd

from .processing import MAX_ENDPOINT_MINUTES, MAX_MISSING_BUCKETS


ZERO_COLUMNS = (
    "temperature_zero_invalid", "temperature_zero_interpolated",
    "processed_row_excluded", "history_barrier",
)


def prepare_zero_temperature(bucketed: pd.DataFrame) -> None:
    """Mask zero references before detectors; preserve their original audit value."""

    bucketed["temperature_zero_invalid"] = bucketed["temperature_raw_celsius"].eq(0.0)
    bucketed["_temperature_original_raw"] = bucketed["temperature_raw_celsius"]
    bucketed.loc[bucketed["temperature_zero_invalid"], "temperature_raw_celsius"] = np.nan


def finalize_zero_temperature(frame: pd.DataFrame) -> pd.DataFrame:
    """Repair only complete short runs; retain excluded rows as explicit barriers.

    Called after generic repairs, before final rounding. Endpoints are fixed
    original, non-imputed observations; repairs cannot become new endpoints.
    """

    result = frame.copy()
    invalid = result["temperature_zero_invalid"].fillna(False).astype(bool)
    result["temperature_zero_invalid"] = invalid
    result["temperature_zero_interpolated"] = False
    result["processed_row_excluded"] = invalid
    result["history_barrier"] = invalid
    result["processed_exclusion_reason"] = ""
    result["temperature_zero_run_length"] = 0
    result["temperature_raw_celsius"] = result.pop("_temperature_original_raw")
    result.loc[invalid, "temperature_celsius"] = np.nan
    for name in ("temperature_imputed", "temperature_repaired", "temperature_offset_corrected",
                 "temperature_review_required"):
        result.loc[invalid, name] = False
    result.loc[invalid, "temperature_repair_method"] = "none"
    result.loc[invalid, "temperature_offset_celsius"] = 0.0
    result.loc[invalid, "temperature_outlier"] = True
    result.loc[invalid, "temperature_unresolved"] = True
    result.loc[invalid, "temperature_reason"] = "temperature_zero_invalid"
    reliable = result["source_row_count"].gt(0) & ~invalid
    for prefix in ("temperature", "tilt1_inline", "tilt1_crossline"):
        reliable &= ~result[f"{prefix}_unresolved"] & ~result[f"{prefix}_imputed"]
    reliable &= ~result["temperature_review_required"] & ~result["temperature_offset_corrected"]
    reliable &= ~result["temperature_outlier"]
    groups = invalid.ne(invalid.shift(fill_value=False)).cumsum()
    config = ("temperature_jump_check_enabled", "temperature_rule_version")
    for _, indexes in result.loc[invalid].groupby(groups.loc[invalid], sort=False).groups.items():
        positions = list(indexes)
        result.loc[positions, "temperature_zero_run_length"] = len(positions)
        reason = "zero_run_exceeds_two" if len(positions) > MAX_MISSING_BUCKETS else "zero_no_reliable_endpoints"
        result.loc[positions, "processed_exclusion_reason"] = reason
        before, after = positions[0] - 1, positions[-1] + 1
        if len(positions) > MAX_MISSING_BUCKETS or before < 0 or after >= len(result):
            continue
        # A zero median can also come from conflicting/out-of-range raw samples.
        # Sentinel repair must never override the existing hard-invalid rule.
        if result.loc[positions, ["temperature_physical_outlier", "temperature_bucket_conflict"]].any().any():
            continue
        if not reliable.iloc[before] or not reliable.iloc[after]:
            continue
        window = result.iloc[before:after + 1]
        times = pd.to_datetime(window["timestamp"])
        duration = times.iloc[-1] - times.iloc[0]
        if (not times.diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all()
                or duration > pd.Timedelta(minutes=MAX_ENDPOINT_MINUTES)
                or times.dt.normalize().nunique() != 1
                or any(window[name].nunique(dropna=False) != 1 for name in config)):
            continue
        endpoints = result.iloc[[before, after]]
        values = endpoints["temperature_raw_celsius"].to_numpy(float)
        if (not np.isfinite(values).all() or (values == 0).any()
                or endpoints["temperature_physical_outlier"].any()
                or endpoints["temperature_bucket_conflict"].any()):
            continue
        fraction = (result.loc[positions, "timestamp"] - times.iloc[0]) / duration
        result.loc[positions, "temperature_celsius"] = values[0] + (values[1] - values[0]) * fraction
        for name in ("temperature_zero_interpolated", "temperature_imputed", "temperature_repaired"):
            result.loc[positions, name] = True
        for name in ("processed_row_excluded", "history_barrier", "temperature_unresolved"):
            result.loc[positions, name] = False
        result.loc[positions, "processed_exclusion_reason"] = ""
        result.loc[positions, "temperature_reason"] = "temperature_zero_interpolation"
        result.loc[positions, "temperature_repair_method"] = "interpolation"
    return result
