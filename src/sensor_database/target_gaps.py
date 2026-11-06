"""Estimate short missing target runs without treating estimates as observations."""

import numpy as np
import pandas as pd

from .processing import MAX_ENDPOINT_MINUTES, MAX_MISSING_BUCKETS


def interpolate_target_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill bounded target gaps in the personal cleaning result, with audit flags."""
    result = frame.copy()
    config = ("temperature_jump_check_enabled", "temperature_rule_version")
    for column in ("tilt2_inline", "tilt2_crossline"):
        result[f"{column}_observed"] = result[column]
        result[f"{column}_gap_imputed"] = False
        missing = result[column].isna()
        groups = missing.ne(missing.shift(fill_value=False)).cumsum()
        for indexes in result.loc[missing].groupby(groups.loc[missing], sort=False).groups.values():
            positions = list(indexes)
            before, after = positions[0] - 1, positions[-1] + 1
            if len(positions) > MAX_MISSING_BUCKETS or before < 0 or after >= len(result):
                continue
            window = result.iloc[before:after + 1]
            times = pd.to_datetime(window["timestamp"])
            duration = times.iloc[-1] - times.iloc[0]
            if (not times.diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all()
                    or duration > pd.Timedelta(minutes=MAX_ENDPOINT_MINUTES)
                    or times.dt.normalize().nunique() != 1
                    or any(window[name].nunique(dropna=False) != 1 for name in config)
                    or window["processed_row_excluded"].fillna(False).any()):
                continue
            endpoints = result.iloc[[before, after]]
            values = endpoints[f"{column}_observed"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or endpoints["source_row_count"].le(0).any():
                continue
            fraction = (result.loc[positions, "timestamp"] - times.iloc[0]) / duration
            result.loc[positions, column] = values[0] + (values[1] - values[0]) * fraction
            result.loc[positions, f"{column}_gap_imputed"] = True
    return result
