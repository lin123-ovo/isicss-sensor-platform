"""Versioned interpolation of fully missing short observation runs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

_FIELDS = (
    ('tilt1_inline_model', 'tilt1_inline_raw', 'tilt1_inline'),
    ('tilt1_crossline_model', 'tilt1_crossline_raw', 'tilt1_crossline'),
    ('temperature_celsius', 'temperature_raw_celsius', 'temperature'),
    ('tilt2_inline', 'tilt2_inline_observed', 'tilt2_inline'),
    ('tilt2_crossline', 'tilt2_crossline_observed', 'tilt2_crossline'),
)
_CONFIG = ('temperature_jump_check_enabled', 'temperature_rule_version')


def interpolate_whole_row_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill 1–2 fully missing ten-minute rows using two observed endpoints.

    Raw fields and target-observation fields remain unchanged. Returned estimates
    can use an endpoint on the next day and must never become target truth.
    """
    if frame.empty:
        result = frame.copy()
        result['whole_row_gap_imputed'] = pd.Series(dtype=bool)
        result['whole_gap_reference_sufficient'] = pd.Series(dtype=bool)
        return result
    result = frame.sort_values('timestamp', kind='stable').reset_index(drop=True).copy()
    times = pd.to_datetime(result['timestamp'], errors='raise')
    if times.duplicated().any() or times.isna().any():
        raise ValueError('整行缺口插值要求唯一有效的时间桶。')
    if not times.is_monotonic_increasing:
        raise ValueError('整行缺口插值要求时间有序。')
    missing_times = []
    for i in range(1, len(result)):
        left, right = result.iloc[i-1], result.iloc[i]
        delta = right['timestamp'] - left['timestamp']
        if delta in (pd.Timedelta(minutes=20), pd.Timedelta(minutes=30)):
            # Do not materialize unrelated gaps when even the raw endpoints or
            # their processing profiles cannot support a five-field repair.
            endpoints = result.iloc[[i-1, i]]
            if (endpoints['source_row_count'].le(0).any()
                    or not np.isfinite(endpoints[[raw for _, raw, _ in _FIELDS]].to_numpy(dtype=float)).all()
                    or any(endpoints[col].isna().any() or endpoints[col].nunique() != 1 for col in _CONFIG)):
                continue
            missing_times.extend(pd.date_range(left['timestamp'] + pd.Timedelta(minutes=10),
                                               right['timestamp'] - pd.Timedelta(minutes=10), freq='10min'))
    bool_columns = [col for col in result if is_bool_dtype(result[col].dtype)]
    if missing_times:
        index = pd.DatetimeIndex(result.timestamp).union(pd.DatetimeIndex(missing_times)).sort_values()
        result = result.set_index('timestamp').reindex(index).rename_axis('timestamp').reset_index()
    result['source_row_count'] = result['source_row_count'].fillna(0).astype(int)
    result['is_inserted_gap'] = result['source_row_count'].eq(0)
    for col in bool_columns:
        result[col] = result[col].fillna(False).astype(bool)
    for col in result:
        if col.endswith(('_source_count', '_outlier_run_length', '_hampel_reference_count')):
            result[col] = result[col].fillna(0)
    for col in (*_CONFIG, 'segment_id', 'gateway'):
        result[col] = result[col].ffill()
    result['whole_row_gap_imputed'] = False
    result['whole_gap_reference_sufficient'] = False
    raw = [raw for _, raw, _ in _FIELDS]
    missing = result[raw].isna().all(axis=1)
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    for indexes in result.loc[missing].groupby(groups.loc[missing], sort=False).groups.values():
        positions = list(indexes)
        before, after = positions[0]-1, positions[-1]+1
        if len(positions) > 2 or before < 0 or after >= len(result):
            continue
        ends = result.iloc[[before, after]]
        if any(ends[col].isna().any() or ends[col].nunique(dropna=False) != 1 for col in _CONFIG):
            continue
        # Synthetic bucket placeholders have no own import profile: inherit only
        # after confirming the two real endpoint profiles are identical.
        inserted_positions = [pos for pos in positions if result.iloc[pos]['is_inserted_gap']]
        for col in _CONFIG:
            result.loc[inserted_positions, col] = ends.iloc[0][col]
        window = result.iloc[before:after+1]
        t = pd.to_datetime(window.timestamp)
        if not t.diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all():
            continue
        if any(window[col].isna().any() or window[col].nunique(dropna=False) != 1 for col in _CONFIG):
            continue
        if window['processed_row_excluded'].fillna(False).any() or window['history_barrier'].fillna(False).any():
            continue
        ends = result.iloc[[before, after]]
        if ends['source_row_count'].le(0).any() or not np.isfinite(ends[raw].to_numpy(dtype=float)).all():
            continue
        if (ends['temperature_review_required'].fillna(False).any()
                or ends['temperature_offset_corrected'].fillna(False).any()):
            continue
        bad = False
        for _, _, prefix in _FIELDS[:3]:
            for suffix in ('physical_outlier', 'bucket_conflict', 'outlier', 'imputed', 'unresolved'):
                if ends[f'{prefix}_{suffix}'].fillna(False).any(): bad = True
        for _, _, prefix in _FIELDS[3:]:
            if ends[f'{prefix}_gap_imputed'].fillna(False).any(): bad = True
        if bad:
            continue
        # Refuse anomalous invalid observations, including temperature sentinel zero.
        if window.loc[window.index[1:-1], ['temperature_zero_invalid', 'temperature_review_required']].fillna(False).any().any():
            continue
        fractions = (result.loc[positions, 'timestamp'] - t.iloc[0]) / (t.iloc[-1] - t.iloc[0])
        for value, observed, prefix in _FIELDS:
            left, right = ends[observed].to_numpy(dtype=float)
            result.loc[positions, value] = left + (right-left)*fractions.to_numpy(dtype=float)
            result.loc[positions, f'{prefix}_gap_imputed'] = True
            if prefix in ('tilt1_inline', 'tilt1_crossline', 'temperature'):
                result.loc[positions, f'{prefix}_imputed'] = True
                result.loc[positions, f'{prefix}_unresolved'] = False
                result.loc[positions, f'{prefix}_reason'] = 'whole_row_gap_interpolation'
        result.loc[positions, 'temperature_repaired'] = True
        result.loc[positions, 'temperature_repair_method'] = 'interpolation'
        result.loc[positions, 'whole_row_gap_imputed'] = True
        reference_ok = all((ends[f'{p}_hampel_reference_count'] >= 30).all() for _,_,p in _FIELDS[:3])
        result.loc[positions, 'whole_gap_reference_sufficient'] = bool(reference_ok)
    # Populate bookkeeping only; do not mark unfilled buckets as observed.
    result['temperature_rule_version'] = result['temperature_rule_version'].fillna('temperature_jump_v1')
    result['temperature_jump_check_enabled'] = result['temperature_jump_check_enabled'].fillna(False).astype(bool)
    result['segment_id'] = result['segment_id'].ffill().astype(int)
    result['gateway'] = result['gateway'].fillna('CSG124')
    return result
