"""离线温度修复候选：短跳点插值、闭合偏移事件及两侧趋势估计。

所有计算使用实际观测时间；原值只读。候选还需由主清洗流程用冻结
Hampel 策略复查，不能直接作为训练输入。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RATE_LIMIT = 2.0
REFERENCE_MINUTES = 90.0
MIN_REFERENCE_POINTS = 3
MIN_REFERENCE_SPAN_MINUTES = 20.0
MAX_OFFSET_DISAGREEMENT = 1.0
MAX_REFERENCE_RESIDUAL = 1.0
MAX_SPIKE_POINTS = 2
MAX_OBSERVATION_GAP_MINUTES = 20.0

REPAIR_COLUMNS = (
    "temperature_repair_method", "temperature_offset_celsius", "temperature_event_id",
    "temperature_offset_corrected", "temperature_repaired",
    "temperature_post_repair_hampel_outlier", "_temperature_candidate_celsius",
)


def repair_settings() -> dict[str, float | int]:
    """返回参与规则签名的全部候选修复参数。"""
    return {
        "rate_limit_celsius_per_10min": RATE_LIMIT,
        "max_observation_gap_minutes": MAX_OBSERVATION_GAP_MINUTES,
        "max_spike_buckets": MAX_SPIKE_POINTS,
        "reference_window_minutes": REFERENCE_MINUTES,
        "min_reference_points": MIN_REFERENCE_POINTS,
        "min_reference_span_minutes": MIN_REFERENCE_SPAN_MINUTES,
        "max_offset_disagreement_celsius": MAX_OFFSET_DISAGREEMENT,
        "max_reference_residual_celsius": MAX_REFERENCE_RESIDUAL,
        "min_offset_celsius_exclusive": RATE_LIMIT,
    }


def exceeds_limit(value: float | np.ndarray) -> bool | np.ndarray:
    """避免浮点误差将恰好位于阈值的变化误判为突变。"""
    return (value > RATE_LIMIT) & ~np.isclose(value, RATE_LIMIT, rtol=0.0, atol=1e-10)


def initialize_repair_columns(frame: pd.DataFrame) -> None:
    frame["temperature_repair_method"] = "none"
    frame["temperature_offset_celsius"] = 0.0
    frame["temperature_event_id"] = ""
    frame["temperature_offset_corrected"] = False
    frame["temperature_repaired"] = False
    frame["temperature_post_repair_hampel_outlier"] = False
    frame["_temperature_candidate_celsius"] = np.nan


def _reference_temperature(
    times: pd.DatetimeIndex, raw: np.ndarray, positions: np.ndarray, target: int,
) -> float | None:
    if len(positions) < MIN_REFERENCE_POINTS:
        return None
    minutes = (times[positions] - times[target]).total_seconds().to_numpy() / 60.0
    if np.ptp(minutes) < MIN_REFERENCE_SPAN_MINUTES:
        return None
    design = np.column_stack((minutes, np.ones(len(minutes))))
    slope, intercept = np.linalg.lstsq(design, raw[positions], rcond=None)[0]
    residual = raw[positions] - (slope * minutes + intercept)
    if np.max(np.abs(residual)) > MAX_REFERENCE_RESIDUAL:
        return None
    return float(intercept)


def propose_temperature_repairs(part: pd.DataFrame) -> None:
    """原地标记一个完整、连续、开启v2的实际桶段，并生成修复候选。"""
    raw = part["temperature_raw_celsius"].to_numpy(dtype=float)
    values = raw.copy()
    times = pd.DatetimeIndex(part["observation_timestamp"])
    n = len(part)
    hard = (part["temperature_physical_outlier"] | part["temperature_bucket_conflict"]).to_numpy(bool)
    reliable = np.isfinite(raw) & ~hard
    hampel = part["temperature_hampel_outlier"].to_numpy(bool)
    elapsed = np.full(n, np.nan)
    if n > 1:
        elapsed[1:] = (times[1:] - times[:-1]).total_seconds() / 60.0
    valid_edges = np.zeros(n, dtype=bool)
    valid_edges[1:] = reliable[1:] & reliable[:-1] & (elapsed[1:] > 0) & (
        elapsed[1:] <= MAX_OBSERVATION_GAP_MINUTES
    )

    def rates(series: np.ndarray) -> np.ndarray:
        result = np.full(n, np.nan)
        positions = np.flatnonzero(valid_edges)
        result[positions] = (series[positions] - series[positions - 1]) / elapsed[positions] * 10.0
        return result

    original_rates = rates(raw)
    suspicious = np.zeros(n, dtype=bool)
    isolated = np.zeros(n, dtype=bool)
    methods = np.full(n, "none", dtype=object)
    offsets = np.zeros(n)
    event_ids = np.full(n, "", dtype=object)

    def identify(start: int, stop: int) -> None:
        suspicious[start:stop] = True
        event_ids[start:stop] = f"temperature:{times[start].isoformat()}"

    # 短事件先处理，避免它们制造虚假的长段入口或污染两侧参考值。
    position = 1
    while position < n - 1:
        current_rate = (
            abs(values[position] - values[position - 1]) / elapsed[position] * 10.0
            if valid_edges[position] else np.nan
        )
        if not exceeds_limit(current_rate) or suspicious[position - 1]:
            position += 1
            continue
        repaired = False
        for length in range(1, MAX_SPIKE_POINTS + 1):
            stop = position + length
            if stop >= n or not valid_edges[position:stop + 1].all():
                continue
            if hampel[position - 1] or hampel[stop] or suspicious[stop]:
                continue
            entry_change = values[position] - values[position - 1]
            exit_change = values[stop] - values[stop - 1]
            if (
                entry_change * exit_change >= 0
                or not exceeds_limit(abs(exit_change) / elapsed[stop] * 10.0)
            ):
                continue
            duration = (times[stop] - times[position - 1]).total_seconds() / 60.0
            if exceeds_limit(abs(values[stop] - values[position - 1]) / duration * 10.0):
                continue
            fractions = (times[position:stop] - times[position - 1]).total_seconds() / (duration * 60.0)
            proposed = values[position - 1] + (values[stop] - values[position - 1]) * fractions
            deviations = raw[position:stop] - proposed
            if not exceeds_limit(np.abs(deviations)).all() or not (
                (deviations > 0).all() or (deviations < 0).all()
            ):
                continue
            identify(position, stop)
            methods[position:stop] = "interpolation"
            values[position:stop] = proposed
            position = stop + 1
            repaired = True
            break
        if not repaired:
            position += 1

    signed_rates = rates(values)
    events: list[tuple[int, int, bool, bool]] = []
    position = 1
    while position < n:
        if not exceeds_limit(abs(signed_rates[position])):
            position += 1
            continue
        start = position
        direction = np.sign(signed_rates[start])
        stop = start + 1
        internal_jump = False
        while stop < n:
            if exceeds_limit(abs(signed_rates[stop])):
                if np.sign(signed_rates[stop]) == -direction:
                    break
                internal_jump = True
            stop += 1
        identify(start, stop)
        isolated[start:stop] = True
        events.append((start, stop, stop < n, internal_jump))
        position = stop + 1

    # 先标记全部事件，再选参考点，避免相邻可疑段互相充当正常数据。
    references = reliable & ~hampel & ~suspicious
    for start, stop, closed, internal_jump in events:
        if not closed or internal_jump or stop - start < 3 or not reliable[start:stop].all():
            continue
        if not valid_edges[start:stop + 1].all() or suspicious[start - 1] or suspicious[stop]:
            continue
        left_minutes = (times[start] - times).total_seconds().to_numpy() / 60.0
        right_minutes = (times - times[stop - 1]).total_seconds().to_numpy() / 60.0
        left_positions = np.flatnonzero(references & (left_minutes > 0) & (left_minutes <= REFERENCE_MINUTES))
        right_positions = np.flatnonzero(references & (right_minutes > 0) & (right_minutes <= REFERENCE_MINUTES))
        left = _reference_temperature(times, raw, left_positions, start)
        right = _reference_temperature(times, raw, right_positions, stop - 1)
        if left is None or right is None:
            continue
        left_offset, right_offset = raw[start] - left, raw[stop - 1] - right
        if (
            left_offset * right_offset <= 0
            or not exceeds_limit(abs(left_offset))
            or not exceeds_limit(abs(right_offset))
            or abs(left_offset - right_offset) > MAX_OFFSET_DISAGREEMENT
        ):
            continue
        offset = (left_offset + right_offset) / 2.0
        proposed = raw[start:stop] - offset
        boundary_rates = np.array([
            abs(proposed[0] - values[start - 1]) / elapsed[start] * 10.0,
            abs(values[stop] - proposed[-1]) / elapsed[stop] * 10.0,
        ])
        if exceeds_limit(boundary_rates).any():
            continue
        values[start:stop] = proposed
        methods[start:stop] = "offset"
        offsets[start:stop] = offset
        isolated[start:stop] = False

    values[isolated | hard] = np.nan
    part["temperature_rate_celsius_per_10min"] = np.abs(original_rates)
    part["temperature_jump_trigger"] = exceeds_limit(np.abs(original_rates))
    part["temperature_jump_outlier"] = suspicious
    part["temperature_jump_isolated"] = isolated
    part["temperature_repair_method"] = methods
    part["temperature_offset_celsius"] = offsets
    part["temperature_event_id"] = event_ids
    part["_temperature_candidate_celsius"] = values
