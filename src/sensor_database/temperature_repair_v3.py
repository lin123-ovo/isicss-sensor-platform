"""独立的v3温度流程：有限局部事件、双侧趋势评分和保留原值的待判断状态。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .temperature_repair import (
    MAX_OBSERVATION_GAP_MINUTES, MAX_REFERENCE_RESIDUAL, MAX_SPIKE_POINTS,
    MIN_REFERENCE_POINTS, MIN_REFERENCE_SPAN_MINUTES, RATE_LIMIT, REFERENCE_MINUTES,
    exceeds_limit,
)


RULE_VERSION = "temperature_jump_v3"
MAX_EVENT_MINUTES = 360.0
MIN_EVENT_BUCKETS = 3
MIN_TEMPERATURE = -40.0
MAX_TEMPERATURE = 50.0
NUMERICAL_TOLERANCE = 1e-10
AUDIT_COLUMNS = (
    "temperature_review_required", "temperature_review_reason", "temperature_event_start",
    "temperature_event_end", "temperature_left_offset_celsius", "temperature_right_offset_celsius",
    "temperature_candidate_score", "temperature_entry_rate_after", "temperature_exit_rate_after",
    "temperature_estimate_end",
)


def repair_settings_v3() -> dict[str, float | int]:
    """返回v3的固定参数，参与批次版本签名。"""
    return {
        "rate_limit_celsius_per_10min": RATE_LIMIT,
        "max_observation_gap_minutes": MAX_OBSERVATION_GAP_MINUTES,
        "max_spike_buckets": MAX_SPIKE_POINTS,
        "reference_window_minutes": REFERENCE_MINUTES,
        "min_reference_points": MIN_REFERENCE_POINTS,
        "min_reference_span_minutes": MIN_REFERENCE_SPAN_MINUTES,
        "max_reference_residual_celsius": MAX_REFERENCE_RESIDUAL,
        "max_event_minutes": MAX_EVENT_MINUTES,
        "min_event_buckets": MIN_EVENT_BUCKETS,
        "minimum_temperature_celsius": MIN_TEMPERATURE,
        "maximum_temperature_celsius": MAX_TEMPERATURE,
        "allow_cross_date": 0,
        "hampel_veto": 0,
        "require_boundary_improvement": 1,
        "offset_score_weight": 1,
        "reference_score_weight": 1,
        "boundary_score_weight": 1,
        "score_tie_tolerance": NUMERICAL_TOLERANCE,
    }


def initialize_v3_audit(frame: pd.DataFrame) -> None:
    """为全部规则版本补齐可选审计列，旧版本保持默认空值。"""
    frame["temperature_review_required"] = False
    frame["temperature_review_reason"] = ""
    for column in ("temperature_event_start", "temperature_event_end", "temperature_estimate_end"):
        frame[column] = pd.NaT
    for column in AUDIT_COLUMNS[4:-1]:
        frame[column] = np.nan


@dataclass(frozen=True)
class ReferenceFit:
    """局部参考直线在目标处的温度、十分钟趋势及均方误差。"""
    expected: float
    rate_per_10min: float
    mean_squared_error: float


@dataclass(frozen=True)
class OffsetCandidate:
    """已通过边界验证的统一偏移候选；stop为首个未修正点。"""
    stop: int
    offset: float
    left_offset: float
    right_offset: float
    score: float
    entry_rate: float
    exit_rate: float


def _fit_reference(
    times: pd.DatetimeIndex, raw: np.ndarray, positions: np.ndarray, target: int,
) -> tuple[ReferenceFit | None, str]:
    if len(positions) < MIN_REFERENCE_POINTS:
        return None, "insufficient_reference_points"
    minutes = (times[positions] - times[target]).total_seconds().to_numpy() / 60.0
    if np.ptp(minutes) < MIN_REFERENCE_SPAN_MINUTES:
        return None, "insufficient_reference_span"
    design = np.column_stack((minutes, np.ones(len(minutes))))
    slope, intercept = np.linalg.lstsq(design, raw[positions], rcond=None)[0]
    residuals = raw[positions] - (slope * minutes + intercept)
    if np.max(np.abs(residuals)) > MAX_REFERENCE_RESIDUAL:
        return None, "reference_fit_residual"
    return ReferenceFit(float(intercept), float(slope * 10.0), float(np.mean(residuals ** 2))), ""


def _short_spikes(
    raw: np.ndarray, times: pd.DatetimeIndex, good: np.ndarray, elapsed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    values = raw.copy()
    spikes = np.zeros(len(raw), dtype=bool)
    events: list[tuple[int, int]] = []
    position = 1
    while position < len(raw) - 1:
        if not good[position - 1:position + 1].all() or spikes[position - 1]:
            position += 1
            continue
        entry = values[position] - values[position - 1]
        if not exceeds_limit(abs(entry) / elapsed[position] * 10.0):
            position += 1
            continue
        for length in range(1, MAX_SPIKE_POINTS + 1):
            stop = position + length
            if stop >= len(raw) or not good[position - 1:stop + 1].all() or spikes[stop]:
                continue
            exit_change = values[stop] - values[stop - 1]
            if entry * exit_change >= 0 or not exceeds_limit(abs(exit_change) / elapsed[stop] * 10.0):
                continue
            duration = (times[stop] - times[position - 1]).total_seconds()
            if exceeds_limit(abs(values[stop] - values[position - 1]) / duration * 600.0):
                continue
            fractions = (times[position:stop] - times[position - 1]).total_seconds() / duration
            proposed = values[position - 1] + (values[stop] - values[position - 1]) * fractions
            deviations = raw[position:stop] - proposed
            if not exceeds_limit(np.abs(deviations)).all() or not (
                (deviations > 0).all() or (deviations < 0).all()
            ):
                continue
            values[position:stop] = proposed
            spikes[position:stop] = True
            events.append((position, stop))
            position = stop + 1
            break
        else:
            position += 1
    return values, spikes, events


def _candidate(
    start: int, stop: int, direction: float, values: np.ndarray, raw: np.ndarray,
    times: pd.DatetimeIndex, elapsed: np.ndarray, rates: np.ndarray, references: np.ndarray,
) -> tuple[OffsetCandidate | None, str, tuple[float, float] | None]:
    indexes = np.arange(len(raw))
    left_minutes = (times[start] - times).total_seconds().to_numpy() / 60.0
    right_minutes = (times - times[stop - 1]).total_seconds().to_numpy() / 60.0
    left = np.flatnonzero(references & (left_minutes > 0) & (left_minutes <= REFERENCE_MINUTES))
    right = np.flatnonzero(references & (right_minutes > 0) & (right_minutes <= REFERENCE_MINUTES))
    # 两侧参考不得越过另一个尚待解释的跳变；当前退出点本身可以作为正常参考。
    previous = np.flatnonzero((indexes < start) & exceeds_limit(np.abs(rates)))
    following = np.flatnonzero((indexes > stop) & exceeds_limit(np.abs(rates)))
    if len(previous):
        left = left[left >= previous[-1]]
    if len(following):
        right = right[right < following[0]]
    left_fit, left_reason = _fit_reference(times, raw, left, start)
    right_fit, right_reason = _fit_reference(times, raw, right, stop - 1)
    if left_fit is None or right_fit is None:
        return None, left_reason or right_reason, None
    dl = values[start] - left_fit.expected
    dr = values[stop - 1] - right_fit.expected
    estimates = (dl, dr)
    if dl * direction <= NUMERICAL_TOLERANCE or dr * direction <= NUMERICAL_TOLERANCE:
        return None, "offset_direction_mismatch", estimates
    offset = (dl + dr) / 2.0
    shifted = values[start:stop] - offset
    entry = (shifted[0] - values[start - 1]) / elapsed[start] * 10.0
    exit_rate = (values[stop] - shifted[-1]) / elapsed[stop] * 10.0
    if not np.isfinite(shifted).all() or not (
        (shifted >= MIN_TEMPERATURE) & (shifted <= MAX_TEMPERATURE)
    ).all():
        return None, "corrected_physical_range", estimates
    if exceeds_limit(np.array([abs(entry), abs(exit_rate)])).any():
        return None, "corrected_boundary_rate", estimates
    if (
        abs(entry) >= abs(rates[start]) - NUMERICAL_TOLERANCE
        or abs(exit_rate) >= abs(rates[stop]) - NUMERICAL_TOLERANCE
    ):
        return None, "boundary_not_improved", estimates
    score = (
        ((dl - offset) ** 2 + (dr - offset) ** 2) / 2.0
        + (left_fit.mean_squared_error + right_fit.mean_squared_error) / 2.0
        + ((entry - left_fit.rate_per_10min) ** 2 + (exit_rate - right_fit.rate_per_10min) ** 2) / 2.0
    )
    return OffsetCandidate(stop, offset, dl, dr, score, entry, exit_rate), "", estimates


def propose_temperature_repairs_v3(part: pd.DataFrame) -> None:
    """处理一个连续、同日、无硬异常的v3上下文，不调用旧温度清洗流程。"""
    raw = part["temperature_raw_celsius"].to_numpy(dtype=float)
    times = pd.DatetimeIndex(part["observation_timestamp"])
    hard = (part["temperature_physical_outlier"] | part["temperature_bucket_conflict"]).to_numpy(bool)
    good = np.isfinite(raw) & ~hard
    n = len(part)
    elapsed = np.full(n, np.nan)
    if n > 1:
        elapsed[1:] = (times[1:] - times[:-1]).total_seconds() / 60.0
    values, spikes, short_events = _short_spikes(raw, times, good, elapsed)
    signed_rates = np.full(n, np.nan)
    original_rates = np.full(n, np.nan)
    valid = good[1:] & good[:-1] & (elapsed[1:] > 0) & (elapsed[1:] <= MAX_OBSERVATION_GAP_MINUTES)
    edges = np.flatnonzero(valid) + 1
    signed_rates[edges] = (values[edges] - values[edges - 1]) / elapsed[edges] * 10.0
    original_rates[edges] = (raw[edges] - raw[edges - 1]) / elapsed[edges] * 10.0
    triggers = exceeds_limit(np.abs(signed_rates))
    suspect = spikes.copy()
    review = np.zeros(n, dtype=bool)
    shifted = np.zeros(n, dtype=bool)
    methods = np.where(spikes, "interpolation", "none").astype(object)
    reasons = np.full(n, "", dtype=object)
    offsets = np.zeros(n)
    ids = np.full(n, "", dtype=object)
    starts = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ends = starts.copy()
    audit = {name: np.full(n, np.nan) for name in AUDIT_COLUMNS[4:-1]}
    estimate_ends = starts.copy()

    def identify(start: int, stop: int) -> None:
        suspect[start:stop] = True
        ids[start:stop] = f"temperature:{times[start].isoformat()}"
        starts[start:stop] = times[start].to_datetime64()
        ends[start:stop] = times[stop - 1].to_datetime64()

    for start, stop in short_events:
        identify(start, stop)
    position = 1
    while position < n:
        if not triggers[position]:
            position += 1
            continue
        start = position
        direction = np.sign(signed_rates[start])
        boundary = start + 1
        boundary_included = False
        limit_reason = "context_boundary_or_end"
        while boundary < n:
            if (times[boundary] - times[start]).total_seconds() / 60.0 > MAX_EVENT_MINUTES:
                limit_reason = "max_event_duration"
                break
            if times[boundary].date() != times[start].date() or not np.isfinite(signed_rates[boundary]):
                limit_reason = "context_boundary"
                break
            if triggers[boundary]:
                if np.sign(signed_rates[boundary]) == direction:
                    limit_reason = "new_same_direction_jump"
                else:
                    boundary_included = True
                    limit_reason = "reverse_jump"
                break
            boundary += 1
        references = good & ~spikes & ~review & ~shifted
        best: OffsetCandidate | None = None
        failures: Counter[str] = Counter()
        last_estimates: tuple[float, float] | None = None
        last_estimate_stop: int | None = None
        maximum_stop = boundary + int(boundary_included)
        for stop in range(start + MIN_EVENT_BUCKETS, maximum_stop):
            if signed_rates[stop] * direction >= 0:
                continue
            candidate, reason, estimates = _candidate(
                start, stop, direction, values, raw, times, elapsed, signed_rates, references,
            )
            if estimates is not None:
                last_estimates = estimates
                last_estimate_stop = stop
            if candidate is None:
                failures[reason] += 1
            elif best is None or candidate.score < best.score - NUMERICAL_TOLERANCE:
                best = candidate
        if best is not None:
            stop = best.stop
            identify(start, stop)
            values[start:stop] -= best.offset
            shifted[start:stop] = True
            methods[start:stop] = np.where(spikes[start:stop], "interpolation_offset", "offset")
            offsets[start:stop] = best.offset
            estimate_ends[start:stop] = times[stop - 1].to_datetime64()
            audit["temperature_left_offset_celsius"][start:stop] = best.left_offset
            audit["temperature_right_offset_celsius"][start:stop] = best.right_offset
            audit["temperature_candidate_score"][start:stop] = best.score
            audit["temperature_entry_rate_after"][start:stop] = best.entry_rate
            audit["temperature_exit_rate_after"][start:stop] = best.exit_rate
            position = stop + 1
        else:
            stop = boundary
            identify(start, stop)
            review[start:stop] = True
            values[start:stop] = raw[start:stop]
            spikes[start:stop] = False
            methods[start:stop] = "review_required"
            failure = failures.most_common(1)[0][0] if failures else "no_reliable_opposite_exit"
            reasons[start:stop] = f"{failure};{limit_reason}"
            if last_estimates is not None and last_estimate_stop is not None:
                estimate_ends[start:stop] = times[last_estimate_stop - 1].to_datetime64()
                audit["temperature_left_offset_celsius"][start:stop] = last_estimates[0]
                audit["temperature_right_offset_celsius"][start:stop] = last_estimates[1]
            position = boundary + int(boundary_included)
    values[~good] = np.nan
    part["temperature_rate_celsius_per_10min"] = np.abs(original_rates)
    part["temperature_jump_trigger"] = exceeds_limit(np.abs(original_rates))
    part["temperature_jump_outlier"] = suspect
    part["temperature_jump_isolated"] = False
    part["temperature_repair_method"] = methods
    part["temperature_offset_celsius"] = offsets
    part["temperature_event_id"] = ids
    part["temperature_offset_corrected"] = shifted
    part["temperature_repaired"] = spikes | shifted
    part["temperature_review_required"] = review
    part["temperature_review_reason"] = reasons
    part["temperature_event_start"] = starts
    part["temperature_event_end"] = ends
    part["temperature_estimate_end"] = estimate_ends
    for column, data in audit.items():
        part[column] = data
    part["_temperature_candidate_celsius"] = values
