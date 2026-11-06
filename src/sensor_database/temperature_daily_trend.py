"""同日异常凸起检测与双侧平滑衔接；正常观测及原温度保持只读。"""

from __future__ import annotations

import numpy as np
import pandas as pd


RULE_VERSION = "temperature_daily_trend_v5"
HALF_WINDOW_MINUTES = 240.0
MIN_REFERENCE_POINTS = 12
MIN_REFERENCE_SPAN_MINUTES = 90.0
EVENT_RESIDUAL_CELSIUS = 1.5
EVENT_EDGE_CELSIUS_PER_10MIN = 1.5
RETURN_TOLERANCE_CELSIUS = 0.6
MAX_EVENT_MINUTES = 180.0
SHOULDER_DIP_CELSIUS = 0.6
ROBUST_ITERATIONS = 3
MIN_ROBUST_SCALE_CELSIUS = 0.3
ROBUST_CUTOFF = 4.685
MAX_GAP_MINUTES = 20.0
AUDIT_COLUMNS = (
    "temperature_daily_trend_celsius", "temperature_daily_residual_celsius",
    "temperature_daily_offset_celsius", "temperature_daily_checked",
    "temperature_daily_corrected", "temperature_daily_status",
)


def daily_trend_settings() -> dict[str, float | int]:
    """返回完整固定参数用于新版本签名，不改变旧温度版本的参数。"""
    return {
        "daily_half_window_minutes": HALF_WINDOW_MINUTES,
        "daily_min_reference_points": MIN_REFERENCE_POINTS,
        "daily_min_reference_span_minutes": MIN_REFERENCE_SPAN_MINUTES,
        "daily_event_residual_celsius": EVENT_RESIDUAL_CELSIUS,
        "daily_event_edge_celsius_per_10min": EVENT_EDGE_CELSIUS_PER_10MIN,
        "daily_return_tolerance_celsius": RETURN_TOLERANCE_CELSIUS,
        "daily_max_event_minutes": MAX_EVENT_MINUTES,
        "daily_shoulder_dip_celsius": SHOULDER_DIP_CELSIUS,
        "daily_monotone_hermite_bridge": 1,
        "daily_robust_iterations": ROBUST_ITERATIONS,
        "daily_min_robust_scale_celsius": MIN_ROBUST_SCALE_CELSIUS,
        "daily_robust_cutoff": ROBUST_CUTOFF,
        "daily_polynomial_degree": 2,
        "daily_mad_factor": 1.4826,
        "daily_max_gap_minutes": MAX_GAP_MINUTES,
        "daily_cross_date": 0,
        "daily_soft_deviation_excludes_model": 0,
        "minimum_temperature_celsius": -40.0,
        "maximum_temperature_celsius": 50.0,
    }


def initialize_daily_audit(frame: pd.DataFrame) -> None:
    """为新旧版本提供稳定审计列。"""
    for name in AUDIT_COLUMNS[:3]:
        frame[name] = np.nan
    frame["temperature_daily_checked"] = False
    frame["temperature_daily_corrected"] = False
    frame["temperature_daily_status"] = "disabled"


def estimate_local_trend(minutes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """按实际分钟拟合同一连续日内的稳健局部二次趋势，不修改输入。

    每点前后各4小时，至少12个有效点并覆盖90分钟；不足返回NaN。
    三轮Tukey残差加权减弱孤立尖峰的影响。不规定日最高温发生时间、
    不强制单峰，不使用其他日期、倾角或2号传感器真值。
    """
    time = np.asarray(minutes, dtype=float)
    raw = np.asarray(values, dtype=float)
    if time.ndim != 1 or raw.ndim != 1 or len(time) != len(raw):
        raise ValueError("时间和温度必须是等长一维数组")
    if not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError("趋势时间必须有限且严格递增")
    if np.isinf(raw).any():
        raise ValueError("温度不能包含无穷大")
    good = np.isfinite(raw)
    robust = good.astype(float)
    trend = np.full(len(raw), np.nan)
    for iteration in range(ROBUST_ITERATIONS + 1):
        for i, target in enumerate(time):
            distance = np.abs(time - target)
            selected = good & (distance < HALF_WINDOW_MINUTES) & (robust > 1e-8)
            if (selected.sum() < MIN_REFERENCE_POINTS
                    or np.ptp(time[selected]) < MIN_REFERENCE_SPAN_MINUTES):
                trend[i] = np.nan
                continue
            scaled = (time[selected] - target) / HALF_WINDOW_MINUTES
            weights = (1 - np.abs(scaled) ** 3) ** 3 * robust[selected]
            design = np.column_stack((np.ones(len(scaled)), scaled, scaled ** 2))
            root_weight = np.sqrt(weights)
            coefficients, _, rank, _ = np.linalg.lstsq(
                design * root_weight[:, None], raw[selected] * root_weight, rcond=None)
            trend[i] = coefficients[0] if rank == 3 else np.nan
        if iteration == ROBUST_ITERATIONS:
            break
        residual = raw - trend
        valid = good & np.isfinite(trend)
        if valid.sum() < MIN_REFERENCE_POINTS:
            break
        center = np.median(residual[valid])
        scale = max(MIN_ROBUST_SCALE_CELSIUS,
                    float(1.4826 * np.median(np.abs(residual[valid] - center))))
        standardized = (residual - center) / (ROBUST_CUTOFF * scale)
        robust = np.where(valid & (np.abs(standardized) < 1), (1 - standardized ** 2) ** 2, 0.)
    return trend


def repair_daily_events(
    minutes: np.ndarray, raw: np.ndarray, trend: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """识别有明显进入及恢复边界的局部异常段，以同日两侧正常点平滑衔接。

    残差及相邻残差变化率同时超过阈值才触发，保留平滑的真实日内温峰。
    最长3小时，缺测、不可靠端点或未闭合事件不修复。返回候选、扣除量与状态。
    """
    time = np.asarray(minutes, dtype=float)
    values = np.asarray(raw, dtype=float)
    reference = np.asarray(trend, dtype=float)
    if values.ndim != 1 or values.shape != reference.shape or time.shape != values.shape:
        raise ValueError("时间、温度和趋势必须是等长一维数组")
    if not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError("趋势时间必须有限且严格递增")
    if np.isinf(values).any() or np.isinf(reference).any():
        raise ValueError("温度和趋势不能包含无穷大")
    candidate = values.copy()
    residual = values - reference
    status = np.where(np.isfinite(residual), "checked", "insufficient_same_day_context").astype(object)
    edge = np.full(len(values), np.nan)
    if len(values) > 1:
        elapsed = np.diff(time)
        valid = (elapsed > 0) & (elapsed <= MAX_GAP_MINUTES)
        edge[1:] = np.where(valid, np.diff(residual) * 10. / elapsed, np.nan)
    i = 1
    while i < len(values):
        if (not np.isfinite(edge[i]) or not np.isfinite(residual[i - 1])
                or abs(residual[i]) <= EVENT_RESIDUAL_CELSIUS
                or abs(edge[i]) <= EVENT_EDGE_CELSIUS_PER_10MIN
                or residual[i] * edge[i] <= 0):
            i += 1
            continue
        direction = np.sign(residual[i])
        start = i
        stop = i + 1
        while stop < len(values):
            if (not np.isfinite(edge[stop]) or time[stop] - time[i] > MAX_EVENT_MINUTES):
                break
            if direction * residual[stop] <= RETURN_TOLERANCE_CELSIUS:
                break
            stop += 1
        closed = (stop < len(values) and np.isfinite(edge[stop])
                  and time[stop] - time[i] <= MAX_EVENT_MINUTES
                  and abs(residual[stop]) <= EVENT_RESIDUAL_CELSIUS
                  and direction * residual[stop] <= RETURN_TOLERANCE_CELSIUS)
        if not closed or abs(residual[i - 1]) > EVENT_RESIDUAL_CELSIUS:
            status[start:stop] = "unclosed_event_kept_raw"
            i = max(i + 1, stop)
            continue
        # 凸起前一刻若有明显反向小凹点，将这个肩部噪点一并接平，
        # 两侧仍落在实际未修复观测上，避免用小凹点把整段曲线压低。
        if (i >= 2 and direction * (values[i - 1] - values[i - 2]) < -SHOULDER_DIP_CELSIUS
                and abs(residual[i - 2]) <= EVENT_RESIDUAL_CELSIUS
                and np.isfinite(edge[i - 1])):
            start = i - 1
        left, right = start - 1, stop
        duration = time[right] - time[left]
        secant = (values[right] - values[left]) / duration
        left_slope = ((values[left] - values[left - 1]) / (time[left] - time[left - 1])
                      if left > 0 and np.isfinite(edge[left]) else secant)
        right_slope = ((values[right + 1] - values[right]) / (time[right + 1] - time[right])
                       if right + 1 < len(values) and np.isfinite(edge[right + 1]) else secant)
        if abs(secant) < 1e-12:
            left_slope = right_slope = 0.
        else:
            left_slope = np.sign(secant) * min(abs(left_slope), 3 * abs(secant)) if left_slope * secant > 0 else 0.
            right_slope = np.sign(secant) * min(abs(right_slope), 3 * abs(secant)) if right_slope * secant > 0 else 0.
            magnitude = np.hypot(left_slope / secant, right_slope / secant)
            if magnitude > 3:
                left_slope *= 3 / magnitude
                right_slope *= 3 / magnitude
        u = (time[start:stop] - time[left]) / duration
        proposed = ((2 * u ** 3 - 3 * u ** 2 + 1) * values[left]
                    + (u ** 3 - 2 * u ** 2 + u) * duration * left_slope
                    + (-2 * u ** 3 + 3 * u ** 2) * values[right]
                    + (u ** 3 - u ** 2) * duration * right_slope)
        if np.isfinite(proposed).all() and ((proposed >= -40) & (proposed <= 50)).all():
            candidate[start:stop] = proposed
            status[start:stop] = "event_bridged"
        i = stop + 1
    return candidate, values - candidate, status


def propose_daily_temperature(part: pd.DataFrame) -> None:
    """处理同日、同设置、连续且无硬异常的实际桶段；不应用旧跳变平移。

    只改变派生候选列，软偏差不再标记待判断或自动排除建模；缺失、物理
    越界和桶冲突仍由原硬异常流程处理。无足够同日参考时直接保留原值。
    """
    raw = part["temperature_raw_celsius"].to_numpy(float)
    times = pd.DatetimeIndex(part["observation_timestamp"])
    hard = (part["temperature_physical_outlier"] | part["temperature_bucket_conflict"]).to_numpy(bool)
    good = np.isfinite(raw) & ~hard
    values = np.where(good, raw, np.nan)
    minutes = (times - times[0]).total_seconds().to_numpy() / 60.
    trend = estimate_local_trend(minutes, values)
    candidate, offset, status = repair_daily_events(minutes, values, trend)
    checked = good & np.isfinite(trend)
    changed = checked & (np.abs(offset) > 1e-10)
    rates = np.full(len(part), np.nan)
    if len(part) > 1:
        elapsed = np.diff(minutes)
        comparable = good[1:] & good[:-1] & (elapsed > 0) & (elapsed <= MAX_GAP_MINUTES)
        edges = np.flatnonzero(comparable) + 1
        rates[edges] = np.abs(raw[edges] - raw[edges - 1]) / elapsed[edges - 1] * 10.
    part["temperature_rate_celsius_per_10min"] = rates
    part["_temperature_candidate_celsius"] = candidate
    part["temperature_daily_trend_celsius"] = trend
    part["temperature_daily_residual_celsius"] = raw - trend
    part["temperature_daily_offset_celsius"] = np.where(checked, offset, np.nan)
    part["temperature_daily_checked"] = checked
    part["temperature_daily_corrected"] = changed
    part["temperature_daily_status"] = status
    part["temperature_repair_method"] = np.where(changed, "daily_trend", "none")
    part["temperature_offset_celsius"] = offset
    part["temperature_offset_corrected"] = changed
    part["temperature_repaired"] = changed
    # 每一连续修正段都有自己的实际起止；正常桶不被延伸标为异常。
    groups = pd.Series(changed, index=part.index).ne(
        pd.Series(changed, index=part.index).shift(fill_value=False)).cumsum()
    for _, event in part.loc[changed].groupby(groups.loc[changed], sort=False):
        start = pd.Timestamp(event["observation_timestamp"].iloc[0])
        end = pd.Timestamp(event["observation_timestamp"].iloc[-1])
        part.loc[event.index, "temperature_event_id"] = f"daily_trend:{start.isoformat()}"
        part.loc[event.index, "temperature_event_start"] = start
        part.loc[event.index, "temperature_event_end"] = end
