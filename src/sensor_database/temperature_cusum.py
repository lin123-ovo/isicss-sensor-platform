"""v4离线温度 CUSUM：历史日内参考、可追溯候选和整段验证。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .temperature_repair_v3 import repair_settings_v3


RULE_VERSION = "temperature_cusum_v4"
APPLY_RULE_VERSION = RULE_VERSION
RULE_VERSIONS = (RULE_VERSION,)
REFERENCE_DAYS = 7
CALIBRATION_DAYS = 7
MIN_REFERENCE_DAYS = 3
MIN_MORNING_POINTS = 9
MIN_CALIBRATION_POINTS = 100
SIGMA_FLOOR = 0.3
K = 0.5
H = 5.0
RATE_COMPARISON_ATOL = 1e-10
MORNING_SLOTS = range(60, 72)  # 10:00—11:50，早于全部被检查点。
SCAN_SLOTS = range(72, 109)  # 12:00—18:00，与离线试验一致。
BOOL_COLUMNS = (
    "temperature_cusum_checked", "temperature_cusum_alarm_positive",
    "temperature_cusum_alarm_negative", "temperature_cusum_outlier",
    "temperature_cusum_candidate_applied",
)
FLOAT_COLUMNS = (
    "temperature_cusum_reference_celsius", "temperature_cusum_sigma_celsius",
    "temperature_cusum_positive", "temperature_cusum_negative",
    "temperature_cusum_candidate_celsius", "temperature_cusum_offset_celsius",
)
AUDIT_COLUMNS = (*BOOL_COLUMNS, *FLOAT_COLUMNS, "temperature_cusum_status",
                 "temperature_cusum_reference_count", "temperature_cusum_event_id")


def cusum_settings(rule_version: str) -> dict[str, float | int]:
    """v4签名覆盖继承的v3设置和全部新增计算参数。"""
    if rule_version not in RULE_VERSIONS:
        raise ValueError("不支持的CUSUM版本")
    return {**repair_settings_v3(), "cusum_reference_days": REFERENCE_DAYS,
            "cusum_calibration_days": CALIBRATION_DAYS, "cusum_min_reference_days": MIN_REFERENCE_DAYS,
            "cusum_min_morning_points": MIN_MORNING_POINTS,
            "cusum_min_calibration_points": MIN_CALIBRATION_POINTS,
            "cusum_sigma_floor_celsius": SIGMA_FLOOR, "cusum_mad_factor": 1.4826,
            "cusum_k": K, "cusum_h": H, "cusum_morning_start_slot": 60,
            "cusum_morning_stop_slot_exclusive": 72, "cusum_scan_start_slot": 72,
            "cusum_scan_stop_slot_exclusive": 109, "cusum_reference_smooth_window": 3,
            "cusum_candidate_median_window": 5, "cusum_candidate_mean_window": 3,
            "cusum_max_candidate_rate_per_10min": 2.0, "cusum_missing_resets": 1,
            "cusum_rate_comparison_atol": RATE_COMPARISON_ATOL,
            "cusum_positive_candidates_only": 1,
            "cusum_apply_candidate": int(rule_version == APPLY_RULE_VERSION)}


def initialize_cusum_audit(frame: pd.DataFrame) -> None:
    """给旧版本和空结果提供稳定的可选审计列，不改变旧版本数值。"""
    for column in BOOL_COLUMNS:
        frame[column] = False
    for column in FLOAT_COLUMNS:
        frame[column] = np.nan
    frame["temperature_cusum_status"] = "disabled"
    frame["temperature_cusum_reference_count"] = 0
    frame["temperature_cusum_event_id"] = ""


def compute_cusum(residuals: np.ndarray, sigma: float) -> pd.DataFrame:
    """双侧标准化递推；NaN/Inf切断上下文，回溯只标至最后报警点。

    输入为完整10分钟刻度上的一日检查窗。返回在线报警及离线回溯标记，
    不修改输入。参考偏差报警不等同于采集故障。
    """
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("CUSUM残差必须为一维，尺度必须为有限正数")
    high = np.full(len(values), np.nan)
    low = high.copy()
    positive_event = np.zeros(len(values), dtype=bool)
    negative_event = positive_event.copy()
    positive = negative = 0.0
    positive_start = negative_start = 0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            positive = negative = 0.0
            positive_start = negative_start = i + 1
            continue
        if positive == 0:
            positive_start = i
        if negative == 0:
            negative_start = i
        positive = max(0.0, positive + value / sigma - K)
        negative = max(0.0, negative - value / sigma - K)
        high[i], low[i] = positive, negative
        if positive >= H:
            positive_event[positive_start:i + 1] = True
        if negative >= H:
            negative_event[negative_start:i + 1] = True
    return pd.DataFrame({"positive": high, "negative": low,
                         "alarm_positive": high >= H, "alarm_negative": low >= H,
                         "positive_event": positive_event, "negative_event": negative_event})


def estimate_positive_offsets(residuals: pd.Series, event: pd.Series) -> pd.Series:
    """对连续有效上下文独立估计偏移，禁止跨NaN、关闭桶或已有事件平滑。"""
    estimate = pd.Series(0.0, index=residuals.index)
    valid = residuals.notna()
    groups = valid.ne(valid.shift(fill_value=False)).cumsum()
    for _, part in residuals.loc[valid].groupby(groups.loc[valid], sort=False):
        smooth = part.rolling(5, center=True, min_periods=3).median()
        smooth = smooth.rolling(3, center=True, min_periods=2).mean().clip(lower=0)
        selected = event.loc[part.index] & part.gt(0) & smooth.gt(0)
        estimate.loc[part.index[selected]] = smooth.loc[selected]
    return estimate


def add_cusum_candidates(bucketed: pd.DataFrame) -> None:
    """在v3跳变候选生成后运行v4；仅修改内存派生结果，原温度只读。

    历史参考只用开启v4且未被跳变修复/待判断、未越界、无冲突的原观测。
    检测每个完整十分钟窗，缺桶/开关/版本/硬异常/已有事件均切断累积与平滑。
    应用已通过物理和实际边界率检查的正向候选，其余报警保留原温度并要求复核。
    """
    enabled = bucketed["temperature_jump_check_enabled"] & bucketed["temperature_rule_version"].isin(RULE_VERSIONS)
    if not enabled.any():
        return
    clock = pd.to_datetime(bucketed["timestamp"])
    dates = clock.dt.normalize()
    slots = clock.dt.hour * 6 + clock.dt.minute // 10
    blocked = (bucketed["temperature_physical_outlier"] | bucketed["temperature_bucket_conflict"]
               | bucketed["temperature_raw_celsius"].isna() | bucketed["temperature_review_required"]
               | bucketed["temperature_repaired"] | bucketed["temperature_jump_outlier"])
    eligible = enabled & ~blocked
    source = bucketed.loc[eligible, ["temperature_raw_celsius"]].assign(day=dates[eligible], slot=slots[eligible])
    matrix = source.pivot(index="day", columns="slot", values="temperature_raw_celsius").reindex(columns=range(144))
    cache: dict[pd.Timestamp, tuple[pd.Series, pd.Series, str]] = {}

    def reference(day: pd.Timestamp) -> tuple[pd.Series, pd.Series, str]:
        if day in cache:
            return cache[day]
        history = matrix.reindex(pd.date_range(day - pd.Timedelta(days=REFERENCE_DAYS), periods=REFERENCE_DAYS))
        counts = history.count()
        baseline = history.median().where(counts.ge(MIN_REFERENCE_DAYS))
        baseline = baseline.rolling(3, center=True, min_periods=1).median().where(baseline.notna())
        today = matrix.reindex([day]).iloc[0]
        difference = (today - baseline).reindex(MORNING_SLOTS)
        reason = "ready" if difference.notna().sum() >= MIN_MORNING_POINTS else "insufficient_morning_reference"
        baseline = baseline + float(difference.median()) if reason == "ready" else baseline * np.nan
        cache[day] = baseline, counts, reason
        return cache[day]

    bucketed.loc[enabled, "temperature_cusum_status"] = "outside_scan_window"
    for day in dates.loc[enabled].drop_duplicates():
        positions = bucketed.index[enabled & dates.eq(day) & slots.isin(SCAN_SLOTS)]
        if not len(positions):
            continue
        baseline, counts, reason = reference(day)
        bucketed.loc[positions, "temperature_cusum_reference_count"] = counts.reindex(slots.loc[positions]).to_numpy()
        bucketed.loc[positions, "temperature_cusum_reference_celsius"] = baseline.reindex(slots.loc[positions]).to_numpy()
        if reason != "ready":
            bucketed.loc[positions, "temperature_cusum_status"] = reason
            continue
        calibration = []
        for prior in pd.date_range(day - pd.Timedelta(days=CALIBRATION_DAYS), periods=CALIBRATION_DAYS):
            prior_baseline, _, prior_reason = reference(prior)
            if prior_reason == "ready":
                residual = (matrix.reindex([prior]).iloc[0] - prior_baseline).reindex(SCAN_SLOTS)
                calibration.extend(residual.dropna().tolist())
        if len(calibration) < MIN_CALIBRATION_POINTS:
            bucketed.loc[positions, "temperature_cusum_status"] = "insufficient_calibration"
            continue
        values = np.asarray(calibration)
        sigma = max(SIGMA_FLOOR, float(1.4826 * np.median(np.abs(values - np.median(values)))))
        day_frame = bucketed.loc[positions].copy()
        day_frame.index = slots.loc[positions].to_numpy()
        residual = (day_frame["temperature_raw_celsius"] - baseline).reindex(SCAN_SLOTS)
        residual.loc[slots.loc[positions[blocked.loc[positions]]].to_numpy()] = np.nan
        # 版本和开关变化即使没有缺桶也必须清零，避免不同策略共享累积量。
        results = []
        version_grid = day_frame["temperature_rule_version"].reindex(SCAN_SLOTS)
        groups = version_grid.ne(version_grid.shift()).cumsum()
        for _, part in residual.groupby(groups, sort=False):
            score = compute_cusum(part.to_numpy(), sigma)
            score.index = part.index
            score["offset"] = estimate_positive_offsets(part, score.positive_event)
            results.append(score)
        scores = pd.concat(results).sort_index()
        mapped = scores.reindex(day_frame.index)
        bucketed.loc[positions, "temperature_cusum_sigma_celsius"] = sigma
        for suffix, column in (("positive", "positive"), ("negative", "negative"),
                               ("alarm_positive", "alarm_positive"), ("alarm_negative", "alarm_negative")):
            bucketed.loc[positions, f"temperature_cusum_{suffix}"] = mapped[column].to_numpy()
        checked = eligible.loc[positions].to_numpy() & residual.reindex(day_frame.index).notna().to_numpy()
        bucketed.loc[positions, "temperature_cusum_checked"] = checked
        bucketed.loc[positions, "temperature_cusum_status"] = np.where(checked, "checked", "blocked_or_missing_reference")
        suspect = mapped.positive_event.to_numpy() | mapped.negative_event.to_numpy()
        affected = positions[suspect & checked]
        bucketed.loc[affected, "temperature_cusum_outlier"] = True
        bucketed.loc[affected, "temperature_review_required"] = True
        positive = mapped.positive_event.to_numpy() & ~mapped.negative_event.to_numpy()
        for index in affected:
            slot = int(slots.loc[index])
            direction = "positive" if scores.loc[slot, "positive_event"] else "negative"
            bucketed.loc[index, "temperature_review_reason"] = f"cusum_{direction}_deviation"
            bucketed.loc[index, "temperature_repair_method"] = "review_required"
            bucketed.loc[index, "temperature_cusum_event_id"] = f"cusum:{day.date()}:{direction}"
        for _, event in bucketed.loc[affected].groupby("temperature_cusum_event_id", sort=False):
            bucketed.loc[event.index, "temperature_event_id"] = event["temperature_cusum_event_id"]
            bucketed.loc[event.index, "temperature_event_start"] = event["observation_timestamp"].min()
            bucketed.loc[event.index, "temperature_event_end"] = event["observation_timestamp"].max()
        offsets = mapped.offset.to_numpy()
        candidate_indexes = positions[checked & positive & (offsets > 0)]
        candidate = bucketed.loc[candidate_indexes, "temperature_raw_celsius"] - mapped.loc[
            slots.loc[candidate_indexes], "offset"].to_numpy()
        bucketed.loc[candidate_indexes, "temperature_cusum_candidate_celsius"] = candidate
        bucketed.loc[candidate_indexes, "temperature_cusum_offset_celsius"] = offsets[checked & positive & (offsets > 0)]
        # 按连续候选段原子验证，失败不应用部分修复。使用实际时间而非舍入桶时间。
        candidate_mask = bucketed.index.isin(candidate_indexes)
        run_ids = pd.Series(candidate_mask, index=bucketed.index).ne(
            pd.Series(candidate_mask, index=bucketed.index).shift(fill_value=False)).cumsum()
        for _, event in bucketed.loc[candidate_indexes].groupby(run_ids.loc[candidate_indexes], sort=False):
            indexes = event.index
            left = bucketed.index.get_loc(indexes[0]) - 1
            right = bucketed.index.get_loc(indexes[-1]) + 1
            usable = left >= 0 and right < len(bucketed)
            if usable:
                edges = bucketed.iloc[left:right + 1].copy()
                usable = (edges["temperature_jump_check_enabled"].all()
                          and edges["temperature_rule_version"].nunique() == 1
                          and not bool(blocked.loc[edges.index].any())
                          and edges["timestamp"].dt.normalize().eq(day).all()
                          and edges["timestamp"].diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all())
                proposed = edges["_temperature_candidate_celsius"].copy()
                proposed.loc[indexes] = edges.loc[indexes, "temperature_cusum_candidate_celsius"]
                elapsed = pd.to_datetime(edges["observation_timestamp"]).diff().dt.total_seconds().div(600)
                rates = proposed.diff().abs().div(elapsed)
                usable = bool(usable and proposed.between(-40, 50).all() and elapsed.iloc[1:].gt(0).all()
                              and rates.iloc[1:].le(2.0 + RATE_COMPARISON_ATOL).all())
            if not usable:
                bucketed.loc[indexes, "temperature_cusum_status"] = "candidate_boundary_rejected"
                continue
            apply = bucketed.loc[indexes, "temperature_rule_version"].eq(APPLY_RULE_VERSION)
            accepted = indexes[apply]
            bucketed.loc[accepted, "_temperature_candidate_celsius"] = bucketed.loc[accepted, "temperature_cusum_candidate_celsius"]
            bucketed.loc[accepted, "temperature_offset_celsius"] = bucketed.loc[accepted, "temperature_cusum_offset_celsius"]
            bucketed.loc[accepted, "temperature_review_required"] = False
            bucketed.loc[accepted, "temperature_review_reason"] = ""
            bucketed.loc[accepted, "temperature_repair_method"] = "cusum_offset"
            bucketed.loc[accepted, "temperature_cusum_candidate_applied"] = True
