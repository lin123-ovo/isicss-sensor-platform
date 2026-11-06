"""温度突变筛查：检查实际时间变化率，并追踪突变后的整段温度偏移。

这是保守的数据使用规则，不是气象物理定律，也不能证明设备发生故障。
只在内存表中增加标记，原始温度中位数始终保留。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .temperature_repair import (
    REPAIR_COLUMNS, initialize_repair_columns, propose_temperature_repairs, repair_settings,
)
from .temperature_repair_v3 import AUDIT_COLUMNS, initialize_v3_audit, propose_temperature_repairs_v3, repair_settings_v3
from .temperature_cusum import (
    RULE_VERSIONS as CUSUM_VERSIONS, add_cusum_candidates, cusum_settings, initialize_cusum_audit,
)
from .temperature_daily_trend import (
    RULE_VERSION as DAILY_VERSION, AUDIT_COLUMNS as DAILY_AUDIT_COLUMNS,
    daily_trend_settings, initialize_daily_audit, propose_daily_temperature,
)


TEMPERATURE_RATE_LIMIT = 2.0  # 严格超过 2℃/10分钟才触发，恰好等于不触发。
TEMPERATURE_RETURN_TOLERANCE = 2.0  # 回到突变前温度的同侧 2℃范围内视为结束。
RATE_INTERVAL_MINUTES = 10.0
MAX_OBSERVATION_GAP_MINUTES = 20.0
MAX_SPIKE_BUCKETS = 2


def temperature_check_settings(rule_version: str = "temperature_jump_v1") -> dict[str, float | int]:
    """返回固定规则，写入独立副本的策略文件和策略指纹。"""

    if rule_version == DAILY_VERSION:
        return daily_trend_settings()
    if rule_version in CUSUM_VERSIONS:
        return cusum_settings(rule_version)
    if rule_version == "temperature_jump_v3":
        return repair_settings_v3()
    if rule_version == "temperature_jump_v2":
        return repair_settings()
    return {
        "rate_limit_celsius_per_10min": TEMPERATURE_RATE_LIMIT,
        "return_tolerance_celsius": TEMPERATURE_RETURN_TOLERANCE,
        "max_observation_gap_minutes": MAX_OBSERVATION_GAP_MINUTES,
        "max_spike_buckets": MAX_SPIKE_BUCKETS,
    }


def _add_temperature_jump_flags(bucketed: pd.DataFrame) -> None:
    """原地添加变化率、触发边界、可疑段和禁止修复段四列。

    输入是按时间排序、已有 segment_id 的实际桶表。每桶用源时间戳中位数
    作为实际代表时间；分母不用舍入后的桶时间。物理越界和桶内冲突点不能作参考。
    长缺口及不同连续段之间不比较；尚未恢复的偏移保守隔离到本段末尾。
    """

    rates = pd.Series(np.nan, index=bucketed.index, dtype=float)
    triggers = pd.Series(False, index=bucketed.index)
    suspicious = pd.Series(False, index=bucketed.index)
    isolated = pd.Series(False, index=bucketed.index)

    for _, segment in bucketed.groupby("segment_id", sort=False):
        values = segment["temperature_raw_celsius"].to_numpy(dtype=float)
        times = pd.DatetimeIndex(segment["observation_timestamp"])
        bucket_times = pd.DatetimeIndex(segment["timestamp"])
        indexes = segment.index.to_numpy()
        hard = (
            segment["temperature_physical_outlier"]
            | segment["temperature_bucket_conflict"]
        ).to_numpy(dtype=bool)
        reliable = np.isfinite(values) & ~hard
        # 左端若已经被 Hampel 判为异常，其后下降可能只是恢复，不能把它当基准。
        hampel = segment["temperature_hampel_outlier"].to_numpy(dtype=bool)
        elapsed = np.full(len(segment), np.nan)
        if len(segment) > 1:
            elapsed[1:] = (times[1:] - times[:-1]).total_seconds() / 60.0

        # 只比较相邻可靠实际桶，不用补缺点，也不跨过大于 20 分钟的实际缺口。
        comparable = (
            reliable[1:] & reliable[:-1] & ~hampel[:-1]
            & (elapsed[1:] > 0)
            & (elapsed[1:] <= MAX_OBSERVATION_GAP_MINUTES)
        )
        local_rates = np.full(len(segment), np.nan)
        comparable_positions = np.flatnonzero(comparable) + 1
        local_rates[comparable_positions] = (
            np.abs(values[comparable_positions] - values[comparable_positions - 1])
            / elapsed[comparable_positions] * RATE_INTERVAL_MINUTES
        )
        local_triggers = (local_rates > TEMPERATURE_RATE_LIMIT) & ~np.isclose(
            local_rates, TEMPERATURE_RATE_LIMIT, rtol=0.0, atol=1e-10
        )
        rates.loc[indexes] = local_rates
        triggers.loc[indexes] = local_triggers

        position = 1
        while position < len(segment):
            if not local_triggers[position]:
                position += 1
                continue

            # 进入偏移段后固定突变前的参考值，不能因平台内变化缓慢就自动放行。
            start = position
            baseline = values[start - 1]
            direction = 1.0 if values[start] > baseline else -1.0
            stop = start + 1
            while stop < len(segment):
                if (
                    elapsed[stop] <= 0
                    or elapsed[stop] > MAX_OBSERVATION_GAP_MINUTES
                ):
                    break
                if reliable[stop] and (
                    direction * (values[stop] - baseline)
                    <= TEMPERATURE_RETURN_TOLERANCE
                ):
                    break
                stop += 1

            # stop 是首个恢复点，不包括在偏移段里；到文件/连续段末尾则没有恢复点。
            recovered = (
                stop < len(segment)
                and reliable[stop]
                and 0 < elapsed[stop] <= MAX_OBSERVATION_GAP_MINUTES
                and direction * (values[stop] - baseline)
                <= TEMPERATURE_RETURN_TOLERANCE
            )
            affected = indexes[start:stop]
            suspicious.loc[affected] = True
            # 即使只有两个实际桶，若中间缺采样，也不允许把它当作完整短尖峰。
            end = stop + 1 if recovered else stop
            uninterrupted = np.all(
                (bucket_times[start:end] - bucket_times[start - 1:end - 1])
                == pd.Timedelta(minutes=10)
            )
            if stop - start > MAX_SPIKE_BUCKETS or not recovered or not uninterrupted:
                isolated.loc[affected] = True

            # 恢复边界的反向跳变属于同一事件，不能又把恢复后的正常平台判为新异常。
            # 若因实际长缺口退出，则从缺口后的点开始新检查。
            position = stop + 1 if recovered else stop

    bucketed["temperature_rate_celsius_per_10min"] = rates
    bucketed["temperature_jump_trigger"] = triggers
    bucketed["temperature_jump_outlier"] = suspicious
    bucketed["temperature_jump_isolated"] = isolated


JUMP_FLAGS = (
    "temperature_jump_trigger", "temperature_jump_outlier", "temperature_jump_isolated"
)


def _enabled_runs(frame: pd.DataFrame) -> pd.Series:
    """关闭桶以及原连续段边界均切断新检查的上下文。"""

    enabled = frame["temperature_jump_check_enabled"].fillna(False).astype(bool)
    version = frame["temperature_rule_version"].fillna("temperature_jump_v1")
    clock = pd.DatetimeIndex(frame["timestamp"] if "timestamp" in frame else frame.index)
    observation = pd.to_datetime(frame["observation_timestamp"])
    repair_versions = ("temperature_jump_v2", "temperature_jump_v3", *CUSUM_VERSIONS, DAILY_VERSION)
    v2_edge = version.isin(repair_versions) | version.shift().isin(repair_versions)
    clock_gap = pd.Series(clock, index=frame.index).diff().ne(pd.Timedelta(minutes=10))
    elapsed = observation.diff().dt.total_seconds().div(60)
    broken = v2_edge & (clock_gap | elapsed.le(0) | elapsed.gt(MAX_OBSERVATION_GAP_MINUTES))
    v3_edge = version.isin(("temperature_jump_v3", *CUSUM_VERSIONS, DAILY_VERSION)) | version.shift().isin(
        ("temperature_jump_v3", *CUSUM_VERSIONS, DAILY_VERSION))
    dates = pd.Series(clock, index=frame.index).dt.normalize()
    hard = (
        frame["temperature_physical_outlier"].fillna(False).astype(bool)
        | frame["temperature_bucket_conflict"].fillna(False).astype(bool)
        | frame["temperature_raw_celsius"].isna()
    )
    broken |= v3_edge & (dates.ne(dates.shift()) | hard | hard.shift(fill_value=False))
    return (
        enabled.ne(enabled.shift()) | frame["segment_id"].ne(frame["segment_id"].shift())
        | version.ne(version.shift()) | broken
    ).cumsum()


def add_temperature_jump_flags(bucketed: pd.DataFrame) -> None:
    """仅检查连续开启的桶；关闭桶不参与新规则，仍保留原清洗行为。"""

    for column in JUMP_FLAGS:
        bucketed[column] = False
    bucketed["temperature_rate_celsius_per_10min"] = np.nan
    initialize_repair_columns(bucketed)
    initialize_v3_audit(bucketed)
    initialize_cusum_audit(bucketed)
    initialize_daily_audit(bucketed)
    runs = _enabled_runs(bucketed)
    enabled = bucketed["temperature_jump_check_enabled"].astype(bool)
    for _, part in bucketed.loc[enabled].groupby(runs.loc[enabled], sort=False):
        part = part.copy()
        if part["temperature_rule_version"].iloc[0] == DAILY_VERSION:
            propose_daily_temperature(part)
        elif part["temperature_rule_version"].iloc[0] in ("temperature_jump_v3", *CUSUM_VERSIONS):
            propose_temperature_repairs_v3(part)
        elif part["temperature_rule_version"].iloc[0] == "temperature_jump_v2":
            propose_temperature_repairs(part)
        else:
            _add_temperature_jump_flags(part)
        for column in (*JUMP_FLAGS, "temperature_rate_celsius_per_10min", *REPAIR_COLUMNS, *AUDIT_COLUMNS,
                       *DAILY_AUDIT_COLUMNS):
            bucketed.loc[part.index, column] = part[column]
    add_cusum_candidates(bucketed)


def repair_enabled_temperature(completed: pd.DataFrame) -> None:
    """覆盖开启桶的温度清洗值；关闭桶的旧算法结果完全保留。

    输入以桶时间为索引。先按全部异常连续长度分组，再检查硬异常和缺口，
    防止隔离段被拆开后又作为1–2点尖峰修复。
    """

    enabled = completed["temperature_jump_check_enabled"].fillna(False).astype(bool)
    completed["temperature_jump_check_enabled"] = enabled
    for column in JUMP_FLAGS:
        completed[column] = completed[column].astype("boolean").fillna(False).astype(bool)
    runs = _enabled_runs(completed)
    for _, part in completed.loc[enabled].groupby(runs.loc[enabled], sort=False):
        if part["temperature_rule_version"].iloc[0] in (
            "temperature_jump_v2", "temperature_jump_v3", *CUSUM_VERSIONS, DAILY_VERSION):
            continue
        part = part.copy()
        anomaly = part["temperature_outlier"] | part["temperature_jump_outlier"]
        hard = (
            part["temperature_physical_outlier"] | part["temperature_bucket_conflict"]
            | part["temperature_jump_isolated"]
        )
        part["temperature_outlier"] = anomaly
        part["temperature_celsius"] = part["temperature_raw_celsius"].mask(anomaly)
        part["temperature_imputed"] = False
        run_ids = anomaly.ne(anomaly.shift(fill_value=False)).cumsum()
        lengths = anomaly.groupby(run_ids).transform("sum").where(anomaly, 0).astype(int)
        part["temperature_outlier_run_length"] = lengths
        for run_id in run_ids[anomaly & lengths.le(MAX_SPIKE_BUCKETS)].unique():
            positions = np.flatnonzero((run_ids == run_id).to_numpy())
            before, after = int(positions[0]) - 1, int(positions[-1]) + 1
            if before < 0 or after >= len(part) or hard.iloc[positions].any():
                continue
            if anomaly.iloc[before] or anomaly.iloc[after]:
                continue
            # 必须是完整连续刻度；源时间差另外用于插值比例。
            clock = part.index[before:after + 1]
            if not ((clock[1:] - clock[:-1]) == pd.Timedelta(minutes=10)).all():
                continue
            left, right = part["temperature_raw_celsius"].iloc[[before, after]]
            times = pd.DatetimeIndex(part["observation_timestamp"].iloc[before:after + 1])
            minutes = (times[1:] - times[:-1]).total_seconds() / 60.0
            if not np.isfinite([left, right]).all() or times.isna().any():
                continue
            if not ((minutes > 0) & (minutes <= MAX_OBSERVATION_GAP_MINUTES)).all():
                continue
            duration = times[-1] - times[0]
            if abs(right - left) / (duration.total_seconds() / 600.0) > TEMPERATURE_RATE_LIMIT:
                continue
            for position in positions:
                fraction = (part["observation_timestamp"].iloc[position] - times[0]) / duration
                part.iloc[position, part.columns.get_loc("temperature_celsius")] = (
                    left + (right - left) * fraction
                )
                part.iloc[position, part.columns.get_loc("temperature_imputed")] = True

        part["temperature_unresolved"] = anomaly & part["temperature_celsius"].isna()
        reasons = pd.Series("", index=part.index, dtype=object)
        for flag, label in (
            ("temperature_physical_outlier", "physical_range"),
            ("temperature_bucket_conflict", "bucket_conflict"),
            ("temperature_hampel_outlier", "hampel"),
            ("temperature_jump_outlier", "temperature_jump"),
            ("temperature_jump_isolated", "temperature_jump_isolated"),
        ):
            mask = part[flag]
            reasons.loc[mask] = reasons.loc[mask].map(
                lambda value: f"{value}|{label}" if value else label
            )
        part["temperature_reason"] = reasons
        for column in (
            "temperature_outlier", "temperature_outlier_run_length", "temperature_celsius",
            "temperature_imputed", "temperature_unresolved", "temperature_reason",
        ):
            completed.loc[part.index, column] = part[column]
