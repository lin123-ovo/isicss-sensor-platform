"""Input-only time features and estimator policy for crossline residuals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


RESIDUAL_MODEL_NAME = "hist_gradient_boosting"
RESIDUAL_RANDOM_SEED = 20261013
RESIDUAL_MAX_ITER = 150
RESIDUAL_TIME_FEATURE_VERSION = "crossline_input_time_v1"
RESIDUAL_INPUT_COLUMNS = (
    "tilt1_inline_model",
    "tilt1_crossline_model",
    "temperature_celsius",
)


@dataclass(frozen=True)
class CrosslineResidualConfig:
    """Validated hyperparameters for the deployable residual correction."""

    loss: str
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    correction_scale: float
    max_iter: int = RESIDUAL_MAX_ITER
    random_seed: int = RESIDUAL_RANDOM_SEED

    def validate(self) -> None:
        if self.loss not in {"squared_error", "absolute_error"}:
            raise ValueError(f"不支持的残差损失函数：{self.loss}")
        numeric = (
            self.learning_rate,
            self.l2_regularization,
            self.correction_scale,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("残差模型参数必须为有限值。")
        if self.learning_rate <= 0.0 or self.l2_regularization < 0.0:
            raise ValueError("残差学习率必须大于0，L2不能为负数。")
        if not 0.0 <= self.correction_scale <= 1.0:
            raise ValueError("残差修正强度必须介于0到1。")
        if self.max_leaf_nodes < 2 or self.min_samples_leaf < 1:
            raise ValueError("残差树结构参数无效。")
        if self.max_iter < 1:
            raise ValueError("残差模型迭代次数必须大于0。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CrosslineResidualConfig":
        try:
            config = cls(
                loss=str(value["loss"]),
                learning_rate=float(value["learning_rate"]),
                max_leaf_nodes=int(value["max_leaf_nodes"]),
                min_samples_leaf=int(value["min_samples_leaf"]),
                l2_regularization=float(value["l2_regularization"]),
                correction_scale=float(value["correction_scale"]),
                max_iter=int(value.get("max_iter", RESIDUAL_MAX_ITER)),
                random_seed=int(value.get("random_seed", RESIDUAL_RANDOM_SEED)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("残差模型参数不完整或类型无效。") from error
        config.validate()
        return config


def residual_parameter_candidates() -> Iterable[CrosslineResidualConfig]:
    """Yield the fixed, auditable residual-search policy."""

    for loss, learning_rate, leaves, minimum_leaf, l2, scale in product(
        ("squared_error", "absolute_error"),
        (0.03, 0.06),
        (7, 15),
        (30, 80),
        (0.1, 1.0),
        (0.0, 0.25, 0.5, 0.75, 1.0),
    ):
        yield CrosslineResidualConfig(
            loss=loss,
            learning_rate=learning_rate,
            max_leaf_nodes=leaves,
            min_samples_leaf=minimum_leaf,
            l2_regularization=l2,
            correction_scale=scale,
        )


def make_residual_estimator(
    config: CrosslineResidualConfig,
) -> HistGradientBoostingRegressor:
    """Build one deterministic residual estimator from validated parameters."""

    config.validate()
    return HistGradientBoostingRegressor(
        loss=config.loss,
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=False,
        random_state=config.random_seed,
    )


def build_input_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create causal features from timestamp and sensor-1 inputs only.

    The caller must provide a strictly chronological, unique time series. Exact
    lags remain missing when a ten-minute bucket is absent; the histogram tree
    handles those missing values without inventing sensor observations.
    """

    required = {"timestamp", *RESIDUAL_INPUT_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"横线残差特征缺少字段：{missing}")
    if frame.empty:
        raise ValueError("横线残差特征不能基于空数据。")

    timestamps = pd.to_datetime(frame["timestamp"], errors="raise").reset_index(drop=True)
    if timestamps.duplicated().any():
        raise ValueError("横线残差特征不允许重复时间戳。")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("横线残差特征要求时间严格递增。")

    values = frame.loc[:, list(RESIDUAL_INPUT_COLUMNS)].reset_index(drop=True)
    matrix = values.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("横线残差特征的当前输入包含缺失或非有限值。")

    if "history_segment" in frame and frame["history_segment"].nunique() > 1:
        parts = [build_input_time_features(part.drop(columns="history_segment"))
                 for _, part in frame.groupby("history_segment", sort=False)]
        return pd.concat(parts, ignore_index=True)

    result = pd.DataFrame(index=values.index)
    for column in RESIDUAL_INPUT_COLUMNS:
        current = values[column].astype(float)
        series = pd.Series(current.to_numpy(dtype=float), index=timestamps)
        result[column] = current
        for minutes in (10, 30, 60):
            lag = series.reindex(
                timestamps - pd.Timedelta(minutes=minutes)
            ).to_numpy(dtype=float)
            result[f"{column}_lag_{minutes}m"] = lag
            result[f"{column}_delta_{minutes}m"] = current.to_numpy(dtype=float) - lag
        for minutes in (30, 60):
            rolling = series.rolling(f"{minutes}min", min_periods=1)
            result[f"{column}_mean_{minutes}m"] = rolling.mean().to_numpy(dtype=float)
            result[f"{column}_std_{minutes}m"] = rolling.std(ddof=0).to_numpy(
                dtype=float
            )

    hour = timestamps.dt.hour.to_numpy(dtype=float) + (
        timestamps.dt.minute.to_numpy(dtype=float) / 60.0
    )
    result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["minutes_since_previous_row"] = (
        timestamps.diff().dt.total_seconds().div(60.0).fillna(10.0).to_numpy(float)
    )
    return result


def build_residual_features(
    frame: pd.DataFrame,
    base_crossline_prediction: np.ndarray,
) -> pd.DataFrame:
    """Append the Huber base prediction to the 36 input-side time features."""

    result = build_input_time_features(frame)
    base = np.asarray(base_crossline_prediction, dtype=float).reshape(-1)
    if len(base) != len(result) or not np.isfinite(base).all():
        raise ValueError("横线Huber基础预测的长度或数值无效。")
    result["base_crossline_prediction"] = base
    return result


RESIDUAL_FEATURE_COLUMNS = tuple(
    build_residual_features(
        pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2000-01-01")],
                "tilt1_inline_model": [0.0],
                "tilt1_crossline_model": [0.0],
                "temperature_celsius": [0.0],
            }
        ),
        np.array([0.0]),
    ).columns
)
