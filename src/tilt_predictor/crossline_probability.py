"""Frozen crossline probability ensemble with causal thermal history."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sensor_app.model_assets import (
    TABICL_CHECKPOINT_NAME as CHECKPOINT_NAME,
    TABICL_CHECKPOINT_SHA256 as CHECKPOINT_SHA256,
    verified_tabicl_checkpoint,
)


MODEL_NAME = "tabicl_catboost"
FEATURE_VERSION = "tilt_temperature_probability_v1"
INPUT_COLUMNS = ("tilt1_inline_model", "tilt1_crossline_model", "temperature_celsius")
TEMP = INPUT_COLUMNS[2]
FEATURE_COLUMNS = (
    *INPUT_COLUMNS,
    *(TEMP + suffix for suffix in ("_missing", "_held", "_physical_bad")),
    *(f"{TEMP}_{kind}{minutes}" for minutes in (10, 30, 60, 180) for kind in ("lag", "diff")),
    *(f"{TEMP}_{kind}{minutes}" for minutes in (30, 60, 180)
      for kind in ("past_mean", "past_std", "coverage")),
    "temperature_trend_sign60", "temperature_trend_sign180",
)
VALUES = np.round(np.arange(-.10, -.019, .01), 2)
FROZEN_PARAMETERS = {
    "model": MODEL_NAME, "recipe_version": 1, "n_estimators": 4,
    "tabicl_weight": .5, "iterations": 300, "depth": 5,
    "learning_rate": .03, "l2_leaf_reg": 3., "random_seed": 20261022,
}


def normalize_parameters(document: Mapping[str, object]) -> dict[str, object]:
    """Reject changes to the explicitly selected, fixed historical recipe."""
    if not isinstance(document, Mapping) or set(document) != set(FROZEN_PARAMETERS):
        raise ValueError("横线概率融合参数必须包含完整的固定方案配置。")
    for name, expected in FROZEN_PARAMETERS.items():
        actual = document[name]
        if isinstance(actual, bool) or actual != expected:
            raise ValueError(f"横线概率融合的 {name} 固定为 {expected}。")
    return dict(FROZEN_PARAMETERS)


def normalize_probabilities(p: np.ndarray, classes: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); classes = np.asarray(classes)
    if (p.ndim != 2 or p.shape[1] != len(classes) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6)
            or not np.isin(classes, np.arange(9)).all() or len(np.unique(classes)) != len(classes)):
        raise ValueError("九档模型概率或类别无效。")
    full = np.zeros((len(p), 9)); full[:, classes.astype(int)] = p
    full = np.maximum(full, 1e-12)
    return full/full.sum(axis=1, keepdims=True)


def select_level(p: np.ndarray) -> np.ndarray:
    """Apply true-level relative costs once, after probability fusion."""
    p = normalize_probabilities(p, np.arange(9))
    cost = abs(VALUES[:, None]-VALUES[None, :])/abs(VALUES[:, None])
    return VALUES[np.argmin(p @ cost, axis=1)]


def input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", *INPUT_COLUMNS]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError("横线概率模型需要非空的时间、1号两轴和温度。")
    result = frame.loc[:, [*required, *(['history_barrier'] if 'history_barrier' in frame else [])]].copy()
    if 'history_barrier' in frame:
        for column, prefix in zip(INPUT_COLUMNS, ('tilt1_inline', 'tilt1_crossline', 'temperature')):
            flags = [f'{prefix}_unresolved']
            if column == TEMP:
                flags.append('temperature_review_required')
            for flag in flags:
                if flag in frame:
                    result.loc[frame[flag].fillna(False).astype(bool), column] = np.nan
    result["timestamp"] = pd.to_datetime(result.timestamp, errors="raise")
    if result.timestamp.isna().any() or result.timestamp.dt.tz is not None:
        raise ValueError("横线概率模型需要有效的本地无时区时间。")
    if result.timestamp.duplicated().any() or not result.timestamp.equals(result.timestamp.dt.round('10min')):
        raise ValueError("横线概率模型需要唯一、已对齐到10分钟的时间桶。")
    result.loc[:, INPUT_COLUMNS] = result.loc[:, INPUT_COLUMNS].apply(pd.to_numeric, errors="raise")
    if 'history_barrier' in result:
        result['history_barrier'] = result['history_barrier'].fillna(False).astype(bool)
        result.loc[result.history_barrier, list(INPUT_COLUMNS)] = np.nan
    if np.isinf(result.loc[:, INPUT_COLUMNS].to_numpy(float)).any():
        raise ValueError("横线概率模型输入包含无穷大。")
    return result.sort_values("timestamp", kind="stable").reset_index(drop=True)


def replay_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply current zero rules to old archive history without changing its state."""
    from sensor_database.processing import current_processing_version

    result = input_frame(frame)
    if current_processing_version() in ('final_v2', 'final_v3') and result[TEMP].eq(0).any():
        result['history_barrier'] = result.get('history_barrier', False) | result[TEMP].eq(0)
        result.loc[result.history_barrier, list(INPUT_COLUMNS)] = np.nan
    return result


def causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the frozen25-column recipe; target columns are never read."""
    r = replay_input_frame(frame).set_index("timestamp")
    steps = int((r.index[-1]-r.index[0])/pd.Timedelta(minutes=10))+1
    if steps > 600_000:
        raise ValueError("横线概率模型的时间范围过大，请缩小请求时间段。")
    grid = pd.date_range(r.index[0], r.index[-1], freq="10min")
    z = r.reindex(grid)
    observed = pd.Series(grid.isin(r.index), index=grid)
    last_observed = pd.Series(grid.where(observed), index=grid).ffill()
    age = (pd.Series(grid, index=grid)-last_observed).dt.total_seconds()/60
    boundary = (age > 20) | age.shift(fill_value=np.inf).gt(20)
    barriers = z.get('history_barrier', pd.Series(False, index=grid)).fillna(False).astype(bool)
    boundary |= barriers | barriers.shift(fill_value=False)
    runs = boundary.cumsum()
    for name, lo, hi in [(INPUT_COLUMNS[0], -10, 10), (INPUT_COLUMNS[1], -10, 10), (TEMP, -40, 50)]:
        raw = z[name].to_numpy(float); values = np.full(len(z), np.nan)
        bad = np.isfinite(raw) & ((raw < lo) | (raw > hi)); held = np.zeros(len(z), bool)
        last = np.nan; last_time = None
        for i,t in enumerate(grid):
            if barriers.iloc[i]:
                last, last_time = np.nan, None
                continue
            if np.isfinite(raw[i]) and lo <= raw[i] <= hi:
                last, last_time, values[i] = raw[i], t, raw[i]
            elif last_time is not None and t-last_time <= pd.Timedelta(minutes=20):
                values[i] = last; held[i] = True
        z[name] = values
        if name == TEMP:
            z[name+'_missing'] = (~np.isfinite(values)).astype(float)
            z[name+'_held'] = held.astype(float); z[name+'_physical_bad'] = bad.astype(float)
    for minutes in (10, 30, 60, 180):
        lag = z[TEMP].groupby(runs, sort=False).shift(minutes//10)
        z[f'{TEMP}_lag{minutes}'] = lag; z[f'{TEMP}_diff{minutes}'] = z[TEMP]-lag
    past = z[TEMP].groupby(runs, sort=False).shift(1)
    for minutes in (30, 60, 180):
        k = minutes//10
        z[f'{TEMP}_past_mean{minutes}'] = past.groupby(runs, sort=False).transform(lambda s: s.rolling(k, min_periods=1).mean())
        z[f'{TEMP}_past_std{minutes}'] = past.groupby(runs, sort=False).transform(lambda s: s.rolling(k, min_periods=1).std(ddof=0))
        z[f'{TEMP}_coverage{minutes}'] = past.notna().astype(float).groupby(runs, sort=False).transform(lambda s: s.rolling(k, min_periods=1).sum())/k
    for minutes in (60, 180):
        z[f'temperature_trend_sign{minutes}'] = np.sign(z[f'{TEMP}_diff{minutes}'])
    return z.loc[r.index, list(FEATURE_COLUMNS)]


def features_for(frame: pd.DataFrame, context: pd.DataFrame | None = None,
                 seed_history: pd.DataFrame | None = None) -> np.ndarray:
    """Resolve causal history with request inputs taking precedence at equal times."""
    request = input_frame(frame)
    parts = [input_frame(p) for p in (seed_history, context, frame) if p is not None and not p.empty]
    combined = pd.concat(parts, ignore_index=True).drop_duplicates('timestamp', keep='last')
    combined = combined[combined.timestamp <= request.timestamp.max()]
    result = causal_features(combined)
    # Preserve the original requested row order, including a nonchronological view.
    return result.reindex(pd.to_datetime(frame.timestamp)).to_numpy(float)


def checkpoint_path() -> Path:
    root = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
    return verified_tabicl_checkpoint(root)


def make_classifier():
    import torch
    from tabicl import TabICLClassifier

    torch.set_num_threads(2)
    return TabICLClassifier(
        n_estimators=4, norm_methods=['none', 'power'], average_logits=True,
        softmax_temperature=1., kv_cache=True, device='mps' if torch.backends.mps.is_available() else 'cpu',
        use_amp=False, use_fa3=False, offload_mode=False, n_jobs=2, batch_size=4,
        random_state=20261022, model_path=str(checkpoint_path()), allow_auto_download=False, verbose=False,
    )


class CrosslineProbabilityRegressor(RegressorMixin, BaseEstimator):
    """Fixed four-member TabICLv2 and CatBoost probability ensemble."""

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if x.ndim != 2 or x.shape[1] != 25 or len(x) != len(y) or len(y) < 300 or np.isinf(x).any():
            raise ValueError("横线概率融合需要至少300条、25列有效时序训练样本。")
        classes = np.argmin(abs(y[:, None]-VALUES), axis=1)
        if not np.isfinite(y).all() or not np.allclose(y, VALUES[classes], atol=1e-10, rtol=0):
            raise ValueError("横线概率融合的训练真值必须为−0.10至−0.02的九个0.01档位。")
        if len(np.unique(classes)) < 2:
            raise ValueError("横线概率融合的训练真值至少需要两个档位。")
        self.n_features_in_ = 25
        self.tabicl_ = make_classifier()
        self.tabicl_.fit(x, classes)
        self.catboost_ = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=.03, l2_leaf_reg=3., random_seed=20261022,
            loss_function='MultiClass', classes_count=9, thread_count=1, task_type='CPU',
            allow_writing_files=False, verbose=False,
        )
        self.catboost_.fit(x, classes)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 25 or np.isinf(x).any():
            raise ValueError("横线概率预测必须提供25列因果特征。")
        if not len(x):
            return np.empty((0, 9))
        parts = []
        for start in range(0, len(x), 256):
            query = x[start:start+256]
            tp = normalize_probabilities(self.tabicl_.predict_proba(query), self.tabicl_.classes_)
            cp = normalize_probabilities(self.catboost_.predict_proba(query), self.catboost_.classes_)
            parts.append(.5*tp+.5*cp)
        return np.concatenate(parts)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return select_level(self.predict_proba(x))

    def __getstate__(self):
        """Serialize portable model state, never a personal checkpoint path."""
        state = self.__dict__.copy()
        classifier = state.pop('tabicl_', None)
        if classifier is not None:
            tab_state = classifier.__getstate__()
            tab_state['model_path'] = None
            tab_state['device'] = 'cpu'
            # Rebuild cached projections lazily; keep frozen training context.
            tab_state.pop('model_kv_cache_', None)
            state['_tabicl_state'] = tab_state
        return state

    def __setstate__(self, state):
        tab_state = state.pop('_tabicl_state', None)
        self.__dict__.update(state)
        if tab_state is not None:
            import torch
            from tabicl import TabICLClassifier

            tab_state['model_path'] = str(checkpoint_path())
            tab_state['allow_auto_download'] = False
            tab_state['device'] = 'mps' if torch.backends.mps.is_available() else 'cpu'
            self.tabicl_ = TabICLClassifier.__new__(TabICLClassifier)
            self.tabicl_.__setstate__(tab_state)


def recipe_metadata() -> dict[str, object]:
    return {'recipe_version': 1, 'checkpoint': CHECKPOINT_NAME, 'checkpoint_sha256': CHECKPOINT_SHA256,
            'values': VALUES.tolist(), 'tabicl_weight': .5, 'catboost_weight': .5,
            'decoder': 'minimum_expected_absolute_relative_error',
            'history_minutes': 180, 'target_observations_at_prediction': False,
            'availability': 'ten_minute_bucket_nominal_plus_five_minutes',
            'input_history': 'only_current_and_past_sensor1_axes_and_temperature'}
