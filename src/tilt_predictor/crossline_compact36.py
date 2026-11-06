"""Fixed, serial TabICLv2/TabM ensemble and causal compact thermal inputs.

The public estimator consumes 34 columns; the frozen training-only thermal
regression adds two residuals. Device models exist only inside fit/predict.
"""
from __future__ import annotations

import copy
import gc
from importlib.metadata import version
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from tabicl import TabICLClassifier
    from torch import Tensor
    from torch.nn import Module

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer

from . import crossline_probability as legacy

MODEL_NAME = 'tabicl_tabm_compact36'
FEATURE_VERSION = 'tilt_temperature_compact36_v1'
STATE_COLUMNS = ('thermal_fast_state', 'thermal_slow_state', 'thermal_depart_fast',
                 'thermal_depart_slow', 'thermal_fast_slow')
BASE_FEATURE_COLUMNS = (*legacy.FEATURE_COLUMNS, *STATE_COLUMNS,
    *(f'{axis}_departure_mean{minutes}' for axis in legacy.INPUT_COLUMNS[:2] for minutes in (60, 180)))
FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, 'tilt1_inline_thermal_residual', 'tilt1_crossline_thermal_residual')
FROZEN_PARAMETERS = {'model': MODEL_NAME, 'recipe_version': 1, 'tabicl_weight': .75,
                     'tabm_weight': .25, 'ridge_alpha': 10., 'n_estimators': 4,
                     'tabm_blocks': 3, 'tabm_width': 256, 'tabm_members': 32}
HISTORY_FLAGS = ('is_inserted_gap', 'temperature_review_required', 'temperature_unresolved',
                 'tilt1_inline_unresolved', 'tilt1_crossline_unresolved', 'history_barrier', 'whole_row_gap_imputed')
THERMAL = [BASE_FEATURE_COLUMNS.index(c) for c in (legacy.TEMP, *STATE_COLUMNS[:2])]


def normalize_parameters(document: Mapping[str, object]) -> dict[str, object]:
    """Validate the complete immutable recipe; reject silent hyperparameter changes."""
    if (not isinstance(document, Mapping) or set(document) != set(FROZEN_PARAMETERS)
            or any(isinstance(document[k], bool) or document[k] != v for k, v in FROZEN_PARAMETERS.items())):
        raise ValueError('36维组合必须使用完整的固定配置，不能修改融合比例或网络参数。')
    return dict(FROZEN_PARAMETERS)


def recipe_metadata() -> dict[str, object]:
    """Describe both feature contracts and every frozen training/inference setting."""
    return {'recipe_version': 1, 'parameters': dict(FROZEN_PARAMETERS),
        'checkpoint': legacy.CHECKPOINT_NAME, 'checkpoint_sha256': legacy.CHECKPOINT_SHA256,
        'base_feature_columns': list(BASE_FEATURE_COLUMNS), 'derived_feature_columns': list(FEATURE_COLUMNS),
        'temperature_half_lives_minutes': [30, 360], 'carry_reset_minutes': 20,
        'motion_windows_minutes': [60, 180], 'ridge_alpha': 10.,
        'tabicl': {'norm_methods': ['power'], 'seed': 20261022, 'n_jobs': 1, 'member_batch': 1, 'query_batch': 128},
        'tabm': {'version': '0.0.3', 'rtdl_num_embeddings_version': '0.0.12', 'seed': 20261023,
            'blocks': 3, 'width': 256, 'members': 32, 'dropout': .1,
            'learning_rate': .002, 'weight_decay': .0003, 'batch': 1024,
            'gradient_clip': 5., 'transform': 'median_active_quantile256_normal_clip10',
            'inner_train_fraction': .8, 'purge_hours': 24, 'minimum_inner_fit': 200,
            'max_epochs': 200, 'patience': 20, 'minimum_nll_improvement': .0001},
        'values': legacy.VALUES.tolist(), 'decoder': 'minimum_expected_absolute_relative_error',
        'input_history': 'full_training_exogenous_history', 'target_observations_at_prediction': False}


def input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain exogenous observations, masking inputs requiring manual review."""
    if not set(['timestamp', *legacy.INPUT_COLUMNS]).issubset(frame.columns):
        raise ValueError('36维组合缺少时间、1号两轴或温度。')
    clean = frame.copy()
    for column in legacy.INPUT_COLUMNS:
        prefix = {'tilt1_inline_model': 'tilt1_inline', 'tilt1_crossline_model': 'tilt1_crossline',
                  'temperature_celsius': 'temperature'}[column]
        flags = [f'{prefix}_unresolved']
        if column == legacy.TEMP:
            flags.append('temperature_review_required')
        for flag in flags:
            if flag in clean:
                clean.loc[clean[flag].fillna(False).astype(bool), column] = np.nan
    # Cleaning inserts synthetic buckets. They must not become fresh observations.
    if 'is_inserted_gap' in clean:
        barrier = clean.get('history_barrier', pd.Series(False, index=clean.index)).fillna(False).astype(bool)
        repaired = clean.get('whole_row_gap_imputed', pd.Series(False, index=clean.index)).fillna(False).astype(bool)
        clean = clean.loc[~clean.is_inserted_gap.fillna(False).astype(bool) | barrier | repaired]
    if clean.empty:
        return clean.loc[:, ['timestamp', *legacy.INPUT_COLUMNS]].copy()
    return legacy.input_frame(clean)


def causal_features(frame: pd.DataFrame, *, end: pd.Timestamp | None = None,
                    include_grid: bool = False) -> pd.DataFrame:
    """Replay full ten-minute history; neither targets nor future inputs are read."""
    raw = legacy.replay_input_frame(input_frame(frame)).set_index('timestamp')
    if raw.empty:
        raise ValueError('36维组合缺少真实输入历史。')
    finish = raw.index[-1] if end is None else max(raw.index[-1], end)
    steps = int((finish - raw.index[0]) / pd.Timedelta(minutes=10)) + 1
    if steps > 600_000:
        raise ValueError('36维组合时间范围超过600000个时间桶。')
    grid = pd.date_range(raw.index[0], finish, freq='10min')
    # Reproduce the frozen25 map on the full grid, including unobserved query buckets.
    observed_source = pd.Series(grid.isin(raw.index), index=grid)
    last_source = pd.Series(grid.where(observed_source), index=grid).ffill()
    age = (pd.Series(grid, index=grid) - last_source).dt.total_seconds() / 60
    filled = raw.reindex(grid).copy()
    barriers = filled.get('history_barrier', pd.Series(False, index=grid)).fillna(False).astype(bool)
    runs = ((age > 20) | age.shift(fill_value=np.inf).gt(20)
            | barriers | barriers.shift(fill_value=False)).cumsum()
    for column, lo, hi in ((legacy.INPUT_COLUMNS[0], -10, 10),
                            (legacy.INPUT_COLUMNS[1], -10, 10), (legacy.TEMP, -40, 50)):
        value = filled[column].where(filled[column].between(lo, hi))
        last_valid = pd.Series(grid.where(value.notna()), index=grid).groupby(runs, sort=False).ffill()
        valid_age = (pd.Series(grid, index=grid) - last_valid).dt.total_seconds() / 60
        filled[column] = value.groupby(runs, sort=False).ffill().where((valid_age <= 20) & ~barriers)
        if column == legacy.TEMP:
            filled[column + '_missing'] = filled[column].isna().astype(float)
            filled[column + '_held'] = (value.isna() & filled[column].notna()).astype(float)
            original = raw.reindex(grid)[column]
            filled[column + '_physical_bad'] = (original.notna() & ~original.between(lo, hi)).astype(float)
    for minutes in (10, 30, 60, 180):
        lag = filled[legacy.TEMP].groupby(runs, sort=False).shift(minutes // 10)
        filled[f'{legacy.TEMP}_lag{minutes}'] = lag
        filled[f'{legacy.TEMP}_diff{minutes}'] = filled[legacy.TEMP] - lag
    past = filled[legacy.TEMP].groupby(runs, sort=False).shift(1)
    for minutes in (30, 60, 180):
        k = minutes // 10
        for name, operation in (('past_mean', 'mean'), ('past_std', 'std'), ('coverage', 'sum')):
            source = past.notna().astype(float) if name == 'coverage' else past
            if operation == 'std':
                value = source.groupby(runs, sort=False).transform(lambda s: s.rolling(k, min_periods=1).std(ddof=0))
            elif operation == 'sum':
                value = source.groupby(runs, sort=False).transform(lambda s: s.rolling(k, min_periods=1).sum()) / k
            else:
                value = source.groupby(runs, sort=False).transform(lambda s: s.rolling(k, min_periods=1).mean())
            filled[f'{legacy.TEMP}_{name}{minutes}'] = value
    for minutes in (60, 180):
        filled[f'temperature_trend_sign{minutes}'] = np.sign(filled[f'{legacy.TEMP}_diff{minutes}'])
    base = filled.loc[:, list(legacy.FEATURE_COLUMNS)]
    values = raw.reindex(grid)[legacy.TEMP]
    fresh = values.between(-40, 50) & values.notna()
    states = np.full((len(grid), 2), np.nan)
    state = np.full(2, np.nan)
    last = None
    for i, t in enumerate(grid):
        if barriers.iloc[i]:
            state[:] = np.nan
            last = None
            continue
        elapsed = np.inf if last is None else (t - last).total_seconds() / 60
        if fresh.iloc[i]:
            value = float(values.iloc[i])
            if elapsed > 20 or not np.isfinite(state).all():
                state[:] = value
            else:
                state += (1 - np.exp(-np.log(2) * elapsed / np.array([30., 360.]))) * (value - state)
            last = t
            states[i] = state
        elif elapsed <= 20:
            states[i] = state
        else:
            state[:] = np.nan
    thermal = pd.DataFrame(np.column_stack([states, filled[legacy.TEMP] - states[:, 0],
        filled[legacy.TEMP] - states[:, 1], states[:, 0] - states[:, 1]]), index=grid, columns=STATE_COLUMNS)
    extra = thermal.copy()
    for axis in legacy.INPUT_COLUMNS[:2]:
        past = filled[axis].groupby(runs, sort=False).shift(1)
        for minutes in (60, 180):
            mean = past.groupby(runs, sort=False).transform(lambda s: s.rolling(minutes // 10, min_periods=1).mean())
            extra[f'{axis}_departure_mean{minutes}'] = filled[axis] - mean
    result = pd.concat([base, extra], axis=1).loc[:, list(BASE_FEATURE_COLUMNS)]
    return result if include_grid else result.reindex(raw.index)


def exogenous_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep input availability flags for replay; exclude every sensor2 target field."""
    return frame.loc[:, ['timestamp', *legacy.INPUT_COLUMNS,
                         *(flag for flag in HISTORY_FLAGS if flag in frame)]].copy()


def history_for(frame: pd.DataFrame, context: pd.DataFrame | None = None,
                seed_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge real observations up to request end, inheriting cleaning flags by timestamp."""
    request = legacy.input_frame(frame)
    annotated = frame.copy()
    if context is not None and not context.empty:
        lookup = exogenous_history(context)
        lookup['timestamp'] = pd.to_datetime(lookup.timestamp, errors='raise')
        if lookup.timestamp.duplicated().any():
            raise ValueError('36维组合历史存在重复时间。')
        lookup = lookup.set_index('timestamp').reindex(pd.to_datetime(frame.timestamp))
        for flag in HISTORY_FLAGS:
            if flag not in annotated and flag in lookup:
                annotated[flag] = lookup[flag].fillna(False).to_numpy(dtype=bool)
    parts = [input_frame(p) for p in (seed_history, context, annotated) if p is not None and not p.empty]
    combined = pd.concat(parts, ignore_index=True).drop_duplicates('timestamp', keep='last')
    combined = combined.loc[combined.timestamp <= request.timestamp.max()]
    return legacy.input_frame(combined)


def features_for(frame: pd.DataFrame, context: pd.DataFrame | None = None,
                 seed_history: pd.DataFrame | None = None) -> np.ndarray:
    """Replay past observations with request precedence and preserve requested row order."""
    combined = history_for(frame, context, seed_history)
    return causal_features(combined, end=pd.to_datetime(frame.timestamp).max(), include_grid=True).reindex(
        pd.to_datetime(frame.timestamp)).to_numpy(float)


def matrix(x: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != width or np.isinf(x).any():
        raise ValueError(f'36维组合需要{width}列有限或缺测输入。')
    return x


class ThermalDecomposition:
    """Training-only standardized Ridge thermal component for the two axes."""
    def fit(self, x: np.ndarray) -> ThermalDecomposition:
        x = matrix(x, 34)
        complete = np.isfinite(x[:, [*THERMAL, 0, 1]]).all(1)
        if complete.sum() < 50:
            raise ValueError('温度分解至少需要50条完整训练记录。')
        t = x[complete][:, THERMAL]
        self.mean_ = t.mean(0)
        self.scale_ = np.where(t.std(0) > 1e-8, t.std(0), 1.)
        self.ridge_ = Ridge(alpha=10.).fit((t - self.mean_) / self.scale_, x[complete, :2])
        self.fit_count_ = int(complete.sum())
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = matrix(x, 34)
        t = (x[:, THERMAL] - self.mean_) / self.scale_
        estimate = t @ self.ridge_.coef_.T + self.ridge_.intercept_
        return np.column_stack([x, x[:, :2] - estimate])


class TabMTransform:
    """Frozen median/active-column/quantile transform, fitted on one training segment."""
    def fit(self, x: np.ndarray) -> TabMTransform:
        x = matrix(x, 36)
        self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
        a = self.imputer_.transform(x)
        self.active_ = a.std(0) > 1e-6 * np.maximum(1., abs(a.mean(0)))
        if not self.active_.any():
            raise ValueError('TabM训练输入全部为常量。')
        noise = np.random.default_rng(20261023).normal(0., 1e-5, a[:, self.active_].shape)
        self.quantile_ = QuantileTransformer(n_quantiles=min(256, len(a)), output_distribution='normal',
            subsample=None, random_state=20261023).fit(a[:, self.active_] + noise)
        self.fit_count_ = len(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        a = self.imputer_.transform(matrix(x, 36))
        return np.clip(self.quantile_.transform(a[:, self.active_]), -10, 10).astype(np.float32)


def release_device() -> None:
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def device() -> str:
    import torch
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def make_classifier() -> TabICLClassifier:
    import torch
    from tabicl import TabICLClassifier
    torch.set_num_threads(2)
    return TabICLClassifier(n_estimators=4, norm_methods=['power'], average_logits=True,
        softmax_temperature=1., kv_cache=True, device=device(), use_amp=False, use_fa3=False,
        offload_mode=False, n_jobs=1, batch_size=1, random_state=20261022,
        model_path=str(legacy.checkpoint_path()), allow_auto_download=False, verbose=False)


def make_tabm(width: int) -> Module:
    from tabm import TabM
    return TabM.make(n_num_features=width, d_out=9, n_blocks=3, d_block=256, k=32, dropout=.1)


def _train_tabm(a: np.ndarray, labels: np.ndarray, epochs: int,
                valid: np.ndarray | None = None, valid_labels: np.ndarray | None = None) -> tuple[dict[str, Tensor], dict[str, int | float | None]]:
    import torch
    torch.manual_seed(20261023)
    model = make_tabm(a.shape[1]).to(device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003)
    rng = np.random.default_rng(20261023)
    best, best_epoch, last = np.inf, 1, 0
    try:
        xx = torch.as_tensor(a, device=device())
        yy = torch.as_tensor(labels, dtype=torch.long, device=device())
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(a))
            for start in range(0, len(a), 1024):
                ids = torch.as_tensor(order[start:start + 1024], device=device())
                optimizer.zero_grad(set_to_none=True)
                logits = model(xx[ids])
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), yy[ids].repeat_interleave(32))
                if not torch.isfinite(loss):
                    raise ValueError('TabM训练损失不是有限值。')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            if valid is not None:
                p = _tabm_probabilities(model, valid)
                score = float(-np.log(np.maximum(p[np.arange(len(p)), valid_labels], 1e-12)).mean())
                if score < best - .0001:
                    best, best_epoch, last = score, epoch, epoch
                if epoch - last >= 20:
                    break
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return state, {'epochs_ran': epoch, 'best_epoch': best_epoch,
                       'inner_best_nll': float(best) if valid is not None else None}
    finally:
        del model, optimizer
        # Tensors on the training device must die before empty_cache is called.
        if 'xx' in locals():
            del xx, yy
        if 'logits' in locals():
            del logits, loss, ids
        release_device()


def _tabm_probabilities(model: Module, a: np.ndarray) -> np.ndarray:
    import torch
    model.eval()
    with torch.no_grad():
        p = [torch.softmax(model(torch.as_tensor(a[s:s + 512], device=device())), -1).mean(1).cpu().numpy()
             for s in range(0, len(a), 512)]
    return legacy.normalize_probabilities(np.concatenate(p), np.arange(9))


class Compact36Regressor(RegressorMixin, BaseEstimator):
    """75/25 probability ensemble; archives CPU state and no live device model."""
    def _recipe_metadata(self) -> dict:
        return recipe_metadata()

    def _context_indices(self, x: np.ndarray) -> np.ndarray:
        return np.arange(len(x))

    def _context_count(self) -> int:
        return self.training_info_['training_n']

    def fit(self, x: np.ndarray, y: np.ndarray, *,
            timestamps: pd.DatetimeIndex | pd.Series | None = None) -> Compact36Regressor:
        """Fit serial components; timestamps are required for purged internal early stopping."""
        x = matrix(x, 34)
        y = np.asarray(y, float)
        if y.ndim != 1 or len(x) != len(y) or len(y) < 300 or not np.isfinite(y).all():
            raise ValueError('36维组合至少需要300条有效训练记录。')
        classes = np.argmin(abs(y[:, None] - legacy.VALUES), axis=1)
        if not np.allclose(y, legacy.VALUES[classes], atol=1e-10, rtol=0) or len(np.unique(classes)) < 2:
            raise ValueError('36维组合真值必须属于九档，且至少包含两个档位。')
        if timestamps is None:
            raise ValueError('36维组合训练必须提供时间。')
        times = pd.DatetimeIndex(timestamps)
        if (len(times) != len(x) or times.hasnans or times.has_duplicates
                or not times.is_monotonic_increasing or times.tz is not None):
            raise ValueError('36维组合训练需要严格递增的本地时间。')
        start = int(.8 * len(times))
        fit = np.flatnonzero(times < times[start] - pd.Timedelta(hours=24))
        valid = np.arange(start, len(times))
        if len(fit) < 200:
            raise ValueError('TabM内部80/20划分净化24小时后不足200条训练记录。')
        self.n_features_in_ = 34
        self.decomposition_ = ThermalDecomposition().fit(x)
        full = self.decomposition_.transform(x)
        context_indices = self._context_indices(x)
        if len(np.unique(classes[context_indices])) < 2:
            raise ValueError('TabICL训练上下文至少需要两个真实档位。')
        tab = make_classifier()
        try:
            tab.fit(full[context_indices], classes[context_indices])
            # Save the portable context only, rebuilding KV caches lazily on predict.
            tab._save_kv_cache = False
            self.tabicl_state_ = tab.__getstate__()
            self.tabicl_state_.update(model_path=None, device='cpu', allow_auto_download=False)
            self.tabicl_state_.pop('model_kv_cache_', None)
        finally:
            del tab
            release_device()
        inner_decomposition = ThermalDecomposition().fit(x[fit])
        inner = inner_decomposition.transform(x)
        inner_transform = TabMTransform().fit(inner[fit])
        _, early = _train_tabm(inner_transform.transform(inner[fit]), classes[fit], 200,
                              inner_transform.transform(inner[valid]), classes[valid])
        self.transform_ = TabMTransform().fit(full)
        self.tabm_state_, refit = _train_tabm(self.transform_.transform(full), classes, early['best_epoch'])
        self.training_info_ = {'inner_fit_n': len(fit), 'inner_valid_n': len(valid), 'training_n': len(x),
            'inner_fit_end': times[fit[-1]].isoformat(), 'inner_validation_start': times[valid[0]].isoformat(),
            'early_stopping': early, 'refit_epochs': refit['epochs_ran'],
            'training_start': times[0].isoformat(), 'training_end': times[-1].isoformat(),
            'inner_ridge_mean': inner_decomposition.mean_.tolist(),
            'inner_ridge_fit_count': inner_decomposition.fit_count_, 'inner_transform_fit_count': inner_transform.fit_count_}
        self.recipe_ = self._recipe_metadata()
        self.versions_ = {p: version(p) for p in ('tabm', 'rtdl_num_embeddings', 'tabicl', 'torch', 'scikit-learn', 'numpy')}
        return self

    def validate_state(self) -> None:
        """Reject changed recipes, dimensions, missing context, and malformed CPU weights."""
        import torch
        if (getattr(self, 'recipe_', None) != self._recipe_metadata() or getattr(self, 'n_features_in_', None) != 34
                or self.versions_.get('tabm') != '0.0.3' or self.versions_.get('rtdl_num_embeddings') != '0.0.12'
                or any(version(p) != self.versions_.get(p) for p in
                    ('tabm', 'rtdl_num_embeddings', 'tabicl', 'torch', 'scikit-learn', 'numpy'))):
            raise ValueError('36维组合归档配置或依赖版本不一致。')
        t = self.tabicl_state_
        expected = {'n_features_in_': 36, 'n_estimators': 4, 'norm_methods': ['power'],
            'n_jobs': 1, 'batch_size': 1, 'random_state': 20261022, 'average_logits': True,
            'softmax_temperature': 1., 'model_path': None, 'device': 'cpu', 'allow_auto_download': False,
            'kv_cache': True, 'cache_mode_': 'kv', 'use_amp': False, 'use_fa3': False, 'offload_mode': False}
        if any(t.get(k) != v for k, v in expected.items()) or t.get('ensemble_generator_') is None:
            raise ValueError('36维组合TabICL状态或训练上下文不完整。')
        eg = t['ensemble_generator_']
        n = self.training_info_['training_n']
        context_n = self._context_count()
        mask = getattr(getattr(eg, 'unique_filter_', None), 'features_to_keep_', np.empty(0))
        if (mask.shape != (36,) or mask.dtype != bool or not mask.any()
                or getattr(eg, 'X_', np.empty((0, 0))).shape != (context_n, int(mask.sum()))
                or getattr(getattr(t.get('X_encoder_'), 'tfm_', None), 'n_features_in_', None) != 36
                or getattr(eg, 'y_', np.empty(0)).shape != (context_n,)
                or not np.isin(t.get('classes_', []), np.arange(9)).all()
                or len(t.get('classes_', [])) < 2 or t.get('n_samples_in_') != context_n
                or t.get('n_classes_') != len(t.get('classes_', []))
                or not np.array_equal(getattr(t.get('y_encoder_'), 'classes_', None), t.get('classes_'))
                or not np.isfinite(eg.X_).all() or not np.isin(eg.y_, np.arange(t['n_classes_'])).all()):
            raise ValueError('36维组合TabICL训练矩阵或类别缺失。')
        d = self.decomposition_
        if (d.ridge_.alpha != 10. or d.ridge_.coef_.shape != (2, 3) or d.ridge_.intercept_.shape != (2,)
                or d.mean_.shape != (3,) or d.scale_.shape != (3,) or (d.scale_ <= 0).any()
                or not all(np.isfinite(v).all() for v in (d.mean_, d.scale_, d.ridge_.coef_, d.ridge_.intercept_))
                or self.transform_.imputer_.n_features_in_ != 36
                or self.transform_.active_.shape != (36,) or not self.transform_.active_.any()
                or self.transform_.quantile_.n_features_in_ != int(self.transform_.active_.sum())
                or self.transform_.fit_count_ != n or d.fit_count_ < 50
                or not np.isfinite(self.transform_.imputer_.statistics_).all()
                or not np.isfinite(self.transform_.quantile_.quantiles_).all()):
            raise ValueError('36维组合温度分解或TabM变换维度无效。')
        template = make_tabm(int(self.transform_.active_.sum())).state_dict()
        if set(template) != set(self.tabm_state_) or any(
                not isinstance(v, torch.Tensor) or v.device.type != 'cpu' or v.shape != template[k].shape
                or not torch.isfinite(v).all() for k, v in self.tabm_state_.items()):
            raise ValueError('36维组合TabM权重缺失、维度错误或不是有效CPU状态。')
        history = legacy.input_frame(self.history_frame_)
        if (list(history.columns) != list(self.history_frame_.columns)
                or history.timestamp.max() > pd.Timestamp(self.training_info_['training_end'])
                or history.timestamp.min() > pd.Timestamp(self.training_info_['training_start'])
                or self.training_info_.get('history_rows') != len(history)
                or self.training_info_['inner_fit_n'] < 200
                or self.training_info_['inner_transform_fit_count'] != self.training_info_['inner_fit_n']
                or pd.Timestamp(self.training_info_['inner_validation_start']) -
                    pd.Timestamp(self.training_info_['inner_fit_end']) <= pd.Timedelta(hours=24)
                or not 1 <= self.training_info_['refit_epochs'] <= 200):
            raise ValueError('36维组合缺少完整训练期输入历史。')
        legacy.checkpoint_path()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        try:
            self.validate_state()
        except (AttributeError, KeyError, TypeError, RuntimeError) as error:
            raise ValueError(f'36维组合归档不完整或权重无效：{error}') from error

    def _branch_probabilities(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Load one device component at a time and return each nine-level distribution."""
        from tabicl import TabICLClassifier
        x = matrix(x, 34)
        if not len(x):
            return np.empty((0, 9)), np.empty((0, 9))
        full = self.decomposition_.transform(x)
        state = copy.deepcopy(self.tabicl_state_)
        state.update(model_path=str(legacy.checkpoint_path()), device=device(), allow_auto_download=False)
        tab = TabICLClassifier.__new__(TabICLClassifier)
        try:
            tab.__setstate__(state)
            tab._build_kv_cache()
            p = np.concatenate([legacy.normalize_probabilities(tab.predict_proba(full[s:s + 128]), tab.classes_)
                                for s in range(0, len(full), 128)])
        finally:
            del tab, state
            release_device()
        model = make_tabm(int(self.transform_.active_.sum()))
        try:
            model.load_state_dict(self.tabm_state_, strict=True)
            model.to(device())
            q = _tabm_probabilities(model, self.transform_.transform(full))
        finally:
            del model
            release_device()
        return p, q

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return the unchanged ordinary 75/25 probability fusion."""
        p, q = self._branch_probabilities(x)
        return .75 * p + .25 * q

    def predict(self, x: np.ndarray) -> np.ndarray:
        return legacy.select_level(self.predict_proba(x))
