"""Independent 36-feature tabular classifiers for crossline ablation."""
from __future__ import annotations

import copy
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from . import crossline_compact36 as compact
from . import tabicl_options as options


class SingleBranchRegressor(RegressorMixin, BaseEstimator):
    """Fit one branch using the same causal features and nine-level decoder."""

    def __init__(self, branch: str, n_estimators: int = 4, *, context_fraction: float = 1.,
                 temperature_calibration: bool = False, parameter_version: int = 1):
        self.branch = branch
        self.n_estimators = n_estimators
        self.context_fraction = context_fraction
        self.temperature_calibration = temperature_calibration
        self.parameter_version = parameter_version

    def fit(self, x, y, *, timestamps=None):
        x = compact.matrix(x, 34)
        if type(self.n_estimators) is not int or self.n_estimators not in (4, 16):
            raise ValueError('TabICL集成成员数量无效。')
        y = np.asarray(y, dtype=float)
        if self.branch not in ('tabicl', 'tabm') or y.ndim != 1 or len(x) != len(y) or len(y) < 300 or not np.isfinite(y).all():
            raise ValueError('单模型至少需要300条有效训练记录。')
        labels = np.argmin(abs(y[:, None] - compact.legacy.VALUES), axis=1)
        if not np.allclose(y, compact.legacy.VALUES[labels], atol=1e-10, rtol=0) or len(np.unique(labels)) < 2:
            raise ValueError('训练目标须属于九档且至少有两个不同档位。')
        times = pd.DatetimeIndex(timestamps) if timestamps is not None else None
        if times is None or len(times) != len(x) or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing or times.tz is not None:
            raise ValueError('训练需要严格递增的本地时间。')
        settings = options.normalize_options({'context_fraction': self.context_fraction,
                                              'temperature_calibration': self.temperature_calibration})
        if type(self.parameter_version) is not int or self.parameter_version not in (1, 2):
            raise ValueError('TabICLv2参数版本无效。')
        if (self.parameter_version == 1 or self.branch != 'tabicl') and settings != {'context_fraction': 1., 'temperature_calibration': False}:
            raise ValueError('可配置上下文和校准仅用于新版TabICLv2独立模型。')
        self.calibrated_temperature_ = 1.
        calibration = {'enabled': False, 'temperature': 1.}
        if self.branch == 'tabicl' and self.temperature_calibration:
            fit, valid = options.calibration_split(times)
            inner_decomp = compact.ThermalDecomposition().fit(x[fit])
            ids = options.context_indices(x[fit], self.context_fraction)
            if len(np.unique(labels[fit][ids])) < 2:
                raise ValueError('NLL校准拟合上下文至少需要两个档位。')
            classifier = compact.make_classifier()
            classifier.n_estimators = self.n_estimators
            try:
                classifier.fit(inner_decomp.transform(x[fit])[ids], labels[fit][ids])
                inner_query = inner_decomp.transform(x[valid])
                p = np.concatenate([classifier.predict_proba(inner_query[s:s+128]) for s in range(0, len(valid), 128)])
                calibration = {'enabled': True, **options.fit_temperature(p, labels[valid], classifier.classes_),
                               'fit_n': len(fit), 'context_n': len(ids), 'purge_hours': 24,
                               'fit_start': times[fit[0]].isoformat(), 'fit_end': times[fit[-1]].isoformat(),
                               'calibration_start': times[valid[0]].isoformat(), 'calibration_end': times[valid[-1]].isoformat(),
                               'context_index_sha256': options.index_digest(fit[ids])}
                self.calibrated_temperature_ = calibration['temperature']
            finally:
                del classifier
                compact.release_device()
        self.n_features_in_ = 34
        self.decomposition_ = compact.ThermalDecomposition().fit(x)
        full = self.decomposition_.transform(x)
        self.training_info_ = {'training_n': len(x), 'training_start': times[0].isoformat(), 'training_end': times[-1].isoformat()}
        if self.branch == 'tabicl':
            ids = options.context_indices(x, self.context_fraction)
            if len(np.unique(labels[ids])) < 2:
                raise ValueError('所选TabICLv2上下文至少需要两个真实档位，请提高保留比例。')
            self.context_indices_ = ids
            if self.parameter_version == 2:
                self.training_info_.update(context_fraction=self.context_fraction, context_n=len(ids),
                    context_index_sha256=options.index_digest(ids), calibration=calibration, parameter_version=2)
            classifier = compact.make_classifier()
            classifier.n_estimators = self.n_estimators
            classifier.softmax_temperature = self.calibrated_temperature_
            try:
                classifier.fit(full[ids], labels[ids])
                classifier._save_kv_cache = False
                self.tabicl_state_ = classifier.__getstate__()
                self.tabicl_state_.update(model_path=None, device='cpu', allow_auto_download=False)
                self.tabicl_state_.pop('model_kv_cache_', None)
            finally:
                del classifier
                compact.release_device()
        else:
            start = int(.8 * len(times))
            fit = np.flatnonzero(times < times[start] - pd.Timedelta(hours=24))
            valid = np.arange(start, len(times))
            if len(fit) < 200:
                raise ValueError('TabM内部时间划分净化24小时后不足200条训练记录。')
            inner_decomp = compact.ThermalDecomposition().fit(x[fit])
            inner = inner_decomp.transform(x)
            inner_transform = compact.TabMTransform().fit(inner[fit])
            _, early = compact._train_tabm(inner_transform.transform(inner[fit]), labels[fit], 200,
                                            inner_transform.transform(inner[valid]), labels[valid])
            self.transform_ = compact.TabMTransform().fit(full)
            self.tabm_state_, refit = compact._train_tabm(self.transform_.transform(full), labels, early['best_epoch'])
            self.training_info_.update(inner_fit_n=len(fit), inner_valid_n=len(valid),
                                       inner_fit_end=times[fit[-1]].isoformat(),
                                       inner_validation_start=times[valid[0]].isoformat(),
                                       early_stopping=early, refit_epochs=refit['epochs_ran'])
        packages = ('tabicl', 'torch', 'scikit-learn', 'numpy') if self.branch == 'tabicl' else ('tabm', 'rtdl_num_embeddings', 'torch', 'scikit-learn', 'numpy')
        self.versions_ = {p: version(p) for p in packages}
        return self

    def validate_state(self):
        if self.branch not in ('tabicl', 'tabm') or self.n_features_in_ != 34 or self.training_info_['training_n'] < 300:
            raise ValueError('单模型归档配置不完整。')
        packages = (('tabicl', 'torch', 'scikit-learn', 'numpy') if self.branch == 'tabicl'
                    else ('tabm', 'rtdl_num_embeddings', 'torch', 'scikit-learn', 'numpy'))
        if set(self.versions_) != set(packages) or any(version(p) != self.versions_[p] for p in packages):
            raise ValueError('单模型依赖版本与归档不一致。')
        history = compact.legacy.input_frame(self.history_frame_)
        if len(history) != self.training_info_['history_rows'] or history.timestamp.max() != pd.Timestamp(self.training_info_['training_end']):
            raise ValueError('单模型训练期输入历史无效。')
        if self.branch == 'tabicl':
            state = self.tabicl_state_
            context_n = self.training_info_['training_n']
            if getattr(self, 'parameter_version', 1) == 2:
                settings = options.normalize_options({'context_fraction': self.context_fraction,
                                                      'temperature_calibration': self.temperature_calibration})
                context_n = self.training_info_['training_n'] * round(settings['context_fraction']*100) // 100
                ids = np.asarray(self.context_indices_)
                info = self.training_info_
                calibration = info['calibration']
                temperature = self.calibrated_temperature_
                if (ids.dtype.kind not in 'iu' or ids.shape != (context_n,) or context_n < 300
                        or (ids < 0).any() or (ids >= info['training_n']).any() or (np.diff(ids) <= 0).any()
                        or info['parameter_version'] != 2 or info['context_n'] != context_n
                        or info['context_fraction'] != self.context_fraction
                        or info['context_index_sha256'] != options.index_digest(ids)
                        or calibration['enabled'] is not self.temperature_calibration
                        or isinstance(temperature, bool) or not np.isfinite(temperature) or not .25 <= temperature <= 4.
                        or calibration['temperature'] != temperature or state.get('softmax_temperature') != temperature
                        or state.get('average_logits') is not True):
                    raise ValueError('TabICLv2上下文或校准状态与归档参数不一致。')
                if self.temperature_calibration:
                    if (calibration['calibration_n'] < 50 or calibration['context_n'] < 300
                            or calibration['fit_n'] < calibration['context_n']
                            or calibration['calibration_n']+calibration['fit_n'] > info['training_n']
                            or len(set(calibration['classes'])) < 2
                            or not np.isin(calibration['classes'], np.arange(9)).all()
                            or not isinstance(calibration['context_index_sha256'], str)
                            or len(calibration['context_index_sha256']) != 64
                            or not np.isfinite([calibration['nll_before'], calibration['nll_after']]).all()
                            or calibration['nll_after'] < 0 or calibration['nll_after'] > calibration['nll_before']+1e-10
                            or calibration['method'] not in ('heldout_nll_v1', 'heldout_nll_seen_classes_v2')
                            or calibration['bounds'] != [.25, 4.]
                            or calibration['purge_hours'] != 24
                            or not pd.Timestamp(info['training_start']) <= pd.Timestamp(calibration['fit_start']) <= pd.Timestamp(calibration['fit_end'])
                            or not pd.Timestamp(calibration['fit_end']) < pd.Timestamp(calibration['calibration_start'])-pd.Timedelta(hours=24)
                            or not pd.Timestamp(calibration['calibration_start']) <= pd.Timestamp(calibration['calibration_end']) <= pd.Timestamp(info['training_end'])):
                        raise ValueError('TabICLv2校准记录无效或存在时间交叠。')
                    if calibration['method'] == 'heldout_nll_seen_classes_v2':
                        options.validate_calibration_coverage(calibration)
                elif temperature != 1.:
                    raise ValueError('未启用校准的TabICLv2温度必须为1。')
            elif (getattr(self, 'parameter_version', 1) != 1
                    or getattr(self, 'context_fraction', 1.) != 1.
                    or getattr(self, 'temperature_calibration', False) is not False
                    or self.training_info_.get('parameter_version', 1) != 1
                    or state.get('softmax_temperature') != 1.):
                raise ValueError('历史TabICLv2归档配置无效。')
            members = getattr(self, 'n_estimators', 4)
            if type(members) is not int or members not in (4, 16) or state.get('n_estimators') != members:
                raise ValueError('TabICLv2集成成员数量与归档不一致。')
            if state.get('n_features_in_') != 36 or state.get('model_path') is not None or state.get('device') != 'cpu' or state.get('allow_auto_download') is not False:
                raise ValueError('TabICLv2归档状态无效。')
            generator = state.get('ensemble_generator_')
            mask = getattr(getattr(generator, 'unique_filter_', None), 'features_to_keep_', np.empty(0))
            if (mask.shape != (36,) or not mask.any()
                    or getattr(generator, 'X_', np.empty((0, 0))).shape != (context_n, int(mask.sum()))
                    or getattr(generator, 'y_', np.empty(0)).shape != (context_n,)):
                raise ValueError('TabICLv2训练上下文维度无效。')
            compact.legacy.checkpoint_path()
        else:
            import torch
            if self.versions_.get('tabm') != '0.0.3' or self.versions_.get('rtdl_num_embeddings') != '0.0.12':
                raise ValueError('TabM依赖版本不匹配。')
            model = compact.make_tabm(int(self.transform_.active_.sum()))
            model.load_state_dict(self.tabm_state_, strict=True)
            if any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or not torch.isfinite(t).all() for t in self.tabm_state_.values()):
                raise ValueError('TabM权重无效。')

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            self.validate_state()
        except (AttributeError, KeyError, TypeError, RuntimeError) as error:
            raise ValueError(f'单模型归档不完整：{error}') from error

    def predict(self, x):
        x = compact.matrix(x, 34)
        if not len(x):
            return np.empty(0)
        full = self.decomposition_.transform(x)
        if self.branch == 'tabicl':
            from tabicl import TabICLClassifier
            state = copy.deepcopy(self.tabicl_state_)
            state.update(model_path=str(compact.legacy.checkpoint_path()), device=compact.device(), allow_auto_download=False)
            model = TabICLClassifier.__new__(TabICLClassifier)
            try:
                model.__setstate__(state)
                model._build_kv_cache()
                p = np.concatenate([compact.legacy.normalize_probabilities(model.predict_proba(full[s:s + 128]), model.classes_) for s in range(0, len(full), 128)])
            finally:
                del model, state
                compact.release_device()
        else:
            model = compact.make_tabm(int(self.transform_.active_.sum()))
            try:
                model.load_state_dict(self.tabm_state_, strict=True)
                model.to(compact.device())
                p = compact._tabm_probabilities(model, self.transform_.transform(full))
            finally:
                del model
                compact.release_device()
        return compact.legacy.select_level(p)
