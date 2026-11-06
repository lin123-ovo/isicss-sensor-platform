"""Nine-level regional fusion with validated, persisted fixed weights."""
from __future__ import annotations
from typing import Mapping
import numpy as np
from . import crossline_compact36 as compact
from . import crossline_seven as seven

MODEL_NAME = 'tabicl_tabm_nine_search'
FEATURE_VERSION = 'tilt_temperature_nine_search_v1'
BASE_FEATURE_COLUMNS = compact.BASE_FEATURE_COLUMNS
FEATURE_COLUMNS = compact.FEATURE_COLUMNS
features_for = compact.features_for
history_for = compact.history_for
VALUES = compact.legacy.VALUES
BASE_WEIGHTS = (.525, .825, .9)
RISK_THETA = seven.RISK_THETA
FROZEN_PARAMETERS = {**{k: v for k, v in compact.FROZEN_PARAMETERS.items() if k not in ('tabicl_weight', 'tabm_weight')},
                     'model': MODEL_NAME, 'context_fraction': .75,
                     'negative_weight': BASE_WEIGHTS[0], 'middle_weight': BASE_WEIGHTS[1], 'near_zero_weight': BASE_WEIGHTS[2]}


def normalize_parameters(document: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(document, Mapping) or set(document) != set(FROZEN_PARAMETERS):
        raise ValueError('九档融合配置不完整。')
    result = dict(FROZEN_PARAMETERS)
    for key, expected in FROZEN_PARAMETERS.items():
        value = document[key]
        if key.endswith('_weight'):
            if isinstance(value, bool) or not isinstance(value, (float, int, np.floating, np.integer)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('融合比例必须为0到1的有限数字。')
            result[key] = float(value)
        elif isinstance(value, bool) or value != expected:
            raise ValueError(f'九档融合参数 {key} 不符合配方。')
    return result


def recipe_metadata() -> dict[str, object]:
    return {'model': MODEL_NAME, 'feature_version': FEATURE_VERSION, 'base_recipe': compact.recipe_metadata(),
            'allowed_values': VALUES.tolist(), 'region_rule': 'three_contiguous_groups_of_three',
            'context_fraction': .75, 'decision_theta': list(RISK_THETA), 'decision_strength': .5,
            'probability_policy': 'nine_levels_normalized', 'weights_source': 'saved_parameters',
            'target_observations_at_prediction': False}


def fuse_probabilities(icl: np.ndarray, tabm: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    a, b = np.asarray(icl, float), np.asarray(tabm, float)
    w = np.asarray(weights, float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 9 or w.shape != (3,) or not np.isfinite(w).all() or ((w < 0) | (w > 1)).any():
        raise ValueError('九档融合输入或比例无效。')
    if not len(a): return np.empty((0, 9))
    for p in (a, b):
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
            raise ValueError('分支概率必须非负有限且逐行归一化。')
    q = b + (a-b)*np.repeat(w, 3)
    mass = q.sum(1, keepdims=True)
    if (mass <= 0).any(): raise ValueError('融合概率质量为零。')
    return q/mass


def decode(p: np.ndarray) -> np.ndarray:
    if len(p) == 0: return np.empty(0)
    p = np.maximum(compact.legacy.normalize_probabilities(p, np.arange(9)), 1e-12)
    p /= p.sum(1, keepdims=True)
    cost = np.abs(VALUES[:, None]-VALUES[None, :])/np.abs(VALUES[:, None])
    risk = (1+.5*RISK_THETA[0])*(p@cost) + .5*RISK_THETA[1]*np.linspace(-1., 1., 9)
    return VALUES[np.argmin(risk, axis=1)]


class NineRegionalRegressor(seven.SevenRegionalRegressor):
    """Persist a trained fixed proportion, with the seven model's coverage context."""
    def __init__(self, negative_weight: float, middle_weight: float, near_zero_weight: float):
        self.negative_weight = negative_weight
        self.middle_weight = middle_weight
        self.near_zero_weight = near_zero_weight

    def _recipe_metadata(self) -> dict:
        return recipe_metadata()

    def validate_state(self) -> None:
        normalize_parameters({**FROZEN_PARAMETERS, 'negative_weight': self.negative_weight,
                              'middle_weight': self.middle_weight, 'near_zero_weight': self.near_zero_weight})
        super().validate_state()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return fuse_probabilities(*self._branch_probabilities(x),
                                  (self.negative_weight, self.middle_weight, self.near_zero_weight))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return decode(self.predict_proba(x))
