"""Fixed seven-output regional fusion; independent portable recipe, no target routing."""
from __future__ import annotations

from typing import Mapping
import numpy as np

from . import crossline_compact36 as compact

MODEL_NAME = 'tabicl_tabm_seven_regional'
FEATURE_VERSION = 'tilt_temperature_seven_regional_v1'
BASE_FEATURE_COLUMNS = compact.BASE_FEATURE_COLUMNS
FEATURE_COLUMNS = compact.FEATURE_COLUMNS
features_for = compact.features_for
history_for = compact.history_for
ALLOWED = np.arange(1, 8)
VALUES = compact.legacy.VALUES
REGION_WEIGHTS = (.525, .825, .9)
# Frozen before the seven-level experiments; keep its original nine-action positions.
RISK_THETA = (-.00033413956530826527, .0011364309788072016)
FROZEN_PARAMETERS = {
    **{key: value for key, value in compact.FROZEN_PARAMETERS.items() if key not in ('tabicl_weight', 'tabm_weight')},
    'model': MODEL_NAME,
    'context_fraction': .75, 'negative_weight': .525, 'middle_weight': .825, 'near_zero_weight': .9,
}


def normalize_parameters(document: Mapping[str, object]) -> dict[str, object]:
    if (not isinstance(document, Mapping) or set(document) != set(FROZEN_PARAMETERS)
            or any(isinstance(document[k], bool) or document[k] != v for k, v in FROZEN_PARAMETERS.items())):
        raise ValueError('七档分段融合必须使用完整固定配置，不能修改三段比例。')
    return dict(FROZEN_PARAMETERS)


def recipe_metadata() -> dict[str, object]:
    return {
        'parameters': dict(FROZEN_PARAMETERS), 'base_recipe': compact.recipe_metadata(),
        'allowed_values': VALUES[ALLOWED].tolist(), 'target_values': VALUES.tolist(),
        'region_tabicl_weights': list(REGION_WEIGHTS),
        'conditioning': 'zero_each_branch_endpoint_mass_then_normalize_before_regional_fusion',
        'context_selector': 'coverage75_input_strata18_equal_capped_chronological_endpoints',
        'decision': {'theta': list(RISK_THETA), 'strength': .5, 'penalty': 10.,
                     'action_positions': np.linspace(-1., 1., 9).tolist(),
                     'source': 'compact36_coverage75_decision_r10_s05_v1_frozen_test_state',
                     'refit': False},
        'selection': 'seven_level_fusion_20261027_conditioned_seven_36690',
        'target_observations_at_prediction': False,
    }


def choose_context(x: np.ndarray) -> np.ndarray:
    """Reproduce the frozen coverage75 rule using current training inputs only."""
    x = compact.matrix(x, 34)
    n = len(x)
    if n < 400:
        raise ValueError('七档分段融合至少需要400条训练记录，确保75%上下文不少于300条。')
    names = list(BASE_FEATURE_COLUMNS)
    descriptors = np.column_stack((x[:, 2], x[:, names.index('temperature_celsius_diff60')],
        abs(x[:, names.index('tilt1_inline_model_departure_mean60')]) +
        abs(x[:, names.index('tilt1_crossline_model_departure_mean60')])))
    for j in range(3):
        good = np.isfinite(descriptors[:, j])
        if not good.any():
            raise ValueError('工况覆盖选样缺少有效训练输入。')
        descriptors[~good, j] = np.median(descriptors[good, j])
    cuts = np.quantile(descriptors[:, 0], [1/3, 2/3])
    groups = (np.searchsorted(cuts, descriptors[:, 0], side='right')*6 +
        np.where(descriptors[:, 1] < -.1, 0, np.where(descriptors[:, 1] > .1, 2, 1))*2 +
        (descriptors[:, 2] > np.median(descriptors[:, 2])).astype(int))
    counts = np.bincount(groups, minlength=18)
    quotas = np.zeros(18, dtype=int)
    mandatory = np.array([0, n-1])
    np.add.at(quotas, groups[mandatory], 1)
    remaining = int(.75*n)-2
    while remaining:
        active = np.flatnonzero(quotas < counts)
        active = active[quotas[active] == quotas[active].min()]
        chosen = active[:remaining]
        quotas[chosen] += 1
        remaining -= len(chosen)
    keep = set(mandatory.tolist())
    for group in range(18):
        rows = np.flatnonzero(groups == group)
        forced = mandatory[groups[mandatory] == group]
        pool = np.setdiff1d(rows, forced, assume_unique=True)
        need = int(quotas[group]-len(forced))
        if need:
            positions = np.array([len(pool)//2]) if need == 1 else np.linspace(0, len(pool)-1, need, dtype=int)
            keep.update(pool[positions].tolist())
    selected = np.array(sorted(keep), dtype=int)
    if len(selected) != int(.75*n):
        raise ValueError('工况覆盖上下文数量不符。')
    return selected


def fuse_probabilities(icl: np.ndarray, tabm: np.ndarray) -> np.ndarray:
    """Condition each branch, fuse each candidate region, then normalize each row."""
    a, b = np.asarray(icl, float), np.asarray(tabm, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 9:
        raise ValueError('七档融合需要两个同形状的九档概率数组。')
    if not len(a):
        return np.empty((0, 9))
    branches = []
    for p in (a, b):
        if (not np.isfinite(p).all() or (p < 0).any()
                or not np.allclose(p.sum(1), 1., atol=1e-6, rtol=0)):
            raise ValueError('七档融合分支概率必须非负、有限且逐行归一化。')
        q = p.copy()
        q[:, [0, 8]] = 0.
        mass = q.sum(1, keepdims=True)
        if (mass <= 0).any():
            raise ValueError('去掉端点后没有剩余概率。')
        branches.append(q/mass)
    w = np.repeat(REGION_WEIGHTS, 3)
    combined = w*branches[0] + (1-w)*branches[1]
    return combined/combined.sum(1, keepdims=True)


def decode(probabilities: np.ndarray) -> np.ndarray:
    """Apply frozen nine-truth relative cost and weak risk bias over seven actions."""
    if len(probabilities) == 0:
        return np.empty(0)
    p = compact.legacy.normalize_probabilities(probabilities, np.arange(9))
    cost = abs(VALUES[:, None]-VALUES[None, :])/abs(VALUES[:, None])
    scale, bias = RISK_THETA
    risk = (1+.5*scale)*(p@cost) + .5*bias*np.linspace(-1., 1., 9)
    return VALUES[ALLOWED[np.argmin(risk[:, ALLOWED], axis=1)]]


class SevenRegionalRegressor(compact.Compact36Regressor):
    """Fixed weights and calibration; rebuild coverage context on each training pool."""
    def _recipe_metadata(self) -> dict:
        return recipe_metadata()

    def _context_indices(self, x: np.ndarray) -> np.ndarray:
        self.context_indices_ = choose_context(x)
        return self.context_indices_

    def _context_count(self) -> int:
        n = self.training_info_['training_n']
        ids = np.asarray(self.context_indices_)
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) != int(.75*n)
                or len(ids) < 300 or ids[0] != 0 or ids[-1] != n-1 or (np.diff(ids) <= 0).any()):
            raise ValueError('七档分段融合归档的75%上下文索引无效。')
        return len(ids)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return fuse_probabilities(*self._branch_probabilities(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return decode(self.predict_proba(x))
