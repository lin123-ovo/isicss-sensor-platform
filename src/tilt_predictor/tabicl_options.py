"""Versioned single-TabICL options and chronological probability calibration."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

OPTION_KEYS = frozenset({'context_fraction', 'temperature_calibration'})
DEFAULT_OPTIONS = {'context_fraction': .5, 'temperature_calibration': False}


def normalize_options(document: Mapping[str, object]) -> dict[str, float | bool]:
    """Require an integer percentage and an explicit calibration flag."""
    if not isinstance(document, Mapping) or set(document) != OPTION_KEYS:
        raise ValueError('TabICLv2设置必须包含上下文保留比例和NLL校准开关。')
    fraction, enabled = document['context_fraction'], document['temperature_calibration']
    if (isinstance(fraction, bool) or not isinstance(fraction, Real)
            or not np.isfinite(fraction) or not .01 <= fraction <= 1.
            or not np.isclose(float(fraction)*100, round(float(fraction)*100), atol=1e-8, rtol=0)):
        raise ValueError('上下文保留比例须为1%～100%的整数百分比。')
    if type(enabled) is not bool:
        raise ValueError('NLL温度校准开关须为布尔值。')
    return {'context_fraction': round(float(fraction)*100)/100, 'temperature_calibration': enabled}


def normalize_recipe(document: Mapping[str, object], frozen: dict) -> dict:
    """Accept the exact legacy recipe or a complete v2 option-bearing recipe."""
    if not isinstance(document, Mapping):
        raise ValueError('TabICLv2参数须为对象。')
    if type(document.get('recipe_version')) is not int:
        raise ValueError('TabICLv2方案版本须为整数。')
    modern = document.get('recipe_version') == 2
    expected = {**frozen, 'recipe_version': 2} if modern else frozen
    if set(document) != set(expected) | (OPTION_KEYS if modern else set()):
        raise ValueError('TabICLv2参数配置不完整。')
    if any(isinstance(document[k], bool) or document[k] != v for k, v in expected.items()):
        raise ValueError('TabICLv2固定参数或方案版本无效。')
    return {**expected, **(normalize_options({k: document[k] for k in OPTION_KEYS}) if modern else {})}


def context_indices(x: np.ndarray, fraction: float) -> np.ndarray:
    """Select deterministic temperature-spread rows, without accepting labels."""
    normalize_options({'context_fraction': fraction, 'temperature_calibration': False})
    if x.ndim != 2 or x.shape[1] < 3 or not np.isfinite(x[:, 2]).all():
        raise ValueError('上下文选样需要有效的训练温度。')
    count = len(x) * round(fraction*100) // 100
    if count < 300:
        raise ValueError(f'训练池{len(x)}条，保留{fraction:.0%}后只有{count}条上下文，至少需要300条；请增加数据或提高比例。')
    if fraction == 1.:
        return np.arange(len(x), dtype=np.int64)
    order = np.argsort(x[:, 2], kind='stable')
    return np.sort(order[np.linspace(0, len(x)-1, count, dtype=int)]).astype(np.int64)


def index_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def calibration_split(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Keep the last 20% for calibration and purge more than 24 hours."""
    if len(times) < 300 or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing or times.tz is not None:
        raise ValueError('NLL校准需要严格递增的本地训练时间。')
    start = int(.8*len(times))
    fit = np.flatnonzero(times < times[start]-pd.Timedelta(hours=24))
    valid = np.arange(start, len(times))
    if len(fit) < 300 or len(valid) < 50:
        raise ValueError(f'NLL校准净化24小时后拟合段{len(fit)}条、校准段{len(valid)}条；至少需要300/50条，请增加训练数据。')
    return fit, valid


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict:
    """Calibrate seen classes; audit unsupported targets instead of fabricating probabilities.

    NLL describes supported records only. Require at least 50 such records and
    two supported target classes. Invalid probabilities are rejected even on
    unsupported records; they are not a reason to silently discard input.
    """
    p = np.asarray(probabilities, dtype=float)
    labels, classes = np.asarray(labels), np.asarray(classes)
    if labels.ndim != 1 or classes.ndim != 1 or p.ndim != 2 or p.shape != (len(labels), len(classes)):
        raise ValueError('NLL校准概率矩阵与目标档位的维度不一致。')
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1., atol=1e-5)):
        raise ValueError('NLL校准概率必须有限、在[0,1]内且每行和为1。')
    if (labels.dtype.kind not in 'iu' or classes.dtype.kind not in 'iu'
            or not np.isin(labels, np.arange(9)).all() or not np.isin(classes, np.arange(9)).all()
            or len(np.unique(classes)) != len(classes) or len(classes) < 2):
        raise ValueError('NLL校准档位必须属于九档，模型须至少覆盖两个档位。')
    supported = np.isin(labels, classes)
    total, used = len(labels), int(supported.sum())
    unsupported_classes, unsupported_counts = np.unique(labels[~supported], return_counts=True)
    if used < 50 or len(np.unique(labels[supported])) < 2:
        raise ValueError(f'NLL有效校准仅{used}/{total}条、{len(np.unique(labels[supported]))}个档位；'
                         f'至少需要50条且两个已覆盖档位（未覆盖{total-used}条），请增加训练数据或调整训练划分。')
    audit = {'calibration_n': total, 'calibration_used_n': used,
             'calibration_uncovered_n': total-used, 'calibration_coverage': used/total,
             'calibration_used_classes': np.unique(labels[supported]).tolist(),
             'uncovered_class_counts': [{'class': int(c), 'count': int(n)}
                                        for c, n in zip(unsupported_classes, unsupported_counts, strict=True)]}
    p, labels = p[supported], labels[supported]
    locations = np.array([np.flatnonzero(classes == label)[0] for label in labels])
    logp = np.log(np.clip(p, 1e-15, 1.))

    def loss(log_temperature: float) -> float:
        z = logp/np.exp(log_temperature)
        return float(np.mean(logsumexp(z, axis=1)-z[np.arange(len(z)), locations]))

    result = minimize_scalar(loss, bounds=(np.log(.25), np.log(4.)), method='bounded', options={'xatol': 1e-7})
    if not result.success or not np.isfinite([result.fun, result.x]).all():
        raise ValueError('NLL温度校准优化失败，请检查训练数据。')
    candidates = [(loss(np.log(t)), t) for t in (.25, 1., 4., float(np.exp(result.x)))]
    best_nll = min(score for score, _ in candidates)
    score, temperature = min((item for item in candidates if item[0] <= best_nll+1e-12), key=lambda item: abs(np.log(item[1])))
    return {'temperature': temperature, 'nll_before': loss(0.), 'nll_after': score,
            **audit, 'classes': classes.tolist(), 'bounds': [.25, 4.], 'method': 'heldout_nll_seen_classes_v2'}


def validate_calibration_coverage(record: Mapping[str, object]) -> None:
    """Reject incomplete or inconsistent v2 audit records when loading an archive."""
    try:
        total, used, uncovered = (record[k] for k in ('calibration_n', 'calibration_used_n', 'calibration_uncovered_n'))
        coverage = record['calibration_coverage']
        counts, used_classes, classes = (record[k] for k in ('uncovered_class_counts', 'calibration_used_classes', 'classes'))
        if (any(type(n) is not int for n in (total, used, uncovered)) or used < 50
                or uncovered < 0 or total != used+uncovered
                or isinstance(coverage, bool) or not isinstance(coverage, Real)
                or not np.isfinite(coverage) or not np.isclose(coverage, used/total, atol=1e-12, rtol=0)
                or not isinstance(counts, list) or not isinstance(used_classes, list)
                or len(set(used_classes)) < 2 or len(set(used_classes)) != len(used_classes)
                or not set(used_classes) <= set(classes)):
            raise ValueError('NLL校准覆盖记录不一致。')
        seen, summed = set(), 0
        for entry in counts:
            if (not isinstance(entry, dict) or set(entry) != {'class', 'count'}
                    or type(entry['class']) is not int or not 0 <= entry['class'] <= 8
                    or entry['class'] in classes or entry['class'] in seen
                    or type(entry['count']) is not int or entry['count'] <= 0):
                raise ValueError('NLL未覆盖档位记录无效。')
            seen.add(entry['class'])
            summed += entry['count']
        if summed != uncovered:
            raise ValueError('NLL未覆盖记录数量不一致。')
    except (KeyError, TypeError) as error:
        raise ValueError('NLL校准覆盖记录不完整。') from error
