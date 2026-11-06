"""Chronological, cancellable search of three regional fusion weights."""
from __future__ import annotations
from collections.abc import Callable
import hashlib
import json
import numpy as np
import pandas as pd
from . import crossline_nine_regional as nine


class SearchCancelled(Exception):
    """The caller cancelled a weight search."""


def fingerprint(frame: pd.DataFrame, context: pd.DataFrame, source_sha: str, cleaning_sha: str,
                strategy: str, split_label: str) -> str:
    """Bind a search to actual data, exogenous context, split and recipe."""
    digest = hashlib.sha256()
    for data in (frame, context):
        digest.update(json.dumps([(str(c), str(data[c].dtype)) for c in data.columns]).encode())
        digest.update(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
    digest.update(json.dumps({'source': source_sha, 'cleaning': cleaning_sha, 'strategy': strategy,
                              'split': split_label, 'recipe': nine.recipe_metadata()}, sort_keys=True).encode())
    return digest.hexdigest()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    valid = np.isfinite(y) & (np.abs(y) > 1e-12)
    pred = nine.decode(p)
    region = lambda v: np.where(v <= -.08+1e-10, 0, np.where(v >= -.04-1e-10, 2, 1))
    return {'rows': len(y), 'relative_valid': int(valid.sum()), 'zero_excluded': int((~valid).sum()),
            'mare_percent': (float(np.mean(np.abs(pred[valid]-y[valid])/np.abs(y[valid]))*100) if valid.any() else None),
            'cross_region_errors': int((region(y) != region(pred)).sum())}


def search_probabilities(folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                         progress: Callable[[int, int], None] | None = None,
                         cancelled: Callable[[], bool] | None = None,
                         grid_size: int = 101) -> dict:
    """Find pooled MARE minimum on grid; all target values are evaluation labels only."""
    if not folds or grid_size < 2 or grid_size > 101:
        raise ValueError('搜索数据或网格无效。')
    aa, bb, yy = [], [], []
    for a, b, y in folds:
        y = np.asarray(y, float)
        nine.fuse_probabilities(a, b, nine.BASE_WEIGHTS)
        if y.ndim != 1 or len(y) != len(a) or not np.isfinite(y).all():
            raise ValueError('验证标签无效。')
        aa.append(a); bb.append(b); yy.append(y)
    a, b, y = np.concatenate(aa), np.concatenate(bb), np.concatenate(yy)
    valid = np.abs(y) > 1e-12
    if not valid.any(): raise ValueError('验证数据没有可计算相对误差的真值。')
    values = nine.VALUES
    cost = np.abs(values[:, None]-values[None, :])/np.abs(values[:, None])
    position = np.linspace(-1, 1, 9)
    scale, bias = 1+.5*nine.RISK_THETA[0], .5*nine.RISK_THETA[1]
    offset = scale*(b@cost) + bias*position
    slopes = []
    masses = []
    for k in range(3):
        delta = np.zeros_like(a);delta[:, 3*k:3*k+3] = (a-b)[:, 3*k:3*k+3]
        mass = delta.sum(1)
        slopes.append(scale*(delta@cost)+bias*mass[:, None]*position)
        masses.append(mass)
    slopes = np.stack(slopes);masses = np.stack(masses)
    grid = np.linspace(0,1,grid_size); total = grid_size**3
    best_score = float('inf');best_weights = None; tolerance = 1e-12
    for start in range(0,total,64):
        if cancelled and cancelled(): raise SearchCancelled('已取消比例搜索。')
        ids = np.arange(start,min(start+64,total))
        w = np.column_stack((grid[ids//(grid_size*grid_size)], grid[(ids//grid_size)%grid_size], grid[ids%grid_size]))
        mass = 1+w@masses
        scores = offset[None, :, :] + np.einsum('cr,rna->cna', w, slopes, optimize=True)
        choices = scores.argmin(axis=2); smallest = np.take_along_axis(scores, choices[..., None], axis=2)
        close = (np.abs(scores-smallest) <= 1e-8).sum(axis=2)>1
        for candidate in np.flatnonzero(close.any(1)):
            idx = np.flatnonzero(close[candidate] & (mass[candidate] > 0))
            if len(idx):
                choices[candidate, idx] = np.argmin(np.abs(nine.decode(nine.fuse_probabilities(a[idx],b[idx],tuple(w[candidate])))[:,None]-values),axis=1)
        errors = np.abs(values[choices[:,valid]]-y[None,valid])/np.abs(y[None,valid])
        objectives = errors.mean(1)*100
        objectives[(mass<=0).any(1)] = np.inf
        for i in np.flatnonzero(objectives <= min(best_score, float(objectives.min()))+tolerance):
            score = float(objectives[i]); candidate = tuple(float(v) for v in w[i]); distance = sum(abs(v-b) for v,b in zip(candidate,nine.BASE_WEIGHTS))
            prior_distance = float('inf') if best_weights is None else sum(abs(v-b) for v,b in zip(best_weights,nine.BASE_WEIGHTS))
            if score < best_score-tolerance or (abs(score-best_score)<=tolerance and (round(distance,12),candidate)<(round(prior_distance,12),best_weights)):
                best_score,best_weights=score,candidate
        if progress: progress(min(start+len(ids),total),total)
    base = tuple(nine.BASE_WEIGHTS)
    base_p = nine.fuse_probabilities(a,b,base); base_score = _metrics(y,base_p)['mare_percent']
    if base_score < best_score-tolerance or (abs(base_score-best_score)<=tolerance and best_weights != base):
        best_score,best_weights = base_score,base
    if best_weights is None: raise ValueError("所有融合候选均无有效概率质量。")
    results=[]
    for i,(fa,fb,fy) in enumerate(folds):
        results.append({'fold':i+1,'selected':_metrics(fy,nine.fuse_probabilities(fa,fb,best_weights)),
                        'baseline':_metrics(fy,nine.fuse_probabilities(fa,fb,base))})
    return {'weights':list(best_weights), 'grid_size':grid_size, 'grid_candidates':total,
            'objective':'pooled_mean_absolute_relative_error_percent', 'folds':results,
            'selected':_metrics(y,nine.fuse_probabilities(a,b,best_weights)), 'baseline':_metrics(y,base_p)}
