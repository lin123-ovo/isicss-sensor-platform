"""CPU-friendly TabPFN v2 regression for the existing three-feature contract."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

MODEL_NAME = "tabpfn_v2"
MAX_CONTEXT_ROWS = 1000
FROZEN_PARAMETERS = {"model": MODEL_NAME, "max_context_rows": MAX_CONTEXT_ROWS}


class TabPFNV2Regressor(RegressorMixin, BaseEstimator):
    """Fit TabPFN v2 on the most recent training context and predict locally."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNV2Regressor":
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        if (features.ndim != 2 or features.shape[1] != 3 or target.shape != (len(features),)
                or len(features) == 0 or not np.isfinite(features).all()
                or not np.isfinite(target).all()):
            raise ValueError("TabPFN v2 需要非空、有限的三特征训练数据。")
        self.context_rows_ = min(len(features), MAX_CONTEXT_ROWS)
        self.estimator_ = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device="cpu", n_estimators=4, random_state=42,
        )
        self.estimator_.fit(features[-MAX_CONTEXT_ROWS:], target[-MAX_CONTEXT_ROWS:])
        self.n_features_in_ = 3
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != 3 or not np.isfinite(features).all():
            raise ValueError("TabPFN v2 预测需要有限的三特征输入。")
        return np.asarray(self.estimator_.predict(features), dtype=float)
