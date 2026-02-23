from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


@dataclass
class PerModelQualityPredictor:
    max_iter: int = 300
    learning_rate: float = 0.05

    def __post_init__(self):
        self.models: Dict[str, HistGradientBoostingClassifier] = {}

    def fit(self, X: np.ndarray, y_by_model: Dict[str, np.ndarray]):
        for m, y in y_by_model.items():
            clf = HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
            )
            clf.fit(X, y.astype(int))
            self.models[m] = clf
        return self

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for m, clf in self.models.items():
            out[m] = clf.predict_proba(X)[:, 1]
        return out


@dataclass
class LatencyRegressor:
    max_iter: int = 400
    learning_rate: float = 0.05

    def __post_init__(self):
        self.reg = HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=self.learning_rate
        )
        self.is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.reg.fit(X, y.astype(float))
        self.is_fit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            return np.zeros((X.shape[0],), dtype=float)
        return np.maximum(self.reg.predict(X), 0.0)
