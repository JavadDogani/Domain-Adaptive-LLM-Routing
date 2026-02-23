# scripts/dacr/calibration.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    """Platt scaling on logits with safe fallback for single-class data."""
    lr: Optional[LogisticRegression] = None
    const_: Optional[float] = None  # if only one class seen

    def fit(self, probs: np.ndarray, y: np.ndarray):
        probs = np.asarray(probs).reshape(-1)
        y = np.asarray(y).reshape(-1).astype(int)

        uniq = np.unique(y)
        if uniq.size < 2:
            # Degenerate slice: always predict the observed class
            self.const_ = float(uniq[0])
            self.lr = None
            return self

        self.const_ = None
        self.lr = LogisticRegression(solver="lbfgs", max_iter=200)

        eps = 1e-6
        logit = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps)).reshape(-1, 1)
        self.lr.fit(logit, y)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs).reshape(-1)
        if self.lr is None and self.const_ is not None:
            return np.full_like(probs, self.const_, dtype=float)

        eps = 1e-6
        logit = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps)).reshape(-1, 1)
        return self.lr.predict_proba(logit)[:, 1]


class DomainAdaptiveCalibrator:
    """
    One-model domain-adaptive calibration:
      - global Platt calibrator
      - optional per-domain calibrators when enough data
      - fallback to global if domain slice is too small
    """

    def __init__(self, min_domain_samples: int = 20):
        self.min_domain_samples = int(min_domain_samples)
        self.global_cal: Optional[PlattCalibrator] = None
        self.domain_cal: Dict[str, Optional[PlattCalibrator]] = {}

    def fit(self, probs: np.ndarray, y: np.ndarray, domains):
        probs = np.asarray(probs).reshape(-1)
        y = np.asarray(y).reshape(-1).astype(int)
        domains = np.asarray(domains).astype(str)

        # global calibrator
        self.global_cal = PlattCalibrator().fit(probs, y)

        # per-domain calibrators
        self.domain_cal = {}
        for d in np.unique(domains):
            idx = (domains == d)
            if idx.sum() < self.min_domain_samples:
                self.domain_cal[d] = None  # fallback to global
                continue
            # PlattCalibrator handles single-class safely
            self.domain_cal[d] = PlattCalibrator().fit(probs[idx], y[idx])

        return self

    def predict(self, probs: np.ndarray, domains):
        assert self.global_cal is not None, "Call fit() before predict()."
        probs = np.asarray(probs).reshape(-1)
        domains = np.asarray(domains).astype(str)

        out = np.empty_like(probs, dtype=float)
        for d in np.unique(domains):
            idx = (domains == d)
            cal_d = self.domain_cal.get(d, None)
            if cal_d is None:
                out[idx] = self.global_cal.predict(probs[idx])
            else:
                out[idx] = cal_d.predict(probs[idx])
        return out
    
def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs).reshape(-1)
    y = np.asarray(y).reshape(-1).astype(int)

    # guard: if empty
    if probs.size == 0:
        return float("nan")

    # clip for safety
    probs = np.clip(probs, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.size

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        if not np.any(mask):
            continue

        conf = probs[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)