from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class LinUCBBandit:
    """Linear UCB contextual bandit with per-arm parameters."""
    n_features: int
    alpha: float = 1.0
    ridge: float = 1.0

    def __post_init__(self):
        self.A: Dict[str, np.ndarray] = {}
        self.b: Dict[str, np.ndarray] = {}

    def ensure_arm(self, arm: str):
        if arm not in self.A:
            self.A[arm] = self.ridge * np.eye(self.n_features, dtype=np.float64)
            self.b[arm] = np.zeros((self.n_features,), dtype=np.float64)

    def select(self, x: np.ndarray, arms: List[str]) -> Tuple[str, Dict[str, float]]:
        scores: Dict[str, float] = {}
        best_arm, best_score = None, -1e18
        for a in arms:
            self.ensure_arm(a)
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            mu = float(theta @ x)
            sigma = float(np.sqrt(x @ Ainv @ x))
            ucb = mu + self.alpha * sigma
            scores[a] = ucb
            if ucb > best_score:
                best_score = ucb
                best_arm = a
        assert best_arm is not None
        return best_arm, scores

    def update(self, arm: str, x: np.ndarray, reward: float):
        self.ensure_arm(arm)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


@dataclass
class ConstrainedBanditWrapper:
    """
    Lagrangian constrained bandit:
      maximize reward subject to average cost <= C and average latency <= T.

    We implement an online Lagrangian with shaped reward:
      shaped = reward - lam_cost * (cost - C) - lam_lat * (lat - T)
    """
    base: LinUCBBandit
    cost_budget: float
    latency_budget: float
    eta: float = 0.01

    def __post_init__(self):
        self.lam_cost = 0.0
        self.lam_lat = 0.0
        self.avg_cost = 0.0
        self.avg_lat = 0.0
        self.t = 0

    def select(self, x: np.ndarray, arms: List[str]) -> str:
        arm, _ = self.base.select(x, arms)
        return arm

    def update(self, arm: str, x: np.ndarray, reward: float, cost: float, latency: float):
        self.t += 1
        self.avg_cost += (cost - self.avg_cost) / self.t
        self.avg_lat += (latency - self.avg_lat) / self.t

        self.lam_cost = max(0.0, self.lam_cost + self.eta * (self.avg_cost - self.cost_budget))
        self.lam_lat = max(0.0, self.lam_lat + self.eta * (self.avg_lat - self.latency_budget))

        shaped = reward - self.lam_cost * (cost - self.cost_budget) - self.lam_lat * (latency - self.latency_budget)
        self.base.update(arm, x, shaped)
