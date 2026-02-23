from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


def choose_cheapest(models: List[str], est_cost: Dict[str, float]) -> str:
    return min(models, key=lambda m: est_cost[m])


def choose_cascade(models_sorted: List[str], p_quality: Dict[str, float], threshold: float) -> str:
    for m in models_sorted:
        if p_quality[m] >= threshold:
            return m
    return models_sorted[-1]


def choose_predict_optimize(
    models: List[str],
    p_quality: Dict[str, float],
    cost: Dict[str, float],
    latency_s: Dict[str, float],
    lambda_cost: float,
    lambda_lat: float,
) -> str:
    best, best_u = None, -1e18
    for m in models:
        u = p_quality[m] - lambda_cost * cost[m] - lambda_lat * latency_s[m]
        if u > best_u:
            best, best_u = m, u
    assert best is not None
    return best


@dataclass
class RouteLLMBinary:
    weak: str
    strong: str
    threshold: float = 0.5

    def choose(self, p_strong_wins: float) -> str:
        return self.strong if p_strong_wins >= self.threshold else self.weak
