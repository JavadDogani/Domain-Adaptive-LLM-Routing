from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def summarize_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n=("qid", "nunique"),
        mean_quality=("quality", "mean"),
        mean_cost=("cost", "mean"),
        mean_latency_s=("latency_s", "mean"),
        p95_latency_s=("latency_s", lambda x: float(np.quantile(x, 0.95))),
        p99_latency_s=("latency_s", lambda x: float(np.quantile(x, 0.99))),
        slo_violation_rate=("slo_violation", "mean"),
    ).reset_index()
    return out
