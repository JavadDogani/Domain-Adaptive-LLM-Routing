#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from dacr.routerbench import make_ood_split
from dacr.featurize import TextEmbedder, build_query_features
from dacr.predictors import PerModelQualityPredictor, LatencyRegressor
from dacr.calibration import DomainAdaptiveCalibrator, expected_calibration_error
from dacr.policies import choose_cheapest, choose_cascade, choose_predict_optimize, RouteLLMBinary
from dacr.metrics import summarize_metrics


def load_latency_log(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df[df["ok"] == True].copy()
    df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce")
    return df.dropna(subset=["qid", "model", "latency_s"])


def attach_latency(df_long: pd.DataFrame, latency_log: pd.DataFrame) -> pd.DataFrame:
    if latency_log is None or len(latency_log) == 0:
        df_long["latency_s_measured"] = np.nan
        return df_long
    key = latency_log.groupby(["qid", "model"], as_index=False)["latency_s"].mean()
    out = df_long.merge(key, on=["qid", "model"], how="left")
    out = out.rename(columns={"latency_s": "latency_s_measured"})
    return out


def pick_strong_weak_models(train_long: pd.DataFrame) -> Tuple[str, str]:
    per = train_long.groupby("model").agg(mean_q=("quality", "mean"), mean_cost=("cost", "mean")).reset_index()
    strong = per.sort_values(["mean_q", "mean_cost"], ascending=[False, True]).iloc[0]["model"]
    weak = per.sort_values(["mean_cost", "mean_q"], ascending=[True, False]).iloc[0]["model"]
    return str(weak), str(strong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--latency_log", type=str, default=None)
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--ood_holdout", type=str, default="")
    ap.add_argument("--tau_seconds", type=float, default=30.0)
    ap.add_argument("--lambdas", type=str, default="0.05,0.1,0.2")
    ap.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    latency_log = load_latency_log(args.latency_log)
    df = attach_latency(df, latency_log)

    models = sorted(df["model"].unique().tolist())
    holdout = [d.strip() for d in args.ood_holdout.split(",") if d.strip()]

    if holdout:
        train_long, test_long = make_ood_split(df, holdout_domains=holdout, domain_prefix_match=True, train_frac_in_domain=0.0)
    else:
        qids = df["qid"].unique().tolist()
        rng = np.random.default_rng(42)
        rng.shuffle(qids)
        cut = int(0.8 * len(qids))
        train_long = df[df["qid"].isin(qids[:cut])].copy()
        test_long = df[df["qid"].isin(qids[cut:])].copy()

    print(f"[exp] models={len(models)} train_rows={len(train_long)} test_rows={len(test_long)}")

    embedder = TextEmbedder(model_name=args.embedder)
    X_train, domain_map, q_train = build_query_features(
    train_long[["qid","prompt","domain"]], embedder, domain_onehot=True, domain_map=None
    )
    X_test, _, q_test = build_query_features(
    test_long[["qid","prompt","domain"]], embedder, domain_onehot=True, domain_map=domain_map
    )
    qid2i_train = {qid: i for i, qid in enumerate(q_train["qid"].astype(str).tolist())}
    qid2i_test = {qid: i for i, qid in enumerate(q_test["qid"].astype(str).tolist())}

    # y per model on train
    y_by_model_train: Dict[str, np.ndarray] = {}
    for m in models:
        sub = train_long[train_long["model"] == m].drop_duplicates("qid")[["qid", "quality"]]
        y = np.zeros((len(q_train),), dtype=int)
        for _, r in sub.iterrows():
            y[qid2i_train[str(r.qid)]] = int(round(float(r.quality)))
        y_by_model_train[m] = y

    qp = PerModelQualityPredictor().fit(X_train, y_by_model_train)
    p_train = qp.predict_proba(X_train)
    p_test = qp.predict_proba(X_test)

    # Domain-adaptive calibration per model
    domains_train = q_train["domain"].astype(str).values
    domains_test = q_test["domain"].astype(str).values
    ece_rows = []
    for m in models:
        cal = DomainAdaptiveCalibrator().fit(p_train[m], y_by_model_train[m], domains_train)
        p_test[m] = cal.predict(p_test[m], domains_test)

        # build y on test
        sub = test_long[test_long["model"] == m].drop_duplicates("qid")[["qid", "quality"]]
        y = np.zeros((len(q_test),), dtype=int)
        for _, r in sub.iterrows():
            y[qid2i_test[str(r.qid)]] = int(round(float(r.quality)))
        ece_rows.append({"model": m, "ece_test": expected_calibration_error(p_test[m], y)})

    pd.DataFrame(ece_rows).to_csv(results_dir / "calibration_ece.csv", index=False)

    # Latency model (optional)
    lat_reg = LatencyRegressor()
    fallback_lat = {m: 0.0 for m in models}
    if "latency_s_measured" in train_long.columns and train_long["latency_s_measured"].notna().sum() >= 200:
        lat_rows = train_long.drop_duplicates(["qid","model"]).dropna(subset=["latency_s_measured"])
        # build X_lat = Xq + model-onehot
        mid = {m:i for i,m in enumerate(models)}
        Xq = np.zeros((len(lat_rows), X_train.shape[1]), dtype=np.float32)
        M = np.zeros((len(lat_rows), len(models)), dtype=np.float32)
        for i,(qid, m) in enumerate(zip(lat_rows["qid"].astype(str).tolist(), lat_rows["model"].tolist())):
            Xq[i] = X_train[qid2i_train[qid]]
            M[i, mid[m]] = 1.0
        X_lat = np.concatenate([Xq, M], axis=1)
        y_lat = lat_rows["latency_s_measured"].to_numpy(dtype=np.float32)
        lat_reg.fit(X_lat, y_lat)
        # fallback median per model
        for m in models:
            vals = lat_rows[lat_rows["model"]==m]["latency_s_measured"].to_numpy()
            if len(vals) > 0:
                fallback_lat[m] = float(np.median(vals))

    # cost and quality maps from test ground truth
    test_long2 = test_long.drop_duplicates(["qid","model"]).copy()
    cost_map = {(str(r.qid), str(r.model)): float(r.cost) if pd.notna(r.cost) else 0.0 for _, r in test_long2.iterrows()}
    qual_map = {(str(r.qid), str(r.model)): float(r.quality) if pd.notna(r.quality) else 0.0 for _, r in test_long2.iterrows()}
    lat_map = {(str(r.qid), str(r.model)): float(r.latency_s_measured) for _, r in test_long2.dropna(subset=["latency_s_measured"]).iterrows()}

    mid = {m:i for i,m in enumerate(models)}

    def est_latency(qid: str, model: str, xq: np.ndarray) -> float:
        if (qid, model) in lat_map:
            return lat_map[(qid, model)]
        if lat_reg.is_fit:
            M = np.zeros((len(models),), dtype=np.float32)
            M[mid[model]] = 1.0
            x = np.concatenate([xq, M], axis=0)[None, :]
            return float(lat_reg.predict(x)[0])
        return fallback_lat[model]

    weak, strong = pick_strong_weak_models(train_long)
    print(f"[exp] RouteLLM baseline: weak={weak} strong={strong}")

    # RouteLLM-style win predictor
    from sklearn.linear_model import LogisticRegression
    y_win = np.zeros((len(q_train),), dtype=int)
    for i,qid in enumerate(q_train["qid"].astype(str).tolist()):
        qs = float(train_long[(train_long.qid==qid) & (train_long.model==strong)]["quality"].iloc[0])
        qw = float(train_long[(train_long.qid==qid) & (train_long.model==weak)]["quality"].iloc[0])
        y_win[i] = int(qs > qw)
    win_clf = LogisticRegression(max_iter=1000).fit(X_train, y_win)
    p_win_test = win_clf.predict_proba(X_test)[:,1]

    models_sorted_by_cost = train_long.groupby("model")["cost"].mean().sort_values().index.tolist()
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]

    rows = []
    for qi, qid in enumerate(tqdm(q_test["qid"].astype(str).tolist(), desc="evaluate")):
        dom = str(q_test.iloc[qi]["domain"])
        xq = X_test[qi]
        pq = {m: float(p_test[m][qi]) for m in models}
        cst = {m: float(cost_map[(qid, m)]) for m in models}
        lat = {m: est_latency(qid, m, xq) for m in models}

        # fixed baselines
        m_rand = models[hash(qid) % len(models)]
        m_cheapest = choose_cheapest(models, cst)
        m_cascade = choose_cascade(models_sorted_by_cost, pq, threshold=0.7)
        m_routellm = RouteLLMBinary(weak=weak, strong=strong, threshold=0.5).choose(float(p_win_test[qi]))

        for pol, m in [("random", m_rand), ("cheapest", m_cheapest), ("cascade", m_cascade), ("routellm", m_routellm)]:
            rows.append({
                "policy": pol,
                "lambda_cost": np.nan,
                "qid": qid,
                "domain": dom,
                "chosen_model": m,
                "quality": qual_map[(qid, m)],
                "cost": cst[m],
                "latency_s": lat[m],
                "slo_violation": float(lat[m] > args.tau_seconds),
            })

        # predict+opt family (MixLLM-like scalarization)
        for lam in lambdas:
            m_opt = choose_predict_optimize(models, pq, cst, lat, lambda_cost=lam, lambda_lat=0.01)
            rows.append({
                "policy": "predict_optimize",
                "lambda_cost": lam,
                "qid": qid,
                "domain": dom,
                "chosen_model": m_opt,
                "quality": qual_map[(qid, m_opt)],
                "cost": cst[m_opt],
                "latency_s": lat[m_opt],
                "slo_violation": float(lat[m_opt] > args.tau_seconds),
            })

    df_run = pd.DataFrame(rows)
    df_run.to_parquet(results_dir / "per_query.parquet", index=False)

    agg = summarize_metrics(df_run, ["policy", "lambda_cost"])
    agg.to_csv(results_dir / "metrics.csv", index=False)

    by_dom = summarize_metrics(df_run, ["policy", "lambda_cost", "domain"])
    by_dom.to_csv(results_dir / "metrics_by_domain.csv", index=False)

    # Plots
    sub = agg[agg["policy"]=="predict_optimize"].sort_values("lambda_cost")
    if len(sub) > 0:
        plt.figure()
        plt.plot(sub["mean_cost"], sub["mean_quality"], marker="o")
        plt.xlabel("Mean cost (USD)")
        plt.ylabel("Mean quality")
        plt.title("Quality–Cost curve (predict_optimize)")
        plt.tight_layout()
        plt.savefig(results_dir / "curves_quality_cost.png", dpi=200)
        plt.close()

    fixed = agg[agg["policy"].isin(["random","cheapest","cascade","routellm"])]
    if len(fixed) > 0:
        plt.figure()
        plt.plot(fixed["policy"], fixed["p95_latency_s"], marker="o", label="p95")
        plt.plot(fixed["policy"], fixed["p99_latency_s"], marker="o", label="p99")
        plt.ylabel("Latency (s)")
        plt.title("Latency tail")
        plt.xticks(rotation=20)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "latency_tail.png", dpi=200)
        plt.close()

    print(f"[exp] done -> {results_dir}")


if __name__ == "__main__":
    main()
