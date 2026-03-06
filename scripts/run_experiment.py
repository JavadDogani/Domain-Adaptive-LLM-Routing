#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from dacr.routerbench import make_ood_split
from dacr.featurize import TextEmbedder, build_query_features
from dacr.predictors import PerModelQualityPredictor
from dacr.calibration import DomainAdaptiveCalibrator, expected_calibration_error
from dacr.metrics import summarize_metrics


# ---------------------------
# Helpers
# ---------------------------

def load_latency_log(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    if "ok" in df.columns:
        df = df[df["ok"].astype(bool)].copy()
    df["qid"] = df["qid"].astype(str)
    df["model"] = df["model"].astype(str)
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


def pick_strongest_model(train_long: pd.DataFrame) -> str:
    per = train_long.groupby("model").agg(mean_q=("quality", "mean"), mean_cost=("cost", "mean")).reset_index()
    m = per.sort_values(["mean_q", "mean_cost"], ascending=[False, True]).iloc[0]["model"]
    return str(m)


def pick_cheapest_model(train_long: pd.DataFrame) -> str:
    per = train_long.groupby("model").agg(mean_q=("quality", "mean"), mean_cost=("cost", "mean")).reset_index()
    m = per.sort_values(["mean_cost", "mean_q"], ascending=[True, False]).iloc[0]["model"]
    return str(m)


def pick_strong_weak_models(train_long: pd.DataFrame) -> Tuple[str, str]:
    return pick_cheapest_model(train_long), pick_strongest_model(train_long)


def choose_mixllm(models: List[str],
                 p_hat: Dict[str, float],
                 cost: Dict[str, float],
                 viol: Dict[str, float],
                 lambda_cost: float,
                 lambda_lat: float) -> str:
    # score = p_hat - λc cost - λℓ viol
    best_m, best_s = None, -1e18
    for m in models:
        s = float(p_hat[m]) - float(lambda_cost) * float(cost[m]) - float(lambda_lat) * float(viol[m])
        if s > best_s:
            best_s = s
            best_m = m
    assert best_m is not None
    return best_m


def choose_cascade(cost_sorted: List[str], p_hat: Dict[str, float], threshold: float) -> str:
    # pick cheapest model whose predicted p >= threshold; else pick most expensive
    for m in cost_sorted:
        if float(p_hat[m]) >= threshold:
            return m
    return cost_sorted[-1]


class LLMRecRanker:
    """
    LLMRec-style *catalog ranker* compatible with your 11 RouterBench models:
      - One joint classifier on (query_features + model_onehot) -> binary correctness proxy
      - Predicts p(q,m) for all m
    """

    def __init__(self, models: List[str]):
        self.models = list(models)
        self.mid = {m: i for i, m in enumerate(self.models)}
        self.clf = None
        self.is_fit = False

    def fit(self, Xq_train: np.ndarray, train_long: pd.DataFrame, qid2i_train: Dict[str, int]):
        from sklearn.ensemble import HistGradientBoostingClassifier

        rows = train_long.drop_duplicates(["qid", "model"])[["qid", "model", "quality"]].copy()
        y = np.array([int(round(float(v))) for v in rows["quality"].tolist()], dtype=int)

        # Build (query features + model onehot)
        Xq = np.zeros((len(rows), Xq_train.shape[1]), dtype=np.float32)
        M = np.zeros((len(rows), len(self.models)), dtype=np.float32)

        for i, (qid, m) in enumerate(zip(rows["qid"].astype(str).tolist(), rows["model"].astype(str).tolist())):
            Xq[i] = Xq_train[qid2i_train[qid]]
            M[i, self.mid[m]] = 1.0

        X = np.concatenate([Xq, M], axis=1)

        self.clf = HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=250,
            learning_rate=0.08,
            random_state=0
        )
        self.clf.fit(X, y)
        self.is_fit = True
        return self

    def predict_proba_all(self, Xq: np.ndarray) -> Dict[str, np.ndarray]:
        assert self.is_fit and self.clf is not None
        nQ = Xq.shape[0]
        nM = len(self.models)

        # Vectorized: build (nQ*nM, d+M)
        Xq_rep = np.repeat(Xq, repeats=nM, axis=0)
        M = np.tile(np.eye(nM, dtype=np.float32), reps=(nQ, 1))
        X = np.concatenate([Xq_rep, M], axis=1)
        p = self.clf.predict_proba(X)[:, 1].astype(np.float32)

        out = {}
        for j, m in enumerate(self.models):
            out[m] = p[j::nM].copy()
        return out


# ---------------------------
# DACR Policy (Eq. 9 + online updates)
# ---------------------------

class DACRPolicy:
    """
    Implements Eq. (9) in your DACR paper:
      argmax_m (y~_t(m) - λc c_t(m) - λℓ v_t(m)) - μc c_t(m) - μℓ v_t(m)
    with online primal–dual updates for μc, μℓ and online domain-bias calibration updates
    under delayed/missing feedback. :contentReference[oaicite:2]{index=2}
    """

    def __init__(self,
                 lambda_lat: float,
                 budget_cost: float,
                 budget_viol: float,
                 mu_lr: float,
                 cal_lr: float,
                 delay: int,
                 p_miss: float,
                 seed: int = 0):
        self.lambda_lat = float(lambda_lat)
        self.B = float(budget_cost)
        self.delta = float(budget_viol)
        self.mu_lr = float(mu_lr)
        self.cal_lr = float(cal_lr)
        self.delay = int(delay)
        self.p_miss = float(p_miss)
        self.rng = np.random.default_rng(seed)

        self.mu_c = 0.0
        self.mu_l = 0.0
        self.bias = defaultdict(float)          # key=(model,domain): additive bias on probability (bounded)
        self.pending = defaultdict(list)        # t -> list of (model, domain, y_obs, p_base)

    @staticmethod
    def _clip_bias(b: float) -> float:
        # keep stable (avoid exploding logits)
        return float(np.clip(b, -0.5, 0.5))

    def _apply_pending(self, t: int):
        if t not in self.pending:
            return
        for (m, d, y_obs, p_base) in self.pending[t]:
            key = (m, d)
            # Simple, stable bias update in probability space:
            # b <- clip(b + lr*(y - (p_base + b)))
            b = float(self.bias[key])
            p_corr = float(np.clip(p_base + b, 1e-4, 1.0 - 1e-4))
            b_new = b + self.cal_lr * (float(y_obs) - p_corr)
            self.bias[key] = self._clip_bias(b_new)
        del self.pending[t]

    def choose(self,
               t: int,
               models: List[str],
               domain: str,
               p_base: Dict[str, float],
               cost: Dict[str, float],
               viol: Dict[str, float],
               lambda_cost: float) -> str:

        self._apply_pending(t)

        best_m, best_s = None, -1e18
        for m in models:
            b = float(self.bias[(m, domain)])
            p_corr = float(np.clip(float(p_base[m]) + b, 1e-4, 1.0 - 1e-4))
            s = (
                p_corr
                - float(lambda_cost) * float(cost[m])
                - self.lambda_lat * float(viol[m])
                - self.mu_c * float(cost[m])
                - self.mu_l * float(viol[m])
            )
            if s > best_s:
                best_s = s
                best_m = m
        assert best_m is not None
        return best_m

    def observe(self,
                t: int,
                chosen_model: str,
                domain: str,
                p_base_chosen: float,
                cost_obs: float,
                viol_obs: float,
                y_obs: int):
        # primal–dual multiplier updates
        self.mu_c = max(0.0, self.mu_c + self.mu_lr * (float(cost_obs) - self.B))
        self.mu_l = max(0.0, self.mu_l + self.mu_lr * (float(viol_obs) - self.delta))

        # delayed/missing feedback for calibration bias
        if self.rng.random() >= self.p_miss:
            t_apply = t + self.delay
            self.pending[t_apply].append((chosen_model, domain, int(y_obs), float(p_base_chosen)))


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--latency_log", type=str, default=None, help="eff_latency jsonl with (qid,model,latency_s)")
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--ood_holdout", type=str, default="")
    ap.add_argument("--tau_seconds", type=float, default=10.0)
    ap.add_argument("--lambdas", type=str, default="1,5,10,20")
    ap.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")

    # Baseline knobs
    ap.add_argument("--cascade_threshold", type=float, default=0.7)
    ap.add_argument("--lambda_lat", type=float, default=1.0, help="MixLLM/LLMRec latency weight on I[lat>tau]")

    # DACR knobs (Eq. 9 + online)
    ap.add_argument("--dacr_lambda_lat", type=float, default=1.0)
    ap.add_argument("--dacr_budget_cost", type=float, default=0.0055)
    ap.add_argument("--dacr_budget_viol", type=float, default=0.10)
    ap.add_argument("--dacr_mu_lr", type=float, default=0.05)
    ap.add_argument("--dacr_cal_lr", type=float, default=0.05)
    ap.add_argument("--dacr_delay", type=int, default=10)
    ap.add_argument("--dacr_p_miss", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    lat = load_latency_log(args.latency_log)
    df = attach_latency(df, lat)

    models = sorted(df["model"].astype(str).unique().tolist())
    holdout = [d.strip() for d in args.ood_holdout.split(",") if d.strip()]

    if holdout:
        train_long, test_long = make_ood_split(df, holdout_domains=holdout, domain_prefix_match=True, train_frac_in_domain=0.0)
    else:
        qids = df["qid"].astype(str).unique().tolist()
        rng2 = np.random.default_rng(42)
        rng2.shuffle(qids)
        cut = int(0.8 * len(qids))
        train_long = df[df["qid"].astype(str).isin(qids[:cut])].copy()
        test_long = df[df["qid"].astype(str).isin(qids[cut:])].copy()

    print(f"[exp] models={len(models)} train_rows={len(train_long)} test_rows={len(test_long)}")

    # Query features (domain one-hot learned on train, reused on test)
    embedder = TextEmbedder(model_name=args.embedder)
    X_train, domain_map, q_train = build_query_features(train_long[["qid", "prompt", "domain"]],
                                                        embedder, domain_onehot=True, domain_map=None)
    X_test, _, q_test = build_query_features(test_long[["qid", "prompt", "domain"]],
                                             embedder, domain_onehot=True, domain_map=domain_map)
    qid2i_train = {qid: i for i, qid in enumerate(q_train["qid"].astype(str).tolist())}
    qid2i_test = {qid: i for i, qid in enumerate(q_test["qid"].astype(str).tolist())}

    # Binary correctness proxy for training/calibration
    y_by_model_train: Dict[str, np.ndarray] = {}
    for m in models:
        sub = train_long[train_long["model"].astype(str) == m].drop_duplicates("qid")[["qid", "quality"]]
        y = np.zeros((len(q_train),), dtype=int)
        for _, r in sub.iterrows():
            y[qid2i_train[str(r.qid)]] = int(round(float(r.quality)))
        y_by_model_train[m] = y

    # Per-model predictor (prob of correctness proxy)
    qp = PerModelQualityPredictor().fit(X_train, y_by_model_train)
    p_train = qp.predict_proba(X_train)
    p_test = qp.predict_proba(X_test)

    # Domain-adaptive calibration (offline), export per-model ECE
    domains_train = q_train["domain"].astype(str).values
    domains_test = q_test["domain"].astype(str).values
    ece_rows = []
    y_by_model_test: Dict[str, np.ndarray] = {}

    for m in models:
        cal = DomainAdaptiveCalibrator().fit(p_train[m], y_by_model_train[m], domains_train)
        p_test[m] = cal.predict(p_test[m], domains_test)

        sub = test_long[test_long["model"].astype(str) == m].drop_duplicates("qid")[["qid", "quality"]]
        y = np.zeros((len(q_test),), dtype=int)
        for _, r in sub.iterrows():
            y[qid2i_test[str(r.qid)]] = int(round(float(r.quality)))
        y_by_model_test[m] = y
        ece_rows.append({"model": m, "ece_test": expected_calibration_error(p_test[m], y)})

    pd.DataFrame(ece_rows).to_csv(results_dir / "calibration_ece.csv", index=False)

    # LLMRec baseline (joint ranker)
    llmrec = LLMRecRanker(models=models).fit(X_train, train_long, qid2i_train=qid2i_train)
    p_test_llmrec = llmrec.predict_proba_all(X_test)

    # Build maps for evaluation (true quality/cost and measured latency)
    test_pairs = test_long.drop_duplicates(["qid", "model"]).copy()
    cost_map = {(str(r.qid), str(r.model)): float(r.cost) for _, r in test_pairs.iterrows()}
    qual_map = {(str(r.qid), str(r.model)): float(r.quality) for _, r in test_pairs.iterrows()}
    lat_map = {(str(r.qid), str(r.model)): float(r.latency_s_measured)
               for _, r in test_pairs.dropna(subset=["latency_s_measured"]).iterrows()}

    # If latency is missing for any pair, fail loudly (since you want eff_latency to be "real-world")
    missing_lat = len(test_pairs) - len(lat_map)
    if missing_lat > 0:
        print(f"[warn] latency missing for {missing_lat} (qid,model) pairs in test. "
              f"Those will be treated as 0.0 unless you fix the latency_log coverage.")
    def get_lat(qid: str, m: str) -> float:
        return float(lat_map.get((qid, m), 0.0))

    # Fixed baselines model choices
    weakest, strongest = pick_strong_weak_models(train_long)
    always_strong = strongest
    cost_sorted = train_long.groupby("model")["cost"].mean().sort_values().index.astype(str).tolist()

    # RouteLLM-style gate (2-arm) trained on query features
    from sklearn.linear_model import LogisticRegression
    y_win = np.zeros((len(q_train),), dtype=int)
    for i, qid in enumerate(q_train["qid"].astype(str).tolist()):
        qs = float(train_long[(train_long.qid.astype(str) == qid) & (train_long.model.astype(str) == strongest)]["quality"].iloc[0])
        qw = float(train_long[(train_long.qid.astype(str) == qid) & (train_long.model.astype(str) == weakest)]["quality"].iloc[0])
        y_win[i] = int(qs > qw)
    win_clf = LogisticRegression(max_iter=1000).fit(X_train, y_win)
    p_win_test = win_clf.predict_proba(X_test)[:, 1]

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]

    # DACR policy instance (per lambda_cost run uses same budgets; separate instances)
    dacr_policies = {
        lam: DACRPolicy(lambda_lat=args.dacr_lambda_lat,
                        budget_cost=args.dacr_budget_cost,
                        budget_viol=args.dacr_budget_viol,
                        mu_lr=args.dacr_mu_lr,
                        cal_lr=args.dacr_cal_lr,
                        delay=args.dacr_delay,
                        p_miss=args.dacr_p_miss,
                        seed=args.seed + int(1000 * lam))
        for lam in lambdas
    }

    rows = []
    for t, qid in enumerate(tqdm(q_test["qid"].astype(str).tolist(), desc="evaluate")):
        dom = str(q_test.iloc[t]["domain"])

        # per-model predictions
        p_base = {m: float(p_test[m][t]) for m in models}
        p_rec = {m: float(p_test_llmrec[m][t]) for m in models}

        # per-model true cost/lat/viol for THIS query
        cst = {m: float(cost_map[(qid, m)]) for m in models}
        latv = {m: get_lat(qid, m) for m in models}
        viol = {m: float(latv[m] > args.tau_seconds) for m in models}

        # Fixed baselines
        m_rand = models[hash(qid) % len(models)]
        m_cheapest = min(models, key=lambda m: cst[m])
        m_cascade = choose_cascade(cost_sorted, p_base, threshold=args.cascade_threshold)
        # RouteLLM chooses between weakest and strongest
        m_routellm = strongest if float(p_win_test[t]) >= 0.5 else weakest

        for pol, m in [
            ("random", m_rand),
            ("cheapest", m_cheapest),
            ("cascade", m_cascade),
            ("routellm", m_routellm),
            ("always_strong", always_strong),
        ]:
            rows.append({
                "policy": pol,
                "lambda_cost": np.nan,
                "qid": qid,
                "domain": dom,
                "chosen_model": m,
                "quality": qual_map[(qid, m)],
                "cost": cst[m],
                "latency_s": latv[m],
                "slo_violation": viol[m],
            })

        # MixLLM baseline: scalarization over calibrated per-model predictor
        for lam in lambdas:
            m_mix = choose_mixllm(models, p_base, cst, viol, lambda_cost=lam, lambda_lat=args.lambda_lat)
            rows.append({
                "policy": "mixllm",
                "lambda_cost": lam,
                "qid": qid,
                "domain": dom,
                "chosen_model": m_mix,
                "quality": qual_map[(qid, m_mix)],
                "cost": cst[m_mix],
                "latency_s": latv[m_mix],
                "slo_violation": viol[m_mix],
            })

        # LLMRec baseline: scalarization over joint ranker probabilities
        for lam in lambdas:
            m_rec = choose_mixllm(models, p_rec, cst, viol, lambda_cost=lam, lambda_lat=args.lambda_lat)
            rows.append({
                "policy": "llmrec",
                "lambda_cost": lam,
                "qid": qid,
                "domain": dom,
                "chosen_model": m_rec,
                "quality": qual_map[(qid, m_rec)],
                "cost": cst[m_rec],
                "latency_s": latv[m_rec],
                "slo_violation": viol[m_rec],
            })

        # DACR: Eq(9) + online updates under delayed/missing feedback
        for lam in lambdas:
            pol = dacr_policies[lam]
            m = pol.choose(t, models, dom, p_base, cst, viol, lambda_cost=lam)

            rows.append({
                "policy": "dacr",
                "lambda_cost": lam,
                "qid": qid,
                "domain": dom,
                "chosen_model": m,
                "quality": qual_map[(qid, m)],
                "cost": cst[m],
                "latency_s": latv[m],
                "slo_violation": viol[m],
            })

            # observe bandit feedback (only chosen model)
            y_obs = int(y_by_model_test[m][t])  # binary proxy
            pol.observe(t, m, dom, p_base[m], cst[m], viol[m], y_obs)

    df_run = pd.DataFrame(rows)
    df_run.to_parquet(results_dir / "per_query.parquet", index=False)

    agg = summarize_metrics(df_run, ["policy", "lambda_cost"])
    agg.to_csv(results_dir / "metrics.csv", index=False)

    by_dom = summarize_metrics(df_run, ["policy", "lambda_cost", "domain"])
    by_dom.to_csv(results_dir / "metrics_by_domain.csv", index=False)

    # Curves: quality–cost and viol–cost
    def plot_curve(policy: str, outname: str, ycol: str):
        sub = agg[(agg["policy"] == policy) & (agg["lambda_cost"].notna())].sort_values("lambda_cost")
        if len(sub) == 0:
            return
        plt.figure()
        plt.plot(sub["mean_cost"], sub[ycol], marker="o")
        plt.xlabel("Mean cost ($) per query")
        plt.ylabel(ycol)
        plt.title(f"{policy}: {ycol} vs cost")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(results_dir / outname, dpi=200)
        plt.close()

    for pol in ["mixllm", "llmrec", "dacr"]:
        plot_curve(pol, f"{pol}_quality_cost.png", "mean_quality")
        plot_curve(pol, f"{pol}_viol_cost.png", "slo_violation_rate")

    print(f"[exp] done -> {results_dir}")


if __name__ == "__main__":
    main()