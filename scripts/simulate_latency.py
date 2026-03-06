#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None


def load_stream(stream_path: str, mult_agg: str = "mean") -> pd.DataFrame:
    rows = [json.loads(l) for l in open(stream_path, "r", encoding="utf-8") if l.strip()]
    if not rows:
        raise ValueError(f"Empty stream: {stream_path}")

    s = pd.DataFrame(rows)
    if "qid" not in s.columns:
        raise ValueError("stream.jsonl must contain 'qid'.")
    if "lat_mult" not in s.columns:
        # allow older streams; default to 1
        s["lat_mult"] = 1.0
    s["qid"] = s["qid"].astype(str)
    s["lat_mult"] = pd.to_numeric(s["lat_mult"], errors="coerce").fillna(1.0)

    # aggregate if same qid appears multiple times in stream
    if mult_agg == "mean":
        agg = s.groupby("qid", as_index=False)["lat_mult"].mean()
    elif mult_agg == "max":
        agg = s.groupby("qid", as_index=False)["lat_mult"].max()
    else:
        raise ValueError("--mult_agg must be one of: mean, max")
    return agg


def load_latency_config(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    """
    YAML format:
      models:
        <model_name>:
          t_init_s: 0.6
          speed_tps: 60
    """
    if path is None or path == "":
        return {}
    if yaml is None:
        raise ImportError("PyYAML not installed. Install it or omit --lat_cfg.")
    cfg = yaml.safe_load(Path(path).read_text())
    models = cfg.get("models", {}) if isinstance(cfg, dict) else {}
    out: Dict[str, Dict[str, float]] = {}
    for m, v in models.items():
        if not isinstance(v, dict):
            continue
        t_init = float(v.get("t_init_s", np.nan))
        speed = float(v.get("speed_tps", np.nan))
        if np.isfinite(t_init) and np.isfinite(speed) and speed > 0:
            out[str(m)] = {"t_init_s": t_init, "speed_tps": speed}
    return out


def estimate_tokens(text: str, method: str = "max", chars_per_token: float = 4.0) -> int:
    """
    Cheap token estimate to avoid tokenizer dependency.
    - chars: len(text)/chars_per_token
    - words: len(text.split())
    - max: max(chars_est, words_est)
    """
    if text is None:
        return 1
    s = str(text)
    if not s:
        return 1
    chars_est = int(max(1, round(len(s) / max(1e-6, chars_per_token))))
    words_est = int(max(1, len(s.split())))
    if method == "chars":
        return chars_est
    if method == "words":
        return words_est
    return max(chars_est, words_est)


def derive_fallback_params_from_cost(
    model_mean_cost: Dict[str, float],
    default_t_init: float,
    default_speed: float,
    cost_ref: float,
) -> Dict[str, Dict[str, float]]:
    """
    If t_init/speed are missing for some model, derive plausible values from mean cost.
    Intuition: higher-cost models are generally slower and have larger overheads.

    speed_tps = default_speed * (cost_ref / cost_mean)^0.5  (clipped)
    t_init_s  = default_t_init * (cost_mean / cost_ref)^0.3 (clipped)
    """
    out = {}
    for m, c in model_mean_cost.items():
        c = float(c)
        # avoid division issues
        c_eff = max(c, 1e-9)
        speed = default_speed * (cost_ref / c_eff) ** 0.5
        t_init = default_t_init * (c_eff / max(cost_ref, 1e-9)) ** 0.3
        # clip to keep sane
        speed = float(np.clip(speed, 8.0, 250.0))
        t_init = float(np.clip(t_init, 0.05, 3.0))
        out[m] = {"t_init_s": t_init, "speed_tps": speed}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="data/routerbench_raw_long.parquet")
    ap.add_argument("--stream", type=str, required=True, help="data/stream.jsonl (should include lat_mult)")
    ap.add_argument("--out", type=str, required=True, help="output latency log jsonl (effective latency)")
    ap.add_argument("--out_base", type=str, default="", help="optional output latency log with lat_mult=1")
    ap.add_argument("--lat_cfg", type=str, default="", help="YAML with per-model t_init_s and speed_tps")
    ap.add_argument("--mult_agg", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--token_method", type=str, default="max", choices=["max", "chars", "words"])
    ap.add_argument("--chars_per_token", type=float, default=4.0)

    # MixLLM-style latency model knobs
    ap.add_argument("--default_t_init", type=float, default=0.35, help="fallback init overhead (seconds)")
    ap.add_argument("--default_speed_tps", type=float, default=60.0, help="fallback throughput (tokens/sec)")
    ap.add_argument("--noise_cv", type=float, default=0.10, help="multiplicative noise CV on latency")
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.data)

    required_cols = {"qid", "model", "response", "cost"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"--data missing required columns: {missing}. Found: {list(df.columns)}")

    df["qid"] = df["qid"].astype(str)
    df["model"] = df["model"].astype(str)

    # stream qids + per-qid latency multiplier
    s = load_stream(args.stream, mult_agg=args.mult_agg)
    qid_set = set(s["qid"].tolist())
    if not qid_set:
        raise ValueError("No qids found in stream.")

    # restrict to stream qids (so log size matches your evaluation subset if needed)
    sub = df[df["qid"].isin(qid_set)].copy()
    if len(sub) == 0:
        raise ValueError("No (qid,model) rows from --data match stream qids. Check qid formats.")

    # model-level mean cost for fallback param derivation
    model_mean_cost = sub.groupby("model")["cost"].mean().to_dict()
    cost_ref = float(np.median(list(model_mean_cost.values()))) if model_mean_cost else 1e-3

    # load per-model latency config if provided; fill missing using cost-based fallback
    cfg = load_latency_config(args.lat_cfg) if args.lat_cfg else {}
    fallback = derive_fallback_params_from_cost(
        model_mean_cost=model_mean_cost,
        default_t_init=float(args.default_t_init),
        default_speed=float(args.default_speed_tps),
        cost_ref=cost_ref,
    )

    params: Dict[str, Dict[str, float]] = {}
    for m in model_mean_cost:
        if m in cfg:
            params[m] = cfg[m]
        else:
            params[m] = fallback[m]

    # attach lat_mult to each qid
    sub = sub.merge(s, on="qid", how="left")
    sub["lat_mult"] = sub["lat_mult"].fillna(1.0)

    # estimate output tokens from response text (already in RouterBench raw)
    # note: could replace with tokenizer-based counting later
    sub["out_tokens_est"] = sub["response"].astype(str).apply(
        lambda t: estimate_tokens(t, method=args.token_method, chars_per_token=float(args.chars_per_token))
    )

    # MixLLM-style base latency: t_init + out_tokens / speed_tps
    def base_latency_row(m: str, out_tokens: int) -> float:
        p = params[m]
        return float(p["t_init_s"] + (float(out_tokens) / max(1e-6, float(p["speed_tps"]))))

    base_lat = np.array(
        [base_latency_row(m, ot) for m, ot in zip(sub["model"].tolist(), sub["out_tokens_est"].tolist())],
        dtype=float
    )

    # multiplicative noise (keeps ordering but avoids deterministic ties)
    if args.noise_cv > 0:
        eps = rng.normal(0.0, float(args.noise_cv), size=base_lat.shape[0])
        base_lat = np.maximum(0.0, base_lat * (1.0 + eps))

    eff_lat = base_lat * sub["lat_mult"].to_numpy(dtype=float)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        for qid, model, lat in zip(sub["qid"].tolist(), sub["model"].tolist(), eff_lat.tolist()):
            f.write(json.dumps({"qid": qid, "model": model, "latency_s": float(lat), "ok": True}) + "\n")

    if args.out_base:
        outb = Path(args.out_base)
        outb.parent.mkdir(parents=True, exist_ok=True)
        with outb.open("w", encoding="utf-8") as f:
            for qid, model, lat in zip(sub["qid"].tolist(), sub["model"].tolist(), base_lat.tolist()):
                f.write(json.dumps({"qid": qid, "model": model, "latency_s": float(lat), "ok": True}) + "\n")

    # summary
    print(f"[simulate_latency_mixllm] rows={len(sub)} qids={sub['qid'].nunique()} models={sub['model'].nunique()}")
    print(f"[simulate_latency_mixllm] out={outp}")
    if args.out_base:
        print(f"[simulate_latency_mixllm] out_base={args.out_base}")

    print(f"[simulate_latency_mixllm] lat_mult: min={sub['lat_mult'].min():.3f} mean={sub['lat_mult'].mean():.3f} max={sub['lat_mult'].max():.3f}")
    print(f"[simulate_latency_mixllm] base latency: min={base_lat.min():.3f}s mean={base_lat.mean():.3f}s p95={np.quantile(base_lat,0.95):.3f}s")
    print(f"[simulate_latency_mixllm] eff  latency: min={eff_lat.min():.3f}s mean={eff_lat.mean():.3f}s p95={np.quantile(eff_lat,0.95):.3f}s")


if __name__ == "__main__":
    main()