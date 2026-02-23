# scripts/dacr/routerbench.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Wide-schema suffix candidates
RESPONSE_SUFFIX_CANDIDATES = [
    "model_response", "response", "output", "output_text", "completion", "completion_text", "text", "generation"
]
COST_SUFFIX_CANDIDATES = ["total_cost", "cost", "cost_usd", "usd_cost"]
QUALITY_SUFFIX_CANDIDATES = [
    "performance", "perf",
    "is_correct", "correct", "accuracy",
    "score", "final_score", "metric", "quality"
]

def make_ood_split(
    df_long: pd.DataFrame,
    holdout_domains: List[str],
    domain_prefix_match: bool = True,
    train_frac_in_domain: float = 0.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create an OOD split by holding out queries whose domain matches holdout_domains.

    Args:
      df_long: long table with columns [qid, domain, ...]
      holdout_domains: list of domain names/prefixes to hold out for test
      domain_prefix_match: if True, treat each holdout string as a prefix
      train_frac_in_domain: fraction of holdout-domain queries to leak into train (0 = strict OOD)
      seed: RNG seed

    Returns:
      train_df, test_df (both long tables)
    """
    rng = np.random.default_rng(seed)
    domains = df_long["domain"].astype(str)

    if domain_prefix_match:
        is_holdout = domains.apply(lambda d: any(d.startswith(h) for h in holdout_domains))
    else:
        is_holdout = domains.isin(holdout_domains)

    holdout_qids = df_long.loc[is_holdout, "qid"].astype(str).unique().tolist()
    in_qids = df_long.loc[~is_holdout, "qid"].astype(str).unique().tolist()

    rng.shuffle(holdout_qids)
    k = int(len(holdout_qids) * float(train_frac_in_domain))
    holdout_train = set(holdout_qids[:k])
    holdout_test = set(holdout_qids[k:])

    train_qids = set(in_qids) | holdout_train
    test_qids = holdout_test

    train = df_long[df_long["qid"].astype(str).isin(train_qids)].copy()
    test = df_long[df_long["qid"].astype(str).isin(test_qids)].copy()
    return train, test


def _lower_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Map lowercase column name -> original column name."""
    return {c.lower(): c for c in df.columns}


def _find_exact_col_ci(df: pd.DataFrame, colname: str) -> Optional[str]:
    lc = _lower_cols(df)
    return lc.get(colname.lower())


def _find_suffix_col_ci(df: pd.DataFrame, model: str, suffixes: List[str]) -> Optional[str]:
    """
    Case-insensitive lookup for f"{model}|{suffix}".
    """
    lc = _lower_cols(df)
    for suf in suffixes:
        key = f"{model}|{suf}".lower()
        if key in lc:
            return lc[key]
    return None


def _infer_models_from_wide(df: pd.DataFrame) -> List[str]:
    """
    Infer model names in wide schema by finding columns ending with any response suffix candidate.
    """
    models = set()
    for c in df.columns:
        if "|" not in c:
            continue
        prefix, suf = c.split("|", 1)
        if suf.lower() in {s.lower() for s in RESPONSE_SUFFIX_CANDIDATES}:
            models.add(prefix)
    return sorted(models)


def load_routerbench_pkl(pkl_path: str | Path) -> pd.DataFrame:
    """
    Load routerbench pickle. Some pickles store a DF directly; others store a dict with multiple DFs.
    We choose the most "routerbench-like" DF.
    """
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, pd.DataFrame):
        return obj

    # If dict: pick the dataframe that has the most relevant columns
    if isinstance(obj, dict):
        dfs = []
        for k, v in obj.items():
            if isinstance(v, pd.DataFrame):
                dfs.append((k, v))
        if not dfs:
            raise TypeError(f"Pickle dict contains no DataFrame values. Keys={list(obj.keys())[:20]}")

        def score(df: pd.DataFrame) -> int:
            cols = {c.lower() for c in df.columns}
            s = 0
            # long schema hints
            for name in ["model", "prompt", "question", "query", "performance", "total_cost", "cost"]:
                if name in cols:
                    s += 5
            # wide schema hints
            if any("|" in c for c in df.columns):
                s += 3
            if any(c.lower().endswith("|model_response") for c in df.columns):
                s += 10
            return s

        dfs.sort(key=lambda kv: score(kv[1]), reverse=True)
        return dfs[0][1]

    raise TypeError(f"Unsupported pickle type: {type(obj)}")


def to_long_table(
    df: pd.DataFrame,
    quality_suffix: Optional[str] = None,
    domain_col: str = "eval_name",
    skip_missing_quality: bool = False,
) -> pd.DataFrame:
    """
    Convert RouterBench to long table with columns:
      qid, prompt, domain, model, response, cost, quality
    Supports both:
      - long schema (already has 'model' column), and
      - wide schema ({model}|{suffix} columns).
    """

    # ---------- 0) Detect LONG schema ----------
    cols_l = {c.lower() for c in df.columns}
    has_model_col = ("model" in cols_l) or ("model_name" in cols_l)

    # Identify prompt column
    prompt_col = None
    for c in ["prompt", "question", "query", "input", "instruction"]:
        cc = _find_exact_col_ci(df, c)
        if cc:
            prompt_col = cc
            break

    # Identify domain column
    dom_col = _find_exact_col_ci(df, domain_col) or _find_exact_col_ci(df, "domain")

    # Identify id column
    id_col = _find_exact_col_ci(df, "sample_id") or _find_exact_col_ci(df, "id") or _find_exact_col_ci(df, "qid")

    if has_model_col and prompt_col and id_col:
        model_col = _find_exact_col_ci(df, "model") or _find_exact_col_ci(df, "model_name")
        # response/cost/quality column candidates in long schema
        resp_col = None
        for c in ["model_response", "response", "output_text", "output", "completion", "text"]:
            cc = _find_exact_col_ci(df, c)
            if cc:
                resp_col = cc
                break

        cost_col = None
        for c in ["total_cost", "cost", "cost_usd", "usd_cost"]:
            cc = _find_exact_col_ci(df, c)
            if cc:
                cost_col = cc
                break

        q_col = None
        if quality_suffix:
            q_col = _find_exact_col_ci(df, quality_suffix)
        if not q_col:
            for c in ["performance", "perf", "quality", "accuracy", "score", "final_score", "is_correct", "correct"]:
                cc = _find_exact_col_ci(df, c)
                if cc:
                    q_col = cc
                    break

        if not resp_col:
            raise ValueError("Long schema detected (has 'model'), but cannot find response column.")
        if not cost_col:
            raise ValueError("Long schema detected (has 'model'), but cannot find cost column.")
        if not q_col:
            raise ValueError(
                "Long schema detected (has 'model'), but cannot find quality/performance column. "
                "Pass --quality_suffix if your column has a different name."
            )

        out = pd.DataFrame({
            "qid": df[id_col].astype(str),
            "prompt": df[prompt_col].astype(str),
            "domain": df[dom_col].astype(str) if dom_col else "unknown",
            "model": df[model_col].astype(str),
            "response": df[resp_col].astype(str),
            "cost": pd.to_numeric(df[cost_col], errors="coerce"),
            "quality": pd.to_numeric(df[q_col], errors="coerce"),
        })

        out["prompt_len_chars"] = out["prompt"].str.len()
        out["resp_len_chars"] = out["response"].str.len()
        return out

    # ---------- 1) Otherwise treat as WIDE schema ----------
    if not id_col:
        raise ValueError("Wide schema: missing id column (expected sample_id/id/qid).")
    if not prompt_col:
        raise ValueError("Wide schema: missing prompt column (prompt/question/query/input/instruction).")

    if not dom_col:
        # create fallback domain
        dom_col = None

    models = _infer_models_from_wide(df)
    if not models:
        # Provide helpful debug dump
        sample_cols = list(df.columns[:50])
        raise ValueError(
            "No model columns found in wide schema. "
            "Expected columns like '<model>|model_response' or '<model>|response/output_text/...'.\n"
            f"First 50 columns: {sample_cols}"
        )

    rows = []
    for m in models:
        resp_col = _find_suffix_col_ci(df, m, RESPONSE_SUFFIX_CANDIDATES)
        cost_col = _find_suffix_col_ci(df, m, COST_SUFFIX_CANDIDATES)

        if quality_suffix:
            q_col = _find_suffix_col_ci(df, m, [quality_suffix])
        else:
            q_col = None
        if not q_col:
            q_col = _find_suffix_col_ci(df, m, QUALITY_SUFFIX_CANDIDATES)

        if resp_col is None or cost_col is None:
            # skip broken model entries
            continue

        if q_col is None:
            if skip_missing_quality:
                print(f"[warn] skipping model {m}: no quality column")
                continue
            else:
                # print available suffixes for this model
                suf = sorted({c.split("|",1)[1] for c in df.columns if isinstance(c,str) and c.startswith(m+"|")})
                raise ValueError(
                    f"Missing quality column for {m}. "
                    f"Available suffixes: {suf}. "
                    f"Try --quality_suffix <one of the above> or use --skip_missing_quality."
                )

        sub = pd.DataFrame({
            "qid": df[id_col].astype(str),
            "prompt": df[prompt_col].astype(str),
            "domain": df[dom_col].astype(str) if dom_col else "unknown",
            "model": m,
            "response": df[resp_col].astype(str),
            "cost": pd.to_numeric(df[cost_col], errors="coerce"),
            "quality": pd.to_numeric(df[q_col], errors="coerce"),
        })
        rows.append(sub)

    if not rows:
        raise ValueError(
            "No models with (response,cost,quality) were found. "
            "This RouterBench artifact likely lacks quality/performance for all models "
            "(e.g., 0shot/5shot in your current snapshot). "
            "If using raw.pkl, it may be long-schema with different column names; "
            "run the inspection snippet below to see df.columns."
        )

    out = pd.concat(rows, ignore_index=True)
    out["prompt_len_chars"] = out["prompt"].str.len()
    out["resp_len_chars"] = out["response"].str.len()
    return out