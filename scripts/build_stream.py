#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def parse_burst(burst: str) -> List[Tuple[float, float]]:
    if not burst:
        return []
    out = []
    for part in burst.split(","):
        qps_s, dur_s = part.split("@")
        out.append((float(qps_s), float(dur_s)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="routerbench_long.parquet")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    ap.add_argument("--ood_holdout", type=str, default="", help="comma-separated domains for OOD test split")
    ap.add_argument("--out", type=str, required=True, help="output jsonl stream")
    ap.add_argument("--rate_qps", type=float, default=2.0)
    ap.add_argument("--burst", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_queries", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    q = df.drop_duplicates("qid")[["qid", "prompt", "domain"]].copy()

    holdout = [d.strip() for d in args.ood_holdout.split(",") if d.strip()]

    if args.split != "all":
        if holdout:
            is_holdout = q["domain"].astype(str).apply(lambda d: any(d.startswith(h) for h in holdout))
            q = q[is_holdout] if args.split == "test" else q[~is_holdout]
        else:
            q = q.sample(frac=1.0, random_state=args.seed)
            cut = int(0.8 * len(q))
            q = q.iloc[cut:] if args.split == "test" else q.iloc[:cut]

    q = q.head(args.max_queries).reset_index(drop=True)
    print(f"[stream] queries={len(q)} split={args.split}")

    bursts = parse_burst(args.burst)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    events = []
    t = t0
    idx = 0

    # Optional bursts
    for qps, dur in bursts:
        n = int(qps * dur)
        for _ in range(n):
            if idx >= len(q):
                break
            t += 1.0 / qps
            row = q.iloc[idx]
            events.append({"ts": t, "qid": str(row.qid), "domain": str(row.domain), "prompt": str(row.prompt)})
            idx += 1

    # Remaining at base rate
    for i in range(idx, len(q)):
        t += 1.0 / args.rate_qps
        row = q.iloc[i]
        events.append({"ts": t, "qid": str(row.qid), "domain": str(row.domain), "prompt": str(row.prompt)})

    with out_path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[stream] wrote {len(events)} events -> {out_path}")


if __name__ == "__main__":
    main()
