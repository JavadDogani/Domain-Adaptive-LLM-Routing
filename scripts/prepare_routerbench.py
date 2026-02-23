#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from dacr.routerbench import load_routerbench_pkl, to_long_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--hf_repo", type=str, default="withmartian/routerbench")
    ap.add_argument("--hf_file", type=str, default="routerbench_0shot.pkl")
    ap.add_argument("--quality_suffix", type=str, default=None)
    ap.add_argument("--domain_col", type=str, default="eval_name")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] downloading {args.hf_repo}/{args.hf_file} ...")
    pkl_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file, repo_type="dataset")
    print(f"[prepare] loading {pkl_path}")
    wide = load_routerbench_pkl(pkl_path)

    long_df = to_long_table(
    wide,
    domain_col=args.domain_col,
    quality_suffix=args.quality_suffix,
    skip_missing_quality=True,
)
    long_df.to_parquet(out, index=False)
    print(f"[prepare] wrote {out} | rows={len(long_df)}")


if __name__ == "__main__":
    main()
