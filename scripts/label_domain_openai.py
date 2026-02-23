#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm


TAXONOMY = [
    "math", "code", "knowledge_qa", "commonsense", "dialogue_chat",
    "reasoning", "summarization", "translation", "rag", "other",
]

SYSTEM = (
    "You are a fast classifier. "
    "Given a user prompt, output a single label from this list: "
    + ", ".join(TAXONOMY)
    + ". Output ONLY the label."
)


def classify(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt[:6000]},
        ],
        max_output_tokens=10,
        temperature=0.0,
    )
    label = (getattr(resp, "output_text", "") or "").strip().split()[0].lower()
    if label not in TAXONOMY:
        return "other"
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=str, required=True, help="stream.jsonl from build_stream.py")
    ap.add_argument("--out", type=str, required=True, help="output jsonl with domain_llm added")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    events = [json.loads(l) for l in Path(args.stream).read_text(encoding="utf-8").splitlines() if l.strip()]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        for ev in tqdm(events, desc="label_domain"):
            ev["domain_llm"] = classify(client, args.model, ev.get("prompt", ""))
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"[label_domain] wrote {len(events)} -> {outp}")


if __name__ == "__main__":
    main()
