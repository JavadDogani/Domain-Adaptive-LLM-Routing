#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
from openai import AsyncOpenAI
from tqdm import tqdm


@dataclass
class ModelSpec:
    name: str
    provider: str
    price_in: float
    price_out: float


def load_model_catalog(path: str) -> Dict[str, ModelSpec]:
    cfg = yaml.safe_load(Path(path).read_text())
    out: Dict[str, ModelSpec] = {}
    for name, d in cfg.get("models", {}).items():
        out[name] = ModelSpec(
            name=name,
            provider=d["provider"],
            price_in=float(d.get("price_per_million_input", 0.0)),
            price_out=float(d.get("price_per_million_output", 0.0)),
        )
    return out


def cost_from_usage(spec: ModelSpec, prompt_tokens: int, completion_tokens: int) -> float:
    return (spec.price_in * prompt_tokens + spec.price_out * completion_tokens) / 1_000_000.0


async def call_openai(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    resp = await client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    t1 = time.perf_counter()

    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
    completion_tokens = getattr(usage, "output_tokens", None) if usage else None

    return {
        "latency_s": t1 - t0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "response_id": getattr(resp, "id", None),
        "ok": True,
        "error": None,
    }


async def worker(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    spec: ModelSpec,
    event: Dict[str, Any],
    max_output_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    async with sem:
        try:
            res = await call_openai(client, spec.name, event["prompt"], max_output_tokens, temperature)
            pt = int(res["prompt_tokens"] or 0)
            ct = int(res["completion_tokens"] or 0)
            cost = cost_from_usage(spec, pt, ct) if (pt and ct) else None
            return {**event, "model": spec.name, **res, "cost_usd": cost}
        except Exception as e:
            return {**event, "model": spec.name, "ok": False, "error": str(e), "latency_s": None,
                    "prompt_tokens": None, "completion_tokens": None, "cost_usd": None}


async def main_async(args):
    catalog = load_model_catalog(args.config)
    specs = []
    for m in args.models:
        if m not in catalog:
            raise ValueError(f"Model '{m}' not in config: {args.config}")
        if catalog[m].provider != "openai":
            raise ValueError("This script currently supports provider=openai only.")
        specs.append(catalog[m])

    events = [json.loads(l) for l in Path(args.stream).read_text(encoding="utf-8").splitlines() if l.strip()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    # If per_event_models==1, sample a single model per event (cheap); else call all.
    rows_written = 0
    for ev in tqdm(events, desc="events"):
        if args.per_event_models == 1 and len(specs) > 1:
            idx = (hash(ev["qid"]) % len(specs))
            chosen = [specs[idx]]
        else:
            chosen = specs

        tasks = [worker(sem, client, s, ev, args.max_output_tokens, args.temperature) for s in chosen]
        for r in await asyncio.gather(*tasks):
            with out_path.open("a", encoding="utf-8") as fo:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"[latency] wrote {rows_written} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=str, required=True)
    ap.add_argument("--models", type=str, nargs="+", required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_output_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--per_event_models", type=int, default=1, choices=[1, 999])
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
