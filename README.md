# DACR RouterBench Pipeline (Domain-Adaptive Calibrated Routing)

This repository contains an end-to-end, reproducible pipeline for evaluating **LLM routing** under **accuracy–cost–latency** trade-offs with a focus on **OOD robustness via domain-adaptive calibration**. It is built around the **RouterBench (raw)** benchmark, which provides per-(query, model) **quality/performance** scores and **cost** labels, and it optionally attaches **measured latency telemetry** from real executions.

The pipeline produces:
- a standardized long-form dataset (`qid, prompt, domain, model, response, quality, cost`)
- a timestamped request stream (system-time replay)
- optional per-request latency logs (wall-clock)
- routing evaluation results (overall + per-domain metrics, plus plots)

> Important note on latency: RouterBench provides **quality/cost** for its benchmark model zoo. If you measure latency using external API endpoints (e.g., OpenAI models), model names will not match the RouterBench model identifiers. In that case, latency must be treated as a separate telemetry experiment (or you must serve the RouterBench models locally). This repository supports attaching latency logs when identifiers match.

---

## Project Structure

- `scripts/prepare_routerbench.py`  
  Downloads RouterBench from HuggingFace and converts it into a long-form Parquet file.

- `scripts/build_stream.py`  
  Builds a timestamped request stream from the dataset using **current system time** (configurable rate/bursts).

- `scripts/label_domain_openai.py` *(optional)*  
  Adds coarse domain labels using a small LLM classifier (requires API key; can be skipped since RouterBench already includes `eval_name`).

- `scripts/collect_latency_openai.py` *(optional)*  
  Executes model calls and logs **measured wall-clock latency** (and token usage if available). Requires API key.

- `scripts/run_experiment.py`  
  Runs offline routing evaluation (quality/cost from RouterBench) and optionally incorporates latency telemetry. Produces metrics CSVs and plots.

- `dacr/`  
  Core utilities: dataset parsing, feature extraction, predictors, calibration, policies, and metrics.

---

## Installation

Create and activate an environment:


python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## Quickstart: End-to-End Steps (1–6)

Step 1 — Install
- Install Python dependencies.
Command:
  pip install -r requirements.txt


Step 2 — Download + Prepare RouterBench (raw)
- Download routerbench_raw.pkl from HuggingFace and convert it into a long-form Parquet table.
- Output: data/routerbench_raw_long.parquet
  Columns: qid, prompt, domain, model, response, quality, cost
Command:
  python scripts/prepare_routerbench.py \
    --hf_file routerbench_raw.pkl \
    --quality_suffix performance \
    --out data/routerbench_raw_long.parquet

(Optional sanity check)
  python - <<'PY'
  import pandas as pd
  df = pd.read_parquet("data/routerbench_raw_long.parquet")
  print("rows:", len(df), "qids:", df.qid.nunique(), "models:", df.model.nunique())
  print(df["domain"].value_counts().head(15))
  PY


## Step 3 — Build a Streaming Trace with Current System Time
- Build a timestamped stream of queries for “online-style” replay.
- Timestamps are generated from current system time; inter-arrival times follow QPS and optional bursts.
- Output: data/stream.jsonl
Command:
  python scripts/build_stream.py \
  --data data/routerbench_raw_long.parquet \
  --split test \
  --ood_holdout "grade-school-math,mbpp" \
  --out data/stream.jsonl \
  --rate_qps 3 \
  --max_queries 1500 \
  --time_warp 200


## Step 4 — Collect Measured Latency Telemetry
- Execute real model calls for streamed prompts and log wall-clock latency (and token usage if available).
- Requires: OPENAI_API_KEY + configured models in configs/models.yaml
- Output: data/latency_log.jsonl
Command:
  python scripts/simulate_latency.py \
  --data data/routerbench_raw_long.parquet \
  --stream data/stream_ood.jsonl \
  --out data/latency_eff.jsonl \
  --out_base data/latency_base.jsonl \
  --token_method max \
  --chars_per_token 4.0 \
  --noise_cv 0.10

Note:
- RouterBench quality/cost correspond to its benchmark model zoo.
- If your latency log uses different model IDs (e.g., OpenAI models), latency cannot be joined directly unless you:
  (i) serve the same model IDs locally, or
  (ii) add an explicit alignment layer.


## Step 5 — Run Routing Experiments + Generate Metrics
- Run routing evaluation on IID or OOD splits.
- Produces aggregate metrics and per-domain metrics (and plots if enabled).
- Outputs: results_dacr/ (or your chosen --results_dir)
  - metrics.csv
  - metrics_by_domain.csv
  - calibration_ece.csv
  - plots (optional)
Command (example OOD holdout using domains present in RouterBench raw):
    python scripts/run_experiment.py \
  --data data/routerbench_raw_long.parquet \
  --latency_log data/latency_eff.jsonl \
  --results_dir results_dacr \
  --ood_holdout "grade-school-math,mbpp" \
  --tau_seconds 10 \
  --lambdas "1,5,10,20" \
  --lambda_lat 1.0 \
  --dacr_lambda_lat 1.0 \
  --dacr_budget_cost 0.0055 \
  --dacr_budget_viol 0.10 \
  --dacr_mu_lr 0.05 \
  --dacr_cal_lr 0.05 \
  --dacr_delay 10 \
  --dacr_p_miss 0.3

## Check outputs:
  ls -lh results_latency
  head -n 20 results_latency/metrics.csv
  head -n 20 results_latency/metrics_by_domain.csv