"""LLM chooser: Gemini picks one analysis from a short summary of the data slice.

It gets no reward feedback (zero-shot), and is scored on the same held-out replication
reward as the bandits. Every response is cached in reports/llm_cache.json, keyed by a
hash of model + prompt, so reruns (and CI) are free and give identical results.

Live calls need GEMINI_API_KEY in the environment (free tier is enough). The key is
read from the environment only; it is never written anywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import requests

from .analyses import ARMS, COST, FAIL_PENALTY, MAX_CLAIMS
from .data import ROOT

CACHE = ROOT / "reports" / "llm_cache.json"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
N_EPISODES = 100          # 25 per dataset, fixed sample
MIN_INTERVAL_S = 6.5      # stay under free-tier requests-per-minute limits

SCHEMA = {
    "hillstrom": "Email marketing experiment (one row per customer). Numeric: recency, history (past spend), "
                 "mens, womens, newbie, visit, conversion, spend. Categorical: zip_code (Urban/Suburban/Rural), "
                 "segment (email offer received), channel, history_segment. No date column.",
    "telco": "Telecom customers (one row per customer). Numeric: MonthlyCharges, Churn (0/1), tenure, TotalCharges, "
             "SeniorCitizen. Categorical: InternetService (product), Contract, PaymentMethod, gender, "
             "PaperlessBilling, Partner. No date or region column.",
    "retail": "Online retailer invoice lines. Numeric: revenue, Quantity, Price, hour. Date: InvoiceDate. "
              "Categorical: Country (region), StockCode (product).",
    "taxi": "NYC green taxi trips, January 2024. Numeric: fare_amount, trip_distance, tip_amount, total_amount, "
            "trip_minutes, passenger_count. Date: pickup datetime. Categorical: Borough (region), "
            "payment_type (card/cash/...).",
}

ARM_HELP = {
    "descriptive_stats": "is the slice's mean of each outcome different from the whole dataset's mean",
    "correlation": "Spearman correlations between numeric columns (pairs already strongly correlated across the "
                   "whole dataset don't count)",
    "time_series": "does each outcome vary by hour of day / day of week (needs a date column)",
    "regional_comparison": "best vs worst region on each outcome (needs a region column)",
    "product_performance": "each of the top products vs the rest, per outcome (needs a product column)",
    "clustering": "k-means customer/record segments that are stable across both halves",
    "trend": "linear trend over time that forecasts the later 30% better than a naive forecast (needs a date)",
}


def build_prompt(row: dict) -> str:
    arms = "\n".join(f"- {a}: {ARM_HELP[a]} (cost {COST[a]})" for a in ARMS)
    return (
        "You choose ONE analysis to run on a slice of a business dataset.\n\n"
        f"Dataset: {SCHEMA[row['dataset']]}\n"
        f"Slice: {row['slice']}, {row['rows']} rows. Share of zero values in the main outcome: "
        f"{row['target_zero_frac']:.2f}.\n\n"
        "How the choice is scored: the slice is split in half. The analysis makes claims on the first half "
        "(statistical tests with p < 0.05); each claim is re-tested on the second half. "
        f"Score = (replicated - {FAIL_PENALTY} x failed) / {MAX_CLAIMS} - cost, counting at most "
        f"{MAX_CLAIMS} claims. An analysis that can't run on this data scores -cost.\n\n"
        f"Analyses:\n{arms}\n\n"
        'Reply with JSON only: {"analysis": "<one name from the list>"}'
    )


def _key(prompt: str) -> str:
    return hashlib.sha256(f"{MODEL}\n{prompt}".encode()).hexdigest()


def load_cache() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def _call(prompt: str, api_key: str) -> dict:
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    for attempt in range(6):
        t0 = time.perf_counter()
        resp = requests.post(URL.format(model=MODEL), json=body, timeout=60,
                             headers={"x-goog-api-key": api_key})
        latency = time.perf_counter() - t0
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(15 * (attempt + 1))
            continue
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return {"text": text, "latency_s": round(latency, 3),
                "prompt_tokens": usage.get("promptTokenCount"),
                "output_tokens": usage.get("candidatesTokenCount"),
                "total_tokens": usage.get("totalTokenCount")}
    raise RuntimeError("Gemini kept returning rate-limit or server errors")


def parse_arm(text: str) -> int | None:
    try:
        name = json.loads(text).get("analysis", "")
    except (json.JSONDecodeError, AttributeError):
        name = text
    name = str(name).strip().lower()
    for i, a in enumerate(ARMS):
        if a in name:
            return i
    return None


def episode_sample(table) -> list[int]:
    rng = np.random.default_rng(7)
    per = N_EPISODES // table["dataset"].nunique()
    idx = []
    for _, g in table.groupby("dataset"):
        idx += sorted(rng.choice(g.index.to_numpy(), size=per, replace=False).tolist())
    return idx


def choose_all(table, allow_live: bool = True) -> tuple[list[int], dict] | None:
    """Return the LLM's arm for each sampled episode (None if the cache is incomplete and
    no key is set) plus the cache entries used."""
    cache = load_cache()
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) if allow_live else None
    idx, used, last = episode_sample(table), {}, 0.0
    for i in idx:
        prompt = build_prompt(table.loc[i].to_dict())
        k = _key(prompt)
        if k not in cache:
            if not api_key:
                print(f"LLM: {len(used)}/{len(idx)} decisions cached; set GEMINI_API_KEY to fill the rest.")
                return None
            time.sleep(max(0.0, MIN_INTERVAL_S - (time.time() - last)))
            last = time.time()
            cache[k] = {"model": MODEL, "episode": int(table.loc[i, "episode"]), **_call(prompt, api_key)}
            CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8", newline="\n")
            print(f"LLM {len(used) + 1}/{len(idx)}: {cache[k]['text'].strip()} ({cache[k]['latency_s']:.2f}s)")
        used[k] = cache[k]
    return idx, used
