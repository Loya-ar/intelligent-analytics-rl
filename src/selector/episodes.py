"""Episodes: a random slice of one dataset, its context features, and the reward of every arm.

Running all 7 analyses on each slice once and saving the result (the "reward table")
lets every selection method be replayed on exactly the same episodes in seconds.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .analyses import ARMS, run_arm
from .data import ROOT, Dataset, load_all

TABLE = ROOT / "reports" / "reward_table.csv"
SIZES = [400, 1500, 5000]
WINDOWS_DAYS = [3, 7, 14, 30]
MIN_ROWS = 200

# Context features shown to LinUCB (generic, so they transfer across datasets).
FEATURES = ["log_rows", "n_numeric", "has_date", "has_region", "has_product",
            "log_span_days", "is_time_slice", "target_zero_frac", "n_categories"]


def sample_slice(ds: Dataset, rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    """Pick a time window (datasets with dates, half the time) or a segment, then subsample."""
    df = ds.df
    if ds.date is not None and rng.random() < 0.5:
        days = int(rng.choice(WINDOWS_DAYS))
        lo, hi = df[ds.date].min(), df[ds.date].max() - pd.Timedelta(days=days)
        part = df.iloc[:0]
        while len(part) < MIN_ROWS:  # redraw windows that fall in a gap (e.g. holidays)
            start = lo + (hi - lo) * rng.random()
            part = df[(df[ds.date] >= start) & (df[ds.date] < start + pd.Timedelta(days=days))]
        desc = f"time window: {days} days from {start:%Y-%m-%d}"
    else:
        col = ds.segments[rng.integers(len(ds.segments))]
        counts = df[col].value_counts()
        counts = counts[counts >= 300]
        val = counts.index[rng.choice(len(counts), p=(counts / counts.sum()).values)]
        part = df[df[col] == val]
        desc = f"segment: {col} = {val}"
    n = int(rng.choice(SIZES))
    if len(part) > n:
        part = part.sample(n=n, random_state=int(rng.integers(2**31)))
    return part.reset_index(drop=True), desc


def context_features(part: pd.DataFrame, ds: Dataset, desc: str, span_days: float) -> dict:
    cats = [c for c in (ds.region, ds.product) if c is not None]
    return {
        "log_rows": float(np.log10(len(part))),
        "n_numeric": len(ds.numeric),
        "has_date": int(ds.date is not None),
        "has_region": int(ds.region is not None and part[ds.region].nunique() > 1),
        "has_product": int(ds.product is not None and part[ds.product].nunique() > 1),
        "log_span_days": float(np.log1p(span_days)),
        "is_time_slice": int(desc.startswith("time")),
        "target_zero_frac": float((part[ds.targets[0]] == 0).mean()),
        "n_categories": float(sum(part[c].nunique() for c in cats)),
    }


_RHO: dict[str, pd.DataFrame] = {}


def global_rho(ds: Dataset) -> pd.DataFrame:
    """Dataset-wide Spearman matrix (on a fixed 20k-row sample), cached per dataset."""
    if ds.name not in _RHO:
        sample = ds.df[ds.numeric].sample(n=min(len(ds.df), 20000), random_state=0)
        _RHO[ds.name] = sample.corr(method="spearman").fillna(0)
    return _RHO[ds.name]


def build_episode(ds: Dataset, episode_id: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, episode_id])
    part, desc = sample_slice(ds, rng)
    span_days = 0.0
    if ds.date is not None:
        span_days = (part[ds.date].max() - part[ds.date].min()).total_seconds() / 86400
    half = rng.permutation(len(part)) < len(part) // 2
    a, b = part[half], part[~half]
    ctx = {"global_means": ds.df[ds.targets].mean().to_dict(), "global_rho": global_rho(ds),
           "span_days": span_days,
           "seed": int(rng.integers(2**31))}
    row = {"episode": episode_id, "dataset": ds.name, "slice": desc, "rows": len(part)}
    row.update(context_features(part, ds, desc, span_days))
    for arm in ARMS:
        t0 = time.perf_counter()
        res = run_arm(arm, a, b, ds, ctx)
        row[f"r_{arm}"] = round(res["reward"], 6)
        row[f"rep_{arm}"], row[f"fail_{arm}"] = res["replicated"], res["failed"]
        row[f"sec_{arm}"] = round(time.perf_counter() - t0, 4)
    return row


def build_table(per_dataset: int = 300, seed: int = 0, path: Path = TABLE) -> pd.DataFrame:
    datasets = load_all()
    rows, eid = [], 0
    for name in sorted(datasets):
        t0 = time.time()
        for _ in range(per_dataset):
            rows.append(build_episode(datasets[name], eid, seed))
            eid += 1
        print(f"{name}: {per_dataset} episodes in {time.time() - t0:.0f}s")
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Runtimes vary machine to machine; keep them out of the committed table.
    table.drop(columns=[c for c in table if c.startswith("sec_")]).to_csv(path, index=False)
    table.filter(regex="^(episode|dataset|sec_)").to_csv(path.with_name("runtimes.csv"), index=False)
    return table


def load_table(path: Path = TABLE) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-dataset", type=int, default=300)
    args = p.parse_args()
    build_table(args.per_dataset)
