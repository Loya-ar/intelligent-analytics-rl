"""The 7 analyses (bandit arms) and the held-out replication reward.

Each analysis looks at the *discovery* half of a data slice and makes claims, each one a
hypothesis test with p < 0.05. Each claim is then re-tested on the *held-out* half.

    reward = (replicated - 0.5 * failed) / 10 - cost[arm]

* Only the 5 most significant claims on the discovery half count (so reward <= 1), which
  keeps analyses that test many things (all column pairs) from winning on volume alone.
* A claim replicates when the held-out test has p < 0.05 and the effect points the same
  way (details per analysis below).
* cost is a fixed per-analysis compute charge (fixed, not timed, so rewards are exactly
  reproducible). An analysis that can't run on a slice (no date column, say) scores -cost.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import Dataset

ALPHA = 0.05
MAX_CLAIMS = 5
KNOWN_RHO = 0.5
FAIL_PENALTY = 0.5
MIN_GROUP = 20

ARMS = ["descriptive_stats", "correlation", "time_series", "regional_comparison",
        "product_performance", "clustering", "trend"]

# Fixed compute charge per analysis (roughly proportional to measured runtime).
COST = {"descriptive_stats": 0.005, "correlation": 0.01, "time_series": 0.01,
        "regional_comparison": 0.01, "product_performance": 0.015, "clustering": 0.03,
        "trend": 0.01}


def reward_from_claims(replicated: int, failed: int, arm: str) -> float:
    return (replicated - FAIL_PENALTY * failed) / MAX_CLAIMS - COST[arm]


def _keep_top(claims: list[tuple[float, bool]]) -> tuple[int, int]:
    """claims = [(p_discovery, replicated)]; keep the MAX_CLAIMS smallest p-values."""
    claims = sorted(claims, key=lambda c: c[0])[:MAX_CLAIMS]
    rep = sum(1 for _, ok in claims if ok)
    return rep, len(claims) - rep


def _ok(p: float) -> bool:
    return bool(np.isfinite(p) and p < ALPHA)


def _const(x: pd.Series) -> bool:
    return x.nunique(dropna=True) < 2


# 1. Descriptive stats: is this slice's mean different from the dataset-wide mean?
def descriptive_stats(a, b, ds: Dataset, ctx):
    claims = []
    for t in ds.targets:
        mu = ctx["global_means"][t]
        xa, xb = a[t].dropna(), b[t].dropna()
        if len(xa) < MIN_GROUP or _const(xa) or _const(xb):
            continue
        ta, pa = stats.ttest_1samp(xa, mu)
        if _ok(pa):
            tb, pb = stats.ttest_1samp(xb, mu)
            claims.append((pa, _ok(pb) and np.sign(tb) == np.sign(ta)))
    return claims


# 2. Correlation: Spearman correlation between pairs of numeric columns. Pairs already
#    strongly correlated in the full dataset (|rho| >= 0.5, e.g. fare vs total amount) are
#    known facts, not findings, and are skipped.
def correlation(a, b, ds: Dataset, ctx):
    cols = [c for c in ds.numeric if not _const(a[c]) and not _const(b[c])]
    known = ctx["global_rho"]
    claims = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(known.loc[cols[i], cols[j]]) >= KNOWN_RHO:
                continue
            ra, pa = stats.spearmanr(a[cols[i]], a[cols[j]])
            if _ok(pa) and abs(ra) >= 0.1:
                rb, pb = stats.spearmanr(b[cols[i]], b[cols[j]])
                claims.append((pa, _ok(pb) and np.sign(rb) == np.sign(ra)))
    return claims


def _season_bins(df, ds, span_days):
    ts = df[ds.date]
    return ts.dt.hour if span_days <= 14 else ts.dt.day_of_week


# 3. Time series: does the target vary by hour of day (short slices) or day of week?
def time_series(a, b, ds: Dataset, ctx):
    if ds.date is None:
        return []
    claims = []
    ba, bb = _season_bins(a, ds, ctx["span_days"]), _season_bins(b, ds, ctx["span_days"])
    for t in ds.targets:
        ga = [g.values for _, g in a[t].groupby(ba) if len(g) >= 5]
        gb = [g.values for _, g in b[t].groupby(bb) if len(g) >= 5]
        if len(ga) < 3 or len(gb) < 3 or _const(a[t]) or _const(b[t]):
            continue
        pa = stats.kruskal(*ga).pvalue
        if _ok(pa):
            pb = stats.kruskal(*gb).pvalue
            ma, mb = a[t].groupby(ba).mean(), b[t].groupby(bb).mean()
            common = ma.index.intersection(mb.index)
            same_shape = len(common) >= 3 and np.corrcoef(ma[common], mb[common])[0, 1] > 0
            claims.append((pa, _ok(pb) and bool(same_shape)))
    return claims


def _best_vs_worst(a, b, col, t):
    ca = a[col].value_counts()
    groups = ca[ca >= MIN_GROUP].index
    if len(groups) < 2:
        return None
    means = a[a[col].isin(groups)].groupby(col)[t].mean()
    hi, lo = means.idxmax(), means.idxmin()
    xa_hi, xa_lo = a.loc[a[col] == hi, t], a.loc[a[col] == lo, t]
    if _const(pd.concat([xa_hi, xa_lo])):
        return None
    pa = stats.mannwhitneyu(xa_hi, xa_lo).pvalue
    if not _ok(pa):
        return None
    xb_hi, xb_lo = b.loc[b[col] == hi, t], b.loc[b[col] == lo, t]
    if len(xb_hi) < 5 or len(xb_lo) < 5 or _const(pd.concat([xb_hi, xb_lo])):
        return pa, False
    pb = stats.mannwhitneyu(xb_hi, xb_lo).pvalue
    return pa, _ok(pb) and xb_hi.mean() > xb_lo.mean()


# 4. Regional comparison: best vs worst region on each target.
def regional_comparison(a, b, ds: Dataset, ctx):
    if ds.region is None:
        return []
    out = [_best_vs_worst(a, b, ds.region, t) for t in ds.targets]
    return [c for c in out if c is not None]


# 5. Product performance: each of the top-10 products vs all other rows, per target.
def product_performance(a, b, ds: Dataset, ctx):
    if ds.product is None:
        return []
    counts = a[ds.product].value_counts()
    top = counts[counts >= MIN_GROUP].index[:10]
    if len(top) < 2:
        return []
    claims = []
    for prod in top:
        in_a, in_b = a[ds.product] == prod, b[ds.product] == prod
        for t in ds.targets:
            xa, ya = a.loc[in_a, t], a.loc[~in_a, t]
            if len(ya) < MIN_GROUP or _const(a[t]):
                continue
            pa = stats.mannwhitneyu(xa, ya).pvalue
            if not _ok(pa):
                continue
            xb, yb = b.loc[in_b, t], b.loc[~in_b, t]
            if len(xb) < 5 or len(yb) < 5 or _const(b[t]):
                claims.append((pa, False))
                continue
            pb = stats.mannwhitneyu(xb, yb).pvalue
            claims.append((pa, _ok(pb) and np.sign(xb.mean() - yb.mean()) == np.sign(xa.mean() - ya.mean())))
    return claims


# 6. Clustering: k-means segments. A clean structure (silhouette >= 0.25) is k claims;
#    they replicate if k-means on the held-out half agrees with the discovery centroids
#    (adjusted Rand index >= 0.6).
def clustering(a, b, ds: Dataset, ctx):
    cols = [c for c in ds.numeric if not _const(a[c])]
    if len(cols) < 2 or len(a) < 50 or len(b) < 50:
        return []
    rng = np.random.RandomState(ctx["seed"])
    sc = StandardScaler().fit(a[cols])
    xa, xb = sc.transform(a[cols]), sc.transform(b[cols])
    xa = xa[rng.choice(len(xa), min(len(xa), 1000), replace=False)]
    xb = xb[rng.choice(len(xb), min(len(xb), 1000), replace=False)]
    best_k, best_s, best_m = None, -1.0, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(2, 6):
            m = KMeans(n_clusters=k, n_init=3, random_state=ctx["seed"]).fit(xa)
            if len(set(m.labels_)) < 2:
                continue
            s = silhouette_score(xa, m.labels_)
            if s > best_s:
                best_k, best_s, best_m = k, s, m
        if best_m is None or best_s < 0.25:
            return []
        own = KMeans(n_clusters=best_k, n_init=3, random_state=ctx["seed"]).fit_predict(xb)
    stable = adjusted_rand_score(own, best_m.predict(xb)) >= 0.6
    return [(0.0, bool(stable))] * best_k


# 7. Trend: fit a linear trend on the earlier 70% of the slice (by time). A significant
#    slope is a claim; it replicates if extrapolating it to the later 30% beats a naive
#    forecast (mean of the last 7 periods) on mean absolute error.
def trend(a, b, ds: Dataset, ctx):
    if ds.date is None:
        return []
    full = pd.concat([a, b]).sort_values(ds.date)
    freq = "D" if ctx["span_days"] >= 10 else "h"
    cut = full[ds.date].quantile(0.7)
    claims = []
    for t in ds.targets:
        series = full.set_index(ds.date)[t].resample(freq).mean().dropna()
        early, late = series[series.index <= cut], series[series.index > cut]
        if len(early) < 8 or len(late) < 3 or early.nunique() < 2:
            continue
        x = np.arange(len(series))
        xe, xl = x[: len(early)], x[len(early):]
        fit = stats.linregress(xe, early.values)
        if not _ok(fit.pvalue):
            continue
        trend_mae = np.mean(np.abs(late.values - (fit.intercept + fit.slope * xl)))
        naive_mae = np.mean(np.abs(late.values - early.values[-7:].mean()))
        claims.append((fit.pvalue, bool(trend_mae < naive_mae)))
    return claims


ANALYSES = {
    "descriptive_stats": descriptive_stats, "correlation": correlation,
    "time_series": time_series, "regional_comparison": regional_comparison,
    "product_performance": product_performance, "clustering": clustering, "trend": trend,
}


def run_arm(arm: str, a: pd.DataFrame, b: pd.DataFrame, ds: Dataset, ctx: dict) -> dict:
    """Run one analysis on (discovery, held-out) halves and score it."""
    rep, fail = _keep_top(ANALYSES[arm](a, b, ds, ctx))
    return {"replicated": rep, "failed": fail, "reward": reward_from_claims(rep, fail, arm)}
