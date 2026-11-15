"""Replay every selection method on the saved reward table.

Main benchmark: N episodes x S seeds. Each seed draws its own random sequence of
episodes from the table (with replacement), so every method sees the same sequence
for a given seed. Regret is measured against the per-context oracle: the arm with the
best mean reward for that context (dataset x slice-size bucket) across the whole table.

Transfer test: leave one dataset out. LinUCB learns on the other three, then runs on
the held-out dataset; compared with a cold-start LinUCB and the baselines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .analyses import ARMS
from .data import ROOT
from .episodes import FEATURES, load_table
from .methods import UCB1, EpsilonGreedy, Fixed, LinUCB, Random, Rules, Thompson

REPORTS = ROOT / "reports"
FIGURES = ROOT / "docs" / "figures"


# ---------------------------------------------------------------- table helpers
def size_bucket(rows: pd.Series) -> pd.Series:
    return pd.cut(rows, [0, 500, 2000, np.inf], labels=["small", "medium", "large"]).astype(str)


def prepare(table: pd.DataFrame, scale_from: pd.DataFrame | None = None):
    """Rewards (N x K), discrete keys, standardised features with a bias column."""
    R = table[[f"r_{a}" for a in ARMS]].to_numpy()
    keys = (table["dataset"] + "/" + size_bucket(table["rows"])).to_numpy()
    ref = table if scale_from is None else scale_from
    mu, sd = ref[FEATURES].mean(), ref[FEATURES].std().replace(0, 1)
    X = ((table[FEATURES] - mu) / sd).fillna(0).to_numpy()
    X = np.hstack([X, np.ones((len(X), 1))])
    feats = table[FEATURES].to_dict("records")
    return R, keys, X, feats


def oracle_arms(R: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Per-episode arm chosen by the per-context oracle (best mean arm for its key)."""
    out = np.empty(len(R), dtype=int)
    for k in np.unique(keys):
        m = keys == k
        out[m] = int(np.argmax(R[m].mean(axis=0)))
    return out


def replay(policy, idx, R, keys, X, feats) -> np.ndarray:
    rewards = np.empty(len(idx))
    for t, i in enumerate(idx):
        arm = policy.select(X[i], keys[i], feats[i])
        r = R[i, arm]
        policy.update(X[i], keys[i], arm, r)
        rewards[t] = r
    return rewards


def ci95(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return round(float(v.mean()), 4), round(float(half), 4)


def method_factories(R, d):
    best_fixed = int(np.argmax(R.mean(axis=0)))
    return {
        "Random": lambda rng: Random(rng),
        "Fixed rules (old code)": lambda rng: Rules(rng),
        f"Best single arm ({ARMS[best_fixed]}, hindsight)": lambda rng: Fixed(best_fixed, "best"),
        "epsilon-greedy": lambda rng: EpsilonGreedy(rng),
        "UCB1": lambda rng: UCB1(rng),
        "Thompson sampling": lambda rng: Thompson(rng),
        "LinUCB": lambda rng: LinUCB(rng, d),
    }


# ---------------------------------------------------------------- main benchmark
def benchmark(table, n_episodes=400, seeds=20):
    R, keys, X, feats = prepare(table)
    orc = oracle_arms(R, keys)
    orc_r = R[np.arange(len(R)), orc]
    ceiling = R.max(axis=1)
    out, curves = {}, {}  # values rounded to 4 dp so results match across platforms
    seq = {s: np.random.default_rng(1000 + s).integers(len(R), size=n_episodes) for s in range(seeds)}
    for name, make in method_factories(R, X.shape[1]).items():
        pct, pct_last, regret, mean_r, ceil_pct = [], [], [], [], []
        curve = np.zeros((seeds, n_episodes))
        for s in range(seeds):
            idx = seq[s]
            r = replay(make(np.random.default_rng(s)), idx, R, keys, X, feats)
            curve[s] = np.cumsum(orc_r[idx] - r)
            pct.append(100 * r.sum() / orc_r[idx].sum())
            pct_last.append(100 * r[-100:].sum() / orc_r[idx][-100:].sum())
            ceil_pct.append(100 * r.sum() / ceiling[idx].sum())
            regret.append(curve[s, -1])
            mean_r.append(r.mean())
        curves[name] = curve
        out[name] = {"pct_of_oracle": ci95(pct), "pct_of_oracle_last100": ci95(pct_last),
                     "pct_of_hindsight_best": ci95(ceil_pct), "final_regret": ci95(regret),
                     "mean_reward": ci95(mean_r)}
    out["Oracle (best arm per context)"] = {
        "pct_of_oracle": (100.0, 0.0), "pct_of_oracle_last100": (100.0, 0.0),
        "pct_of_hindsight_best": ci95([100 * orc_r[seq[s]].sum() / ceiling[seq[s]].sum() for s in range(seeds)]),
        "final_regret": (0.0, 0.0), "mean_reward": ci95([orc_r[seq[s]].mean() for s in range(seeds)])}
    return out, curves


# ---------------------------------------------------------------- paired significance tests
def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired(a, b) -> dict:
    """Paired t-test and Wilcoxon signed-rank test of a - b (two-sided)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return {"mean_diff": 0.0, "t_p": 1.0, "wilcoxon_p": 1.0}
    return {"mean_diff": float(d.mean()), "t_p": float(stats.ttest_rel(a, b).pvalue),
            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue)}


def paired_tests(curves, reference="Fixed rules (old code)") -> list[dict]:
    """Each learning method vs the fixed rule, paired by seed (every method sees the same
    episodes for a given seed). Compared on reward over the whole run and over the last
    100 episodes, as differences in cumulative regret. Holm-corrected across all tests."""
    ref = curves[reference]
    rows = []
    for name in ["epsilon-greedy", "UCB1", "Thompson sampling", "LinUCB"]:
        c = curves[name]
        for window, f in [("all 400", lambda x: x[:, -1]), ("last 100", lambda x: x[:, -1] - x[:, -101])]:
            # Lower regret = more reward, so (ref regret - method regret) = method reward - ref reward.
            res = paired(f(ref), f(c))
            rows.append({"method": name, "vs": reference, "window": window,
                         "reward_diff": round(res["mean_diff"], 3), "t_p": res["t_p"], "wilcoxon_p": res["wilcoxon_p"]})
    for key in ["t_p", "wilcoxon_p"]:
        for row, adj in zip(rows, holm([r[key] for r in rows])):
            row[f"{key}_holm"] = adj
    for row in rows:
        for k in ["t_p", "wilcoxon_p", "t_p_holm", "wilcoxon_p_holm"]:
            row[k] = float(f"{row[k]:.3g}")
    return rows


# ---------------------------------------------------------------- transfer test
def transfer(table, n_train=300, n_test=100, seeds=20):
    """Returns (% of oracle per method per held-out dataset, paired tests of pre-trained vs
    cold-start LinUCB per dataset, Holm-corrected across the 4 datasets)."""
    results, tests = {}, {}
    for held in sorted(table["dataset"].unique()):
        tr, te = table[table.dataset != held].reset_index(drop=True), table[table.dataset == held].reset_index(drop=True)
        Rtr, ktr, Xtr, ftr = prepare(tr, scale_from=tr)
        Rte, kte, Xte, fte = prepare(te, scale_from=tr)
        orc_r = Rte[np.arange(len(Rte)), oracle_arms(Rte, kte)]
        fixed_from_train = int(np.argmax(Rtr.mean(axis=0)))
        rows = {k: [] for k in ["LinUCB, pre-trained on other 3", "LinUCB, cold start", "Random",
                                "Fixed rules (old code)", f"Best arm on other 3 ({ARMS[fixed_from_train]})"]}
        for s in range(seeds):
            rng = np.random.default_rng(5000 + s)
            itr, ite = rng.integers(len(Rtr), size=n_train), rng.integers(len(Rte), size=n_test)
            denom = orc_r[ite].sum()
            warm = LinUCB(np.random.default_rng(s), Xtr.shape[1])
            replay(warm, itr, Rtr, ktr, Xtr, ftr)
            policies = {"LinUCB, pre-trained on other 3": warm,
                        "LinUCB, cold start": LinUCB(np.random.default_rng(s), Xte.shape[1]),
                        "Random": Random(np.random.default_rng(s)),
                        "Fixed rules (old code)": Rules(None),
                        f"Best arm on other 3 ({ARMS[fixed_from_train]})": Fixed(fixed_from_train, "")}
            for k, pol in policies.items():
                rows[k].append(100 * replay(pol, ite, Rte, kte, Xte, fte).sum() / denom)
        results[held] = {k: ci95(v) for k, v in rows.items()}
        res = paired(rows["LinUCB, pre-trained on other 3"], rows["LinUCB, cold start"])
        tests[held] = {"pct_diff": round(res["mean_diff"], 2), "t_p": res["t_p"], "wilcoxon_p": res["wilcoxon_p"]}
    for key in ["t_p", "wilcoxon_p"]:
        for held, adj in zip(tests, holm([tests[h][key] for h in tests])):
            tests[held][f"{key}_holm"] = adj
    tests = {h: {k: (float(f"{v:.3g}") if k != "pct_diff" else v) for k, v in t.items()} for h, t in tests.items()}
    return results, tests


# ---------------------------------------------------------------- LLM comparison
def llm_eval(table, seeds=20, allow_live=True):
    """Score the LLM chooser on its fixed 100 episodes, next to the other methods on the
    same episodes. LinUCB first learns on 400 other episodes (the LLM gets no feedback)."""
    from . import llm

    got = llm.choose_all(table, allow_live=allow_live)
    if got is None:
        return None
    idx, used = got
    R, keys, X, feats = prepare(table)
    orc_r = R[np.arange(len(R)), oracle_arms(R, keys)]
    idx = np.array(idx)
    denom = orc_r[idx].sum()
    entries = [used[llm._key(llm.build_prompt(table.loc[i].to_dict()))] for i in idx]
    arms = [llm.parse_arm(e["text"]) for e in entries]
    invalid = sum(a is None for a in arms)
    worst = -max(llm.COST.values())
    llm_r = np.array([R[i, a] if a is not None else worst for i, a in zip(idx, arms)])
    best_fixed = int(np.argmax(R.mean(axis=0)))
    rest = np.setdiff1d(np.arange(len(R)), idx)
    lin = []
    for s in range(seeds):
        pol = LinUCB(np.random.default_rng(s), X.shape[1])
        replay(pol, np.random.default_rng(9000 + s).choice(rest, size=400), R, keys, X, feats)
        lin.append(replay(pol, idx, R, keys, X, feats).sum())
    pct = {
        f"LLM ({llm.MODEL}, zero-shot)": 100 * llm_r.sum() / denom,
        "LinUCB after 400 learning episodes (mean of 20 seeds)": 100 * np.mean(lin) / denom,
        f"Best single arm ({ARMS[best_fixed]}, hindsight)": 100 * R[idx, best_fixed].sum() / denom,
        "Fixed rules (old code)": 100 * sum(R[i, Rules(None).select(None, None, feats[i])] for i in idx) / denom,
        "Random (expected)": 100 * R[idx].mean(axis=1).sum() / denom,
    }
    pct = {k: round(float(v), 2) for k, v in pct.items()}
    # Per-episode paired comparison (n = 100 episodes).
    rule_r = np.array([R[i, Rules(None).select(None, None, feats[i])] for i in idx])
    llm_tests = {}
    for label, other in [("LLM vs fixed rules", rule_r), ("LLM vs random (expected)", R[idx].mean(axis=1))]:
        res = paired(llm_r, other)
        llm_tests[label] = {"reward_diff": round(res["mean_diff"], 3), "t_p": float(f"{res['t_p']:.3g}"),
                            "wilcoxon_p": float(f"{res['wilcoxon_p']:.3g}")}
    # Identical prompts share one cached answer, so latency and tokens are per real API call.
    calls = list(used.values())
    lat = np.array([e["latency_s"] for e in calls])
    tok = np.array([e.get("total_tokens") or np.nan for e in calls], dtype=float)
    picks = pd.Series([ARMS[a] if a is not None else "invalid" for a in arms]).value_counts().to_dict()
    return {"model": llm.MODEL, "episodes": len(idx), "api_calls": len(calls), "pct_of_oracle": pct,
            "invalid_answers": invalid, "picks": picks, "latency_s_median": round(float(np.median(lat)), 3),
            "latency_s_p90": round(float(np.quantile(lat, 0.9)), 3),
            "tokens_per_decision": float(np.nanmean(tok)), "paired_tests": llm_tests,
            "cost_note": "$0 (Gemini API free tier, no billing account)."}


# ---------------------------------------------------------------- outputs
def plot_regret(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged: dict[str, np.ndarray] = {}
    for name, c in curves.items():   # identical curves share one line and one label
        same = next((k for k, v in merged.items() if np.array_equal(v, c)), None)
        if same:
            merged[f"{same} = {name}"] = merged.pop(same)
        else:
            merged[name] = c
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, c in merged.items():
        m = c.mean(axis=0)
        half = stats.t.ppf(0.975, len(c) - 1) * c.std(axis=0, ddof=1) / np.sqrt(len(c))
        x = np.arange(1, c.shape[1] + 1)
        ax.plot(x, m, label=name, lw=1.8)
        ax.fill_between(x, m - half, m + half, alpha=0.18)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative regret vs per-context oracle")
    ax.set_title(f"Cumulative regret, mean and 95% CI over {next(iter(curves.values())).shape[0]} seeds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def describe_table(table) -> dict:
    R, keys, _, _ = prepare(table)
    orc = oracle_arms(R, keys)
    oracle_by_key = {k: ARMS[orc[keys == k][0]] for k in sorted(set(keys))}
    arm_means = {a: round(float(R[:, i].mean()), 4) for i, a in enumerate(ARMS)}
    best_share = pd.Series(np.array(ARMS)[R.argmax(axis=1)]).value_counts(normalize=True).round(3).to_dict()
    return {"episodes": len(table), "per_dataset": table.dataset.value_counts().to_dict(),
            "arm_mean_reward": arm_means, "share_of_episodes_where_arm_is_best": best_share,
            "oracle_arm_by_context": oracle_by_key}


def _fmt(ci, digits=1):
    return f"{ci[0]:.{digits}f} ± {ci[1]:.{digits}f}"


def write_markdown(summary, path):
    b, t, cfg = summary["benchmark"], summary["transfer"], summary["config"]
    lines = [f"# Results\n",
             f"Generated by `python run_all.py`. {cfg['n_episodes']} episodes x {cfg['seeds']} seeds, "
             f"drawn from a table of {summary['table']['episodes']} real-data episodes. "
             "Values are mean ± 95% CI over seeds.\n",
             "## Benchmark\n",
             "| Method | % of oracle reward | % of oracle, last 100 episodes | Final cumulative regret |",
             "|---|---:|---:|---:|"]
    for name, r in b.items():
        lines.append(f"| {name} | {_fmt(r['pct_of_oracle'])} | {_fmt(r['pct_of_oracle_last100'])} | {_fmt(r['final_regret'], 2)} |")
    lines += ["\n## Leave-one-dataset-out transfer (% of the held-out dataset's oracle)\n"]
    methods = list(next(iter(t.values())).keys())
    lines += ["| Method | " + " | ".join(t.keys()) + " |", "|---|" + "---:|" * len(t)]
    for m in methods:
        label = m if not m.startswith("Best arm on other 3") else "Best arm on other 3"
        lines.append(f"| {label} | " + " | ".join(_fmt(t[d][m]) for d in t) + " |")
    tt = summary["transfer_tests"]
    lines += ["\nPre-trained vs cold-start LinUCB, paired by seed (Holm-corrected over 4 datasets):\n",
              "| Held-out dataset | Difference (points) | Paired t-test p (Holm) | Wilcoxon p (Holm) |",
              "|---|---:|---:|---:|"]
    for d, r in tt.items():
        lines.append(f"| {d} | {r['pct_diff']:+.1f} | {r['t_p_holm']:.3g} | {r['wilcoxon_p_holm']:.3g} |")
    lines += ["\n## Paired tests vs the fixed rule\n",
              f"Paired by seed (n = {cfg['seeds']}; every method sees the same episodes per seed). "
              "Reward difference = method minus fixed rule, summed over the window. "
              "p-values are Holm-corrected across all 8 comparisons.\n",
              "| Method | Window | Reward difference | Paired t-test p (Holm) | Wilcoxon p (Holm) |",
              "|---|---|---:|---:|---:|"]
    for r in summary["paired_tests"]:
        lines.append(f"| {r['method']} | {r['window']} | {r['reward_diff']:+.2f} | {r['t_p_holm']:.3g} | {r['wilcoxon_p_holm']:.3g} |")
    if summary.get("llm"):
        L = summary["llm"]
        lines += [f"\n## LLM chooser ({L['model']}, {L['episodes']} episodes)\n",
                  "| Method | % of oracle reward (same episodes) |", "|---|---:|"]
        for k, v in L["pct_of_oracle"].items():
            lines.append(f"| {k} | {v:.1f} |")
        picks = ", ".join(f"{k} {v}" for k, v in L["picks"].items())
        lines.append(f"\nLLM picks: {picks}. Invalid answers: {L['invalid_answers']}.\n")
        lines.append(f"{L['episodes']} decisions needed {L['api_calls']} API calls (identical prompts reuse the cached "
                     f"answer). Latency per call: median {L['latency_s_median']:.2f} s, p90 {L['latency_s_p90']:.2f} s. "
                     f"Tokens per call: {L['tokens_per_decision']:.0f}. Cost: {L['cost_note']}\n")
        lines += ["Per-episode paired tests (n = 100 episodes):\n",
                  "| Comparison | Mean reward difference | Paired t-test p | Wilcoxon p |", "|---|---:|---:|---:|"]
        for k, r in L["paired_tests"].items():
            lines.append(f"| {k} | {r['reward_diff']:+.3f} | {r['t_p']:.3g} | {r['wilcoxon_p']:.3g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run(n_episodes=400, seeds=20, allow_live_llm=True):
    table = load_table()
    bench, curves = benchmark(table, n_episodes, seeds)
    trans, trans_tests = transfer(table, seeds=seeds)
    summary = {"config": {"n_episodes": n_episodes, "seeds": seeds}, "table": describe_table(table),
               "benchmark": bench, "paired_tests": paired_tests(curves),
               "transfer": trans, "transfer_tests": trans_tests}
    llm_summary = llm_eval(table, seeds, allow_live=allow_live_llm)
    if llm_summary:
        summary["llm"] = llm_summary
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    write_markdown(summary, REPORTS / "results.md")
    plot_regret(curves, FIGURES / "regret.png")
    return summary


if __name__ == "__main__":
    s = run()
    print((REPORTS / "results.md").read_text(encoding="utf-8"))
