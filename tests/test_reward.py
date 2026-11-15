import numpy as np
import pandas as pd
import pytest

from src.selector import analyses
from src.selector.analyses import ARMS, COST, _keep_top, reward_from_claims, run_arm
from src.selector.episodes import build_episode, global_rho


def _ctx(ds, seed=1):
    return {"global_means": ds.df[ds.targets].mean().to_dict(), "global_rho": global_rho(ds),
            "span_days": 60.0, "seed": seed}


def _halves(ds, seed=0):
    m = np.random.default_rng(seed).permutation(len(ds.df)) < len(ds.df) // 2
    return ds.df[m], ds.df[~m]


def test_reward_formula():
    assert reward_from_claims(5, 0, "correlation") == pytest.approx(1.0 - COST["correlation"])
    assert reward_from_claims(2, 2, "trend") == pytest.approx((2 - 1) / 5 - COST["trend"])
    assert reward_from_claims(0, 0, "clustering") == pytest.approx(-COST["clustering"])


def test_keep_top_caps_and_orders():
    claims = [(0.04, False)] * 3 + [(0.001, True)] * 6
    assert _keep_top(claims) == (5, 0)          # the 5 smallest p-values are kept
    assert _keep_top([(0.01, True), (0.02, False)]) == (1, 1)


def test_planted_region_effect_replicates(toy_dataset):
    a, b = _halves(toy_dataset)
    res = run_arm("regional_comparison", a, b, toy_dataset, _ctx(toy_dataset))
    assert res["replicated"] >= 1               # north vs others on sales is real


def test_known_correlation_is_not_a_finding(toy_dataset):
    a, b = _halves(toy_dataset)
    claims = analyses.correlation(a, b, toy_dataset, _ctx(toy_dataset))
    # x~y is strong in the full data, so it is skipped; nothing else is correlated.
    assert claims == []


def test_inapplicable_arm_scores_minus_cost(toy_dataset):
    no_date = toy_dataset.__class__(**{**toy_dataset.__dict__, "date": None})
    a, b = _halves(no_date)
    for arm in ["trend", "time_series"]:
        assert run_arm(arm, a, b, no_date, _ctx(no_date))["reward"] == pytest.approx(-COST[arm])


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_is_deterministic(toy_dataset, arm):
    a, b = _halves(toy_dataset)
    r1 = run_arm(arm, a, b, toy_dataset, _ctx(toy_dataset))
    r2 = run_arm(arm, a, b, toy_dataset, _ctx(toy_dataset))
    assert r1 == r2
    assert -0.5 - max(COST.values()) <= r1["reward"] <= 1.0


def test_episode_rows_are_reproducible(toy_dataset):
    keep = lambda row: {k: v for k, v in row.items() if not k.startswith("sec_")}  # noqa: E731
    first = keep(build_episode(toy_dataset, 3, seed=0))
    assert first == keep(build_episode(toy_dataset, 3, seed=0))
    assert first != keep(build_episode(toy_dataset, 4, seed=0))
