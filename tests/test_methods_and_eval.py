import json

import numpy as np
import pandas as pd
import pytest

from src.selector import llm
from src.selector.analyses import ARMS
from src.selector.episodes import FEATURES
from src.selector.evaluate import benchmark, ci95, oracle_arms, prepare, replay
from src.selector.methods import UCB1, EpsilonGreedy, Fixed, LinUCB, Thompson

K = len(ARMS)


def toy_table(n=300, seed=0):
    """Two contexts; arm 1 is best in context A, arm 4 in context B, plus noise."""
    rng = np.random.default_rng(seed)
    ds = np.where(np.arange(n) % 2 == 0, "A", "B")
    R = rng.normal(0, 0.05, (n, K))
    R[ds == "A", 1] += 0.6
    R[ds == "B", 4] += 0.6
    t = pd.DataFrame({"episode": range(n), "dataset": ds, "rows": 1000})
    for f in FEATURES:
        t[f] = 0.0
    t["has_date"] = (ds == "B").astype(int)
    for i, a in enumerate(ARMS):
        t[f"r_{a}"] = R[:, i]
    return t


def test_oracle_picks_best_mean_arm_per_context():
    t = toy_table()
    R, keys, _, _ = prepare(t)
    orc = oracle_arms(R, keys)
    assert set(orc[t.dataset == "A"]) == {1} and set(orc[t.dataset == "B"]) == {4}


def test_oracle_policy_has_zero_regret_and_regret_is_cumulative():
    t = toy_table()
    R, keys, X, feats = prepare(t)
    orc = oracle_arms(R, keys)
    idx = np.arange(50)

    class OraclePolicy:
        def select(self, x, key, f=None):
            return 1 if key.startswith("A") else 4

        def update(self, *a):
            pass

    r = replay(OraclePolicy(), idx, R, keys, X, feats)
    regret = np.cumsum(R[idx, orc[idx]] - r)
    assert np.allclose(regret, 0)
    r0 = replay(Fixed(0, "arm0"), idx, R, keys, X, feats)
    regret0 = np.cumsum(R[idx, orc[idx]] - r0)
    assert np.all(np.diff(regret0) > 0) and regret0[-1] == pytest.approx(np.sum(R[idx, orc[idx]] - R[idx, 0]))


@pytest.mark.parametrize("cls", [EpsilonGreedy, UCB1, Thompson])
def test_tabular_bandits_learn_the_best_arm(cls):
    t = toy_table(800)
    R, keys, X, feats = prepare(t)
    r = replay(cls(np.random.default_rng(0)), np.arange(800), R, keys, X, feats)
    assert r[-200:].mean() > 0.45            # near the 0.6 optimum, far above random (~0.17)


def test_linucb_learns_from_features_and_inverse_is_exact():
    t = toy_table(600)
    R, keys, X, feats = prepare(t)
    pol = LinUCB(np.random.default_rng(0), X.shape[1])
    r = replay(pol, np.arange(600), R, keys, X, feats)
    assert r[-200:].mean() > 0.45
    # Sherman-Morrison updates match a direct inverse.
    arm = 1
    A = np.eye(X.shape[1])
    pol2 = LinUCB(np.random.default_rng(0), X.shape[1])
    for i in range(20):
        pol2.update(X[i], None, arm, 0.3)
        A += np.outer(X[i], X[i])
    assert np.allclose(pol2.Ainv[arm], np.linalg.inv(A))


def test_benchmark_is_reproducible_with_fixed_seeds():
    t = toy_table(200)
    a, _ = benchmark(t, n_episodes=60, seeds=3)
    b, _ = benchmark(t, n_episodes=60, seeds=3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_ci95():
    m, h = ci95([1.0, 2.0, 3.0])
    assert m == 2.0 and h == pytest.approx(4.302653 / np.sqrt(3), abs=1e-4)


def test_llm_parsing_and_cache_key():
    assert llm.parse_arm('{"analysis": "clustering"}') == ARMS.index("clustering")
    assert llm.parse_arm('{"analysis": "Trend"}') == ARMS.index("trend")
    assert llm.parse_arm("no idea") is None
    row = {"dataset": "taxi", "slice": "segment: Borough = Queens", "rows": 400, "target_zero_frac": 0.1}
    p = llm.build_prompt(row)
    assert all(a in p for a in ARMS)
    assert llm._key(p) == llm._key(llm.build_prompt(dict(row)))


def test_holm_matches_hand_calculation():
    from src.selector.evaluate import holm
    # sorted p: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 (monotone)
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_paired_tests_sign_follows_reward():
    from src.selector.evaluate import paired_tests
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.uniform(0, 1, (10, 200)), axis=1)
    curves = {"Fixed rules (old code)": base}
    for name in ["epsilon-greedy", "UCB1", "Thompson sampling", "LinUCB"]:
        curves[name] = base + np.linspace(0, 20, 200) + rng.normal(0, 1, (10, 200))  # more regret
    rows = paired_tests(curves)
    assert all(r["reward_diff"] < 0 for r in rows)
    assert all(r["t_p_holm"] >= r["t_p"] for r in rows)
