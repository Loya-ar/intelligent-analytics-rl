"""Analysis-selection methods. Each picks one of the 7 arms per episode.

All methods share one interface:  select(x, key) -> arm index;  update(x, key, arm, reward).
`x` is the context feature vector (for LinUCB), `key` a discrete context label
(dataset x slice-size bucket) for the tabular bandits.
"""
from __future__ import annotations

import numpy as np

from .analyses import ARMS

K = len(ARMS)
A = {a: i for i, a in enumerate(ARMS)}


class Random:
    name = "Random"

    def __init__(self, rng):
        self.rng = rng

    def select(self, x, key, feats=None):
        return int(self.rng.integers(K))

    def update(self, x, key, arm, r):
        pass


class Rules:
    """The old coordinator's fixed column rules, in its order: the statistical agent
    (correlation) acts if there are >= 2 numeric columns, then geographic, then temporal."""
    name = "Fixed rules (old code)"

    def __init__(self, rng):
        pass

    def select(self, x, key, feats=None):
        if feats["n_numeric"] >= 2:
            return A["correlation"]
        if feats["has_region"]:
            return A["regional_comparison"]
        if feats["has_date"]:
            return A["time_series"]
        return A["descriptive_stats"]

    def update(self, x, key, arm, r):
        pass


class Fixed:
    """Always the same arm (used for the best single arm in hindsight)."""

    def __init__(self, arm, name):
        self.arm, self.name = arm, name

    def select(self, x, key, feats=None):
        return self.arm

    def update(self, x, key, arm, r):
        pass


class _Tabular:
    def __init__(self, rng):
        self.rng = rng
        self.n: dict = {}
        self.s: dict = {}

    def _stats(self, key):
        if key not in self.n:
            self.n[key], self.s[key] = np.zeros(K), np.zeros(K)
        return self.n[key], self.s[key]

    def update(self, x, key, arm, r):
        n, s = self._stats(key)
        n[arm] += 1
        s[arm] += r


class EpsilonGreedy(_Tabular):
    name = "epsilon-greedy"

    def __init__(self, rng, eps=0.1):
        super().__init__(rng)
        self.eps = eps

    def select(self, x, key, feats=None):
        n, s = self._stats(key)
        if (n == 0).any():
            return int(self.rng.choice(np.flatnonzero(n == 0)))
        if self.rng.random() < self.eps:
            return int(self.rng.integers(K))
        return int(np.argmax(s / n))


class UCB1(_Tabular):
    name = "UCB1"

    def __init__(self, rng, c=1.0):
        super().__init__(rng)
        self.c = c

    def select(self, x, key, feats=None):
        n, s = self._stats(key)
        if (n == 0).any():
            return int(self.rng.choice(np.flatnonzero(n == 0)))
        return int(np.argmax(s / n + self.c * np.sqrt(2 * np.log(n.sum()) / n)))


class Thompson(_Tabular):
    """Gaussian Thompson sampling: prior N(0, 1) on each arm's mean, noise sd 0.25."""
    name = "Thompson sampling"

    def __init__(self, rng, noise_sd=0.25):
        super().__init__(rng)
        self.tau = 1 / noise_sd**2

    def select(self, x, key, feats=None):
        n, s = self._stats(key)
        prec = 1 + self.tau * n
        mean = self.tau * s / prec
        return int(np.argmax(self.rng.normal(mean, 1 / np.sqrt(prec))))


class LinUCB:
    """Disjoint LinUCB (Li et al., 2010) on the generic context features."""
    name = "LinUCB"

    def __init__(self, rng, d, alpha=1.0):
        self.rng, self.alpha = rng, alpha
        self.Ainv = np.stack([np.eye(d)] * K)
        self.b = np.zeros((K, d))

    def select(self, x, key, feats=None):
        theta = np.einsum("kij,kj->ki", self.Ainv, self.b)
        width = np.sqrt(np.einsum("i,kij,j->k", x, self.Ainv, x))
        p = theta @ x + self.alpha * width
        return int(self.rng.choice(np.flatnonzero(p == p.max())))

    def update(self, x, key, arm, r):
        Ai = self.Ainv[arm]
        Ax = Ai @ x
        self.Ainv[arm] = Ai - np.outer(Ax, Ax) / (1 + x @ Ax)  # Sherman-Morrison
        self.b[arm] += r * x
