import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.selector.data import Dataset  # noqa: E402


@pytest.fixture
def toy_dataset() -> Dataset:
    """Synthetic data with a date, a region, a product and a planted region effect."""
    rng = np.random.default_rng(0)
    n = 4000
    region = rng.choice(["north", "south", "east"], n)
    x = rng.normal(size=n)
    df = pd.DataFrame({
        "when": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.uniform(0, 60, n), unit="D"),
        "region": region,
        "product": rng.choice(list("abcd"), n),
        "seg": rng.choice(["s1", "s2"], n),
        "x": x,
        "y": 2 * x + rng.normal(size=n),          # strongly tied to x everywhere: a "known" pair
        "sales": 10 + 3 * (region == "north") + rng.normal(size=n),
        "z": rng.normal(size=n),
    })
    return Dataset("toy", df, targets=["sales", "z"], numeric=["x", "y", "sales", "z"],
                   date="when", region="region", product="product", segments=["seg"])
