"""Rebuild a sample of episodes from the raw data and compare them with the committed table.

Proves reports/reward_table.csv comes from the code and the checksummed public data.
Usage:  python scripts/check_table_rows.py [--per-dataset 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.selector.data import download_all, load_all  # noqa: E402
from src.selector.episodes import build_episode, load_table  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-dataset", type=int, default=5)
    args = p.parse_args()
    download_all()
    datasets, table = load_all(), load_table()
    bad = 0
    for name, group in table.groupby("dataset"):
        for _, row in group.head(args.per_dataset).iterrows():
            new = build_episode(datasets[name], int(row["episode"]), seed=0)
            for col, old in row.items():
                val = new[col]
                same = np.isclose(val, old, atol=1e-6) if isinstance(old, (float, np.floating)) else val == old
                if not same:
                    bad += 1
                    print(f"MISMATCH episode {row['episode']} {col}: committed {old!r}, rebuilt {val!r}")
        print(f"{name}: checked {min(len(group), args.per_dataset)} episodes")
    print("OK" if bad == 0 else f"{bad} mismatches")
    return int(bad > 0)


if __name__ == "__main__":
    raise SystemExit(main())
