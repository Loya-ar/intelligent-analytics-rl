"""Reproduce every result in the README with one command.

    python run_all.py              # download + verify data, build the reward table if missing, evaluate
    python run_all.py --rebuild    # also rebuild reports/reward_table.csv from the raw data (~5 min)
    python run_all.py --replay     # no downloads: replay from the committed reward table and LLM cache

Outputs: reports/results.md, reports/results.json, docs/figures/regret.png.
LLM decisions come from reports/llm_cache.json; missing ones are fetched only when
GEMINI_API_KEY is set (free tier).
"""
from __future__ import annotations

import argparse

from src.selector import data, episodes, evaluate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebuild", action="store_true", help="rebuild the reward table from raw data")
    p.add_argument("--replay", action="store_true", help="skip data download; use committed table + LLM cache")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seeds", type=int, default=20)
    args = p.parse_args()

    if not args.replay:
        data.download_all()
        if args.rebuild or not episodes.TABLE.exists():
            episodes.build_table()
    evaluate.run(args.episodes, args.seeds, allow_live_llm=not args.replay)
    print((evaluate.REPORTS / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
