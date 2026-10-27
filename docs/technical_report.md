# Intelligent Analytics Assistant: Technical Report

**Author**: Arjun Loya
**Course**: Reinforcement Learning for Agentic AI Systems, Northeastern University
**Date**: August 2025

---

## 1. Summary

This project explores reinforcement learning for choosing which analysis to run on a business dataset. It has two parts:

1. **A contextual-bandit agent** (`src/agents/rl_analytics_agent.py`) that uses UCB to choose among seven analyses based on the dataset's context, and learns a Q-value for each context–analysis pair from the reward.
2. **A multi-agent system** (`src/agents/multi_agent_coordinator.py`) with three specialised agents (Statistical, Geographic, Temporal) that run their analyses, share insights with each other, and learn a Q-value per data context.

The main demonstration (`run_demo.py`) runs **6 coordination episodes** on two synthetic datasets. It is a small proof of concept, not a benchmark. Section 6 lists what was not measured.

## 2. Data

Three synthetic datasets generated with Faker and a fixed seed (`src/utils/data_generator.py`):

| Dataset | Rows | Notable columns |
|---|---:|---|
| Sales | 10,000 | date, region, product, sales rep, customer segment, quantity, unit price, total sales, month, quarter, year |
| Marketing | 5,000 | campaign, channel, impressions, clicks, conversions, cost, revenue, CTR, conversion rate, ROAS |
| HR | 2,000 | department, job level, salary, performance score, years of experience, training hours, satisfaction |

## 3. Method 1: Contextual bandit with UCB

**Context.** Each dataset is reduced to a discrete context key from five features: size bucket (small < 1,000 rows ≤ medium < 10,000 ≤ large), whether numeric columns outnumber categorical ones, domain (from column-name keywords: sales, marketing, hr, general), and whether a date column and a region column exist.

**Actions.** Seven analyses: descriptive stats, correlation analysis, time-series analysis, regional comparison, product performance, clustering and trend analysis.

**Selection.** Every untried action in a context is explored first. After that, the agent picks the action with the highest upper confidence bound:

```
a* = argmax_a [ Q(s,a) + sqrt( 2 · ln(t + 1) / N(s,a) ) ]
```

where `t` is the number of trials in context `s` and `N(s,a)` is how often action `a` has been tried there. An ε-greedy selector (ε = 0.3) is also implemented as an alternative.

**Reward.** 1.0 for a completed analysis, −1.0 for a failed one, plus bonuses: +0.5 per correlation with |r| > 0.5, +0.3 for a regional comparison that finds a best region, and +0.2 for descriptive stats with a numeric summary.

**Update.** `Q(s,a) ← Q(s,a) + α (r − Q(s,a))` with α = 0.1. This is an incremental average of the reward, with no bootstrapping from future states.

## 4. Method 2: Multi-agent coordination

**Agents.**

| Agent | Takes part when the data has… | Analysis |
|---|---|---|
| Statistical | at least two numeric columns | Pairwise correlations, keeping pairs with \|r\| > 0.4 |
| Geographic | a column containing "region" or "location" | Mean of the main sales/revenue measure by region; best and worst region |
| Temporal | a date, time or month column (plus month and year columns to analyse) | Mean of the main sales/revenue measure by month; best and worst month |

**Participation is rule-based.** Each agent's `should_act` checks the column rules above. It does not use the learned Q-values, so the learning in this part records how well each agent does per context but does not yet change which agents take part.

**Episode.** The coordinator asks each agent whether it takes part, runs the participants in order, and after each analysis the agent sends its insight to the others.

**Reward.** 1.0 for a completed analysis, plus +0.3 per correlation found (Statistical), +0.5 for finding a best region (Geographic) or a seasonal pattern (Temporal), and +0.2 if the agent has received messages from other agents (collaboration bonus). A failed analysis gets −0.5.

**Update.** The same incremental rule as Method 1, with one Q-value per agent and context.

## 5. Results

Output of `python run_demo.py`: 3 episodes on the sales data, then 3 on the marketing data.

| Agent | Episodes taken part | Total reward | Final Q-value (sales context) | Final Q-value (marketing context) |
|---|---:|---:|---:|---:|
| Statistical | 6 | 19.60 | 0.55 | 1.22 |
| Geographic | 3 | 5.10 | 0.46 | – |
| Temporal | 3 | 5.10 | 0.46 | – |
| **System** | | **29.80** | | |

- On the sales data all three agents took part. On the marketing data only the Statistical agent did, because the marketing data has no region, date or month columns.
- In every episode at least one agent returned an analysis result (6 of 6).
- Rewards per context are almost constant (for example, the Statistical agent earned 4.5 in each marketing episode), so the Q-values rise smoothly toward that reward. The rising curves show the update rule converging, not the agents discovering better choices.

The figures in `docs/figures/` are drawn from this run by `scripts/make_figures.py`.

## 6. What was not measured

- **No comparison with a baseline.** The runs are not compared with random or fixed choices, so there is no evidence yet that learning improves outcomes.
- **No statistical significance.** `src/evaluation/statistical_validator.py` implements paired t-tests, Wilcoxon tests and effect sizes, but its demo runs on example reward histories, not on these agents' runs. With 3–6 nearly deterministic episodes per agent, a significance test would not be meaningful anyway.
- **Business impact was not measured.** There are no measurements of time saved, cost or return on investment.
- **One seed, synthetic data.** All results come from a single run on synthetic data.

## 7. Next steps

1. Let the learning drive decisions: the coordinator chooses which agents to call, and each agent chooses its analysis, using the UCB bandit from Method 1.
2. Add noise to the rewards (for example, by evaluating each analysis on random subsamples) so there is something to learn.
3. Run several hundred episodes over multiple seeds, compare the learned policy with random and rule-based baselines, and run the significance tests on those real runs.

## Appendix: Code map

| File | Role |
|---|---|
| `run_demo.py` | Runs the 6-episode demonstration and prints per-agent metrics |
| `src/agents/multi_agent_coordinator.py` | The three agents and the coordinator (Method 2) |
| `src/agents/rl_analytics_agent.py` | Contextual-bandit agent (Method 1) |
| `src/agents/analytics_agent.py` | Base analysis functions used by the bandit agent |
| `src/tools/custom_analytics_tools.py` | Correlation, clustering, time-series and KPI tools |
| `src/evaluation/statistical_validator.py` | Statistical tests on reward histories |
| `src/utils/memory_manager.py` | SQLite persistence for learning history |
| `scripts/make_figures.py` | Retrains the agents and draws the README figures |
