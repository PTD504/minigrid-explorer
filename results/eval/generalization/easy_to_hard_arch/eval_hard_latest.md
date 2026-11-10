# MiniGrid Explorer Evaluation Report: HARD Suite

- **Generated:** `2026-11-09 20:13:20`
- **Difficulty Tier:** `hard`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Random** | 1 | **13.47% ± 4.44%** | 2.56% | 2.0% | **0.0%** | 100.0% | 51.8 | -1.760 ± 0.480 | 0.48 ms |
| **Baseline: Frontier BFS** | 1 | **44.29% ± 26.22%** | 11.59% | 31.8% | **6.0%** | 94.0% | 69.9 | +2.337 ± 3.502 | 0.54 ms |
| **Baseline: Wall Follower** | 1 | **19.74% ± 10.57%** | 9.05% | 9.0% | **0.0%** | 99.0% | 115.7 | -0.919 ± 1.453 | 0.45 ms |
| **Baseline: Oracle Planner (Privileged Baseline)** | 1 | **62.02% ± 29.87%** | 15.71% | 50.4% | **29.0%** | 71.0% | 90.6 | +5.114 ± 4.562 | 0.56 ms |
| **ppo_easy_5seeds** | 5 | **60.38% ± 2.89%** | 20.13% | 42.2% | **13.8%** | 68.4% | 327.0 | +2.065 ± 0.771 | 1.06 ms |
| **ppo_nomem_easy_5seeds** | 5 | **65.21% ± 5.49%** | 25.97% | 46.3% | **18.0%** | 62.8% | 368.1 | +3.775 ± 0.924 | 0.94 ms |
| **recurrent_ppo_easy_5seeds** | 5 | **63.17% ± 4.91%** | 27.64% | 47.5% | **18.6%** | 69.0% | 301.6 | +4.085 ± 0.684 | 1.17 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Random** | 9.18% | 48.39% | 1.142 | 13.5% | 13.5% | 13.5% | 2070 |
| **Baseline: Frontier BFS** | 0.00% | 8.52% | 2.651 | 44.1% | 44.3% | 44.3% | 1838 |
| **Baseline: Wall Follower** | 0.00% | 17.37% | 0.965 | 19.5% | 19.7% | 19.7% | 2204 |
| **Baseline: Oracle Planner (Privileged Baseline)** | 0.00% | 6.28% | 2.881 | 61.7% | 62.0% | 62.0% | 1798 |
| **ppo_easy_5seeds** | 0.85% | 59.99% | 1.194 | 55.1% | 59.2% | 60.1% | 940 |
| **ppo_nomem_easy_5seeds** | 0.00% | 25.66% | 0.990 | 56.3% | 62.5% | 64.6% | 1064 |
| **recurrent_ppo_easy_5seeds** | 0.35% | 16.04% | 1.170 | 55.1% | 61.3% | 62.8% | 857 |
