# MiniGrid Explorer Evaluation Report: HARD Suite

- **Generated:** `2026-11-09 20:52:41`
- **Difficulty Tier:** `hard`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_nomem_easy_5seeds** | 5 | **65.21% ± 5.49%** | 25.97% | 46.3% | **18.0%** | 62.8% | 368.1 | +3.775 ± 0.924 | 0.94 ms |
| **ppo_easy_nomem_no_loop_5seeds** | 5 | **64.03% ± 3.24%** | 24.51% | 47.5% | **14.4%** | 59.8% | 406.2 | +3.260 ± 0.561 | 0.94 ms |
| **ppo_easy_nomem_no_bump_5seeds** | 5 | **62.08% ± 7.79%** | 24.68% | 46.2% | **17.2%** | 63.6% | 367.2 | +3.358 ± 1.205 | 0.94 ms |
| **ppo_easy_nomem_no_bump_no_loop_5seeds** | 5 | **66.14% ± 2.31%** | 25.54% | 49.4% | **15.8%** | 57.6% | 422.2 | +3.582 ± 0.237 | 0.95 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_nomem_easy_5seeds** | 0.00% | 25.66% | 0.990 | 56.3% | 62.5% | 64.6% | 1059 |
| **ppo_easy_nomem_no_loop_5seeds** | 0.40% | 32.62% | 0.937 | 53.8% | 61.0% | 63.3% | 1062 |
| **ppo_easy_nomem_no_bump_5seeds** | 0.19% | 29.50% | 0.958 | 52.5% | 58.8% | 61.3% | 1060 |
| **ppo_easy_nomem_no_bump_no_loop_5seeds** | 0.00% | 30.62% | 0.835 | 55.3% | 62.9% | 65.3% | 1053 |
