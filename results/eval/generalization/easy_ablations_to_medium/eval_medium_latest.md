# MiniGrid Explorer Evaluation Report: MEDIUM Suite

- **Generated:** `2026-11-09 20:37:07`
- **Difficulty Tier:** `medium`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_nomem_easy_5seeds** | 5 | **79.98% ± 2.75%** | 29.57% | 57.2% | **44.0%** | 21.0% | 227.3 | +3.117 ± 0.516 | 0.82 ms |
| **ppo_easy_nomem_no_loop_5seeds** | 5 | **78.11% ± 4.90%** | 27.40% | 54.9% | **43.4%** | 16.6% | 242.4 | +2.855 ± 0.681 | 0.82 ms |
| **ppo_easy_nomem_no_bump_5seeds** | 5 | **81.27% ± 3.63%** | 30.47% | 55.6% | **48.0%** | 24.4% | 221.4 | +3.230 ± 0.565 | 0.82 ms |
| **ppo_easy_nomem_no_bump_no_loop_5seeds** | 5 | **79.89% ± 1.99%** | 28.68% | 57.5% | **44.6%** | 17.0% | 243.9 | +3.042 ± 0.412 | 0.82 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_nomem_easy_5seeds** | 0.00% | 32.41% | 1.171 | 69.4% | 75.9% | 79.0% | 1218 |
| **ppo_easy_nomem_no_loop_5seeds** | 0.41% | 36.15% | 1.023 | 68.1% | 74.7% | 76.9% | 1220 |
| **ppo_easy_nomem_no_bump_5seeds** | 0.05% | 25.43% | 1.117 | 67.9% | 76.0% | 79.9% | 1216 |
| **ppo_easy_nomem_no_bump_no_loop_5seeds** | 0.00% | 35.22% | 1.037 | 68.9% | 75.9% | 78.6% | 1219 |
