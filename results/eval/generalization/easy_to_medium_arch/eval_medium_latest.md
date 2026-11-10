# MiniGrid Explorer Evaluation Report: MEDIUM Suite

- **Generated:** `2026-11-09 20:03:40`
- **Difficulty Tier:** `medium`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_easy_5seeds** | 5 | **75.21% ± 3.49%** | 23.52% | 53.3% | **42.4%** | 26.8% | 199.2 | +2.308 ± 0.573 | 0.97 ms |
| **ppo_nomem_easy_5seeds** | 5 | **79.98% ± 2.75%** | 29.57% | 57.2% | **44.0%** | 21.0% | 227.3 | +3.117 ± 0.516 | 0.83 ms |
| **recurrent_ppo_easy_5seeds** | 5 | **78.68% ± 1.11%** | 33.47% | 60.3% | **47.4%** | 29.0% | 197.7 | +3.437 ± 0.271 | 1.05 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_easy_5seeds** | 1.41% | 57.46% | 1.358 | 69.0% | 73.5% | 74.7% | 1032 |
| **ppo_nomem_easy_5seeds** | 0.00% | 32.41% | 1.171 | 69.4% | 75.9% | 79.0% | 1205 |
| **recurrent_ppo_easy_5seeds** | 0.37% | 17.94% | 1.201 | 67.7% | 76.2% | 77.8% | 952 |
