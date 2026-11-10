# MiniGrid Explorer Evaluation Report: HARD Suite

- **Generated:** `2026-11-09 20:21:03`
- **Difficulty Tier:** `hard`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_medium_5seeds** | 5 | **57.67% ± 6.22%** | 17.50% | 39.2% | **10.8%** | 67.8% | 337.2 | +1.432 ± 1.185 | 1.05 ms |
| **ppo_nomem_medium_5seeds** | 5 | **68.40% ± 3.50%** | 26.47% | 49.8% | **20.8%** | 45.2% | 451.2 | +3.850 ± 0.817 | 0.93 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_medium_5seeds** | 0.91% | 66.14% | 1.111 | 53.4% | 56.8% | 57.4% | 950 |
| **ppo_nomem_medium_5seeds** | 0.03% | 31.64% | 0.836 | 56.9% | 65.3% | 67.6% | 1075 |
