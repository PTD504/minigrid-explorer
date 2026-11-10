# MiniGrid Explorer Evaluation Report: EASY Suite

- **Generated:** `2026-11-10 08:06:20`
- **Difficulty Tier:** `easy`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_medium_5seeds** | 5 | **85.25% ± 2.98%** | 21.25% | 53.2% | **63.8%** | 13.2% | 68.0 | +1.941 ± 0.323 | 0.99 ms |
| **ppo_nomem_medium_5seeds** | 5 | **88.29% ± 2.33%** | 26.81% | 59.7% | **71.2%** | 4.8% | 75.2 | +2.399 ± 0.245 | 0.82 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_medium_5seeds** | 1.51% | 46.92% | 1.818 | 73.6% | 83.1% | 84.7% | 1016 |
| **ppo_nomem_medium_5seeds** | 0.09% | 31.57% | 1.575 | 73.3% | 84.3% | 87.3% | 1213 |
