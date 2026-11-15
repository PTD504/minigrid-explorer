# MiniGrid Explorer Evaluation Report: MEDIUM Suite

- **Generated:** `2026-11-15 12:18:04`
- **Difficulty Tier:** `medium`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_medium_5seeds_500k [blank:0,1,2,3]** | 5 | **61.33% ± 6.12%** | 23.63% | 40.0% | **12.2%** | 23.8% | 310.5 | +0.302 ± 0.848 | 0.96 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_medium_5seeds_500k [blank:0,1,2,3]** | 0.20% | 59.88% | 0.558 | 51.7% | 57.9% | 59.9% | 1042 |
