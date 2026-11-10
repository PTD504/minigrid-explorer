# MiniGrid Explorer Evaluation Report: EASY Suite

- **Generated:** `2026-11-09 18:36:19`
- **Difficulty Tier:** `easy`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_nomem_easy_5seeds** | 5 | **90.67% ± 2.05%** | 28.81% | 62.6% | **77.8%** | 4.6% | 71.5 | +2.609 ± 0.107 | 0.82 ms |
| **ppo_easy_nomem_no_loop_5seeds** | 5 | **88.78% ± 1.75%** | 26.41% | 58.3% | **73.8%** | 5.6% | 72.3 | +2.386 ± 0.165 | 0.90 ms |
| **ppo_easy_nomem_no_bump_5seeds** | 5 | **90.37% ± 0.66%** | 28.77% | 61.8% | **77.2%** | 3.8% | 71.9 | +2.598 ± 0.095 | 0.82 ms |
| **ppo_easy_nomem_no_bump_no_loop_5seeds** | 5 | **90.06% ± 1.34%** | 27.10% | 60.9% | **75.4%** | 4.4% | 73.5 | +2.516 ± 0.106 | 0.82 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo_nomem_easy_5seeds** | 0.00% | 22.92% | 1.677 | 75.6% | 85.6% | 88.8% | 1215 |
| **ppo_easy_nomem_no_loop_5seeds** | 0.51% | 30.58% | 1.649 | 74.2% | 84.3% | 87.2% | 1144 |
| **ppo_easy_nomem_no_bump_5seeds** | 0.08% | 23.63% | 1.687 | 75.5% | 84.8% | 88.5% | 1220 |
| **ppo_easy_nomem_no_bump_no_loop_5seeds** | 0.01% | 27.79% | 1.642 | 74.4% | 85.5% | 88.6% | 1217 |
