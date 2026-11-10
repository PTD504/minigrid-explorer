# MiniGrid Explorer Evaluation Report: EASY Suite

- **Generated:** `2026-11-09 19:27:32`
- **Difficulty Tier:** `easy`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Random** | 1 | **40.00% ± 10.40%** | 7.57% | 20.0% | **0.0%** | 100.0% | 34.3 | -1.532 ± 0.629 | 0.39 ms |
| **Baseline: Frontier BFS** | 1 | **78.10% ± 23.43%** | 16.65% | 48.5% | **53.0%** | 44.0% | 31.8 | +1.421 ± 2.135 | 0.43 ms |
| **Baseline: Wall Follower** | 1 | **50.91% ± 20.51%** | 24.18% | 39.0% | **7.0%** | 77.0% | 65.0 | -0.438 ± 1.507 | 0.36 ms |
| **Baseline: Oracle Planner (Privileged Baseline)** | 1 | **88.34% ± 17.09%** | 18.81% | 61.0% | **71.0%** | 27.0% | 31.2 | +2.379 ± 1.900 | 0.43 ms |
| **ppo_easy_5seeds** | 5 | **88.83% ± 1.21%** | 24.57% | 59.3% | **74.8%** | 11.2% | 61.7 | +2.361 ± 0.122 | 0.98 ms |
| **ppo_nomem_easy_5seeds** | 5 | **90.67% ± 2.05%** | 28.81% | 62.6% | **77.8%** | 4.6% | 71.5 | +2.609 ± 0.107 | 0.82 ms |
| **recurrent_ppo_easy_5seeds** | 5 | **89.30% ± 1.40%** | 34.79% | 64.6% | **74.2%** | 9.2% | 75.1 | +2.535 ± 0.135 | 1.03 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Random** | 8.85% | 47.00% | 1.331 | 39.6% | 40.0% | 40.0% | 2595 |
| **Baseline: Frontier BFS** | 0.00% | 19.86% | 2.669 | 73.4% | 78.1% | 78.1% | 2331 |
| **Baseline: Wall Follower** | 0.00% | 22.06% | 1.303 | 46.6% | 50.4% | 50.4% | 2784 |
| **Baseline: Oracle Planner (Privileged Baseline)** | 0.00% | 13.52% | 3.153 | 83.1% | 88.3% | 88.3% | 2300 |
| **ppo_easy_5seeds** | 0.97% | 33.42% | 1.945 | 74.9% | 85.4% | 87.7% | 1017 |
| **ppo_nomem_easy_5seeds** | 0.00% | 22.92% | 1.677 | 75.6% | 85.6% | 88.8% | 1228 |
| **recurrent_ppo_easy_5seeds** | 0.19% | 12.92% | 1.461 | 69.2% | 83.3% | 87.8% | 969 |
