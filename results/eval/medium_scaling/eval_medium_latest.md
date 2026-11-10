# MiniGrid Explorer Evaluation Report: MEDIUM Suite

- **Generated:** `2026-11-09 18:40:18`
- **Difficulty Tier:** `medium`
- **Episodes Evaluated:** `100` (Seeds: `2000000..2000099`)

## Table 1: Primary Exploration & Survival Metrics

| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Random** | 1 | **22.22% ± 7.46%** | 3.89% | 5.7% | **0.0%** | 100.0% | 36.5 | -1.649 ± 0.487 | 0.42 ms |
| **Baseline: Frontier BFS** | 1 | **59.90% ± 28.60%** | 14.85% | 42.0% | **23.0%** | 77.0% | 50.9 | +1.695 ± 2.850 | 0.47 ms |
| **Baseline: Wall Follower** | 1 | **32.90% ± 17.19%** | 14.70% | 19.3% | **0.0%** | 92.0% | 106.1 | -0.887 ± 1.339 | 0.36 ms |
| **Baseline: Oracle Planner (Privileged Baseline)** | 1 | **77.46% ± 25.90%** | 18.98% | 59.0% | **51.0%** | 49.0% | 60.8 | +3.703 ± 3.100 | 0.46 ms |
| **ppo_medium_5seeds** | 5 | **71.83% ± 2.59%** | 20.80% | 49.5% | **33.6%** | 27.4% | 221.8 | +1.573 ± 0.426 | 0.97 ms |
| **ppo_nomem_medium_5seeds** | 5 | **81.79% ± 2.21%** | 29.16% | 58.0% | **51.0%** | 9.2% | 247.8 | +3.364 ± 0.523 | 0.83 ms |

## Table 2: Navigation Regularization & Diagnostic Telemetry

| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Random** | 8.82% | 45.14% | 1.410 | 22.2% | 22.2% | 22.2% | 2369 |
| **Baseline: Frontier BFS** | 0.00% | 9.18% | 2.720 | 59.2% | 59.9% | 59.9% | 2121 |
| **Baseline: Wall Follower** | 0.00% | 28.75% | 1.310 | 32.4% | 32.9% | 32.9% | 2746 |
| **Baseline: Oracle Planner (Privileged Baseline)** | 0.00% | 5.56% | 2.907 | 76.5% | 77.5% | 77.5% | 2176 |
| **ppo_medium_5seeds** | 1.68% | 67.44% | 1.221 | 65.9% | 70.0% | 71.1% | 1031 |
| **ppo_nomem_medium_5seeds** | 0.03% | 32.02% | 1.035 | 68.7% | 77.0% | 80.3% | 1211 |
