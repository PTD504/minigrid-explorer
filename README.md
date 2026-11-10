# MiniGrid Explorer: Autonomous Exploration under Partial Observability and Dynamic Threats

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![MiniGrid](https://img.shields.io/badge/minigrid-3.2.0-green.svg)](https://github.com/Farama-Foundation/Minigrid)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.9.2-orange.svg)](https://github.com/DLR-RM/stable-baselines3)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-red.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-119%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced Reinforcement Learning framework built on top of [Farama MiniGrid](https://github.com/Farama-Foundation/Minigrid) where an autonomous agent learns to explore procedurally generated dungeon environments featuring rooms, corridors, junctions, dead ends, hidden Points of Interest (POIs), and moving dynamic threats (monsters) under strict partial observability (POMDP).

---

## Table of Contents

1. [Quick Start: Smoke Test & Inference](#1-quick-start-smoke-test--inference)
2. [Project Overview & POMDP Formulation](#2-project-overview--pomdp-formulation)
3. [Repository Architecture](#3-repository-architecture)
4. [Installation & Dependency Versions](#4-installation--dependency-versions)
5. [MiniGrid Environment & Procedural Map Design](#5-minigrid-environment--procedural-map-design)
6. [Dynamic Threat: Monster Mechanics & Kinematics](#6-dynamic-threat-monster-mechanics--kinematics)
7. [Observation Space & Spatial Memory Architecture](#7-observation-space--spatial-memory-architecture)
8. [Reward Configuration & Engineering](#8-reward-configuration--engineering)
9. [Baseline Exploration Algorithms](#9-baseline-exploration-algorithms)
10. [Training RL Policies (Workflows & Flags)](#10-training-rl-policies-workflows--flags)
11. [Model Evaluation & Benchmarking Suite](#11-model-evaluation--benchmarking-suite)
12. [Interactive Python Research UI (Dashboard)](#12-interactive-python-research-ui-dashboard)
13. [Diagnostic Visualization Tools](#13-diagnostic-visualization-tools)
14. [Pre-Trained Model Checkpoints & Aliases](#14-pre-trained-model-checkpoints--aliases)
15. [Seed Configuration & Split Rigor](#15-seed-configuration--split-rigor)
16. [Automated Testing & Code Quality](#16-automated-testing--code-quality)
17. [Expected Output Files & Artifacts](#17-expected-output-files--artifacts)
18. [Known Codebase & Software Limitations](#18-known-codebase--software-limitations)

---

## 1. Quick Start: Smoke Test & Inference

Pre-trained model checkpoints are provided in `results/checkpoints/`, enabling map generation, model loading, deterministic policy inference, and live telemetry out-of-the-box without retraining from scratch.

### Headless Inference Evaluation
Run deterministic inference of the pre-trained PPO model across 2 validation episodes on Medium difficulty:

```bash
uv run python scripts/evaluate.py --checkpoint ppo_medium_5seeds --difficulty medium --n-episodes 2
```
*Outputs a terminal telemetry table displaying Revealed Coverage (Mean ± Std), Success Rate, Monster Death Rate, Reward Breakdown, and report file paths.*

### Interactive Research GUI Viewer
Launch the real-time Pygame research dashboard with the pre-trained PPO policy actively exploring a Medium ($25 \times 25$) dungeon:

```bash
# Interactive GUI with pre-trained alias (opens desktop Pygame window)
uv run python scripts/view_ui.py --difficulty medium --agent rl --checkpoint medium

# Interactive GUI with a specific checkpoint (.zip file)
uv run python scripts/view_ui.py --difficulty medium --checkpoint path/to/specific/best_model.zip

# Headless mode (for remote servers/CI, runs 20 steps and saves screenshot)
uv run python scripts/view_ui.py --difficulty medium --agent rl --checkpoint medium --steps 20 --headless --screenshot-output results/ui_viewer_demo.png
```

---

## 2. Project Overview & POMDP Formulation

- **Partially Observable Markov Decision Process (POMDP)**: The agent has **no global map access**, no pre-computed shortest paths, and cannot perceive monsters or walls through occluding obstacles. All navigation decisions rely strictly on egocentric Line-of-Sight (LoS) observations and accumulated spatial memory.
- **Core Objectives**:
  1. **Maximize Area Coverage**: Reach $\ge 95\%$ revealed walkable floor coverage (`revealed_coverage`).
  2. **Discover Hidden POIs**: Locate Points of Interest positioned in dead ends and rooms.
  3. **Dynamic Threat Survival**: Evade moving hostile monsters patrolling and chasing via raycast LoS.
  4. **Efficient Navigation**: Avoid wall collisions, cycling, and repetitive dead-end looping.
  5. **Zero-Shot Generalization**: Explore unseen procedural layouts from held-out random seeds.
- **Focused Discrete Action Space**: Because the objective is pure spatial exploration and navigation without inventory mechanics, the action space is restricted to only essential movement primitives: `Discrete(3)` (`0 = Turn Left`, `1 = Turn Right`, `2 = Move Forward`). Standard MiniGrid object manipulation actions (`pickup`, `drop`, `toggle`) are omitted as they are not needed for map exploration.
- **Strict Information Boundaries**: Ground-truth diagnostic metrics (such as complete dungeon floor area, undiscovered POI locations, and monster patrol paths) are tracked exclusively for benchmarking and logging, strictly isolated from agent observation tensors to preserve POMDP constraints.

---

## 3. Repository Architecture

```text
minigrid-explorer/
├── assets/                      # Curated visual demonstrations, maps, and UI captures
│   ├── agents/                  # Comparative baseline trajectories and coverage curves
│   ├── demo/                    # Observation memory wrapper decompositions
│   ├── maps/                    # Exploration tracking and procedural map grids
│   ├── plots/                   # Publication-quality training dynamics and evaluation curves
│   └── ui/                      # Research GUI screenshots
├── configs/                     # Centralized YAML configuration files
│   ├── env/                     # Environment difficulty presets
│   │   ├── easy.yaml            # 17x17 grid, 3-4 rooms, 2 POIs, 1 monster, T=140
│   │   ├── medium.yaml          # 25x25 grid, 5-7 rooms, 3 POIs, 1 monster, T=390
│   │   └── hard.yaml            # 33x33 grid, 8-10 rooms, 5 POIs, 2 monsters, T=740
│   ├── obs/                     # Observation space parameters (7x7 view, 15x15 memory)
│   │   └── default.yaml
│   ├── reward/                  # Reward weights, penalties, and anti-hacking scales
│   │   └── default.yaml
│   ├── train/                   # RL algorithm hyperparameters
│   │   ├── ppo.yaml             # PPO learning rates, batch sizes, rollout steps
│   │   └── recurrent_ppo.yaml   # Recurrent PPO (LSTM) sequence configurations
│   └── seeds.yaml               # Disjoint seed splits (train, val, test, dev)
├── docs/                        # Technical reports, POMDP formulations, and benchmark documentation
├── explorer/                    # Core Python package
│   ├── agents/                  # Exploration agents and policies
│   │   ├── base.py              # BaseAgent abstract interface
│   │   ├── frontier_explorer.py # Classical Frontier Exploration (BFS Yamauchi baseline)
│   │   ├── oracle_planner.py    # Privileged shortest-path planner (upper bound)
│   │   ├── random_agent.py      # Uniform random agent (naive lower bound)
│   │   ├── rl_agent.py          # Stable-Baselines3 inference wrapper
│   │   ├── policy.py            # Dual-stream NatureCNN feature extractor
│   │   └── wall_follower.py     # Deterministic left-hand wall follower
│   ├── envs/                    # MiniGrid environment extensions
│   │   ├── explore_env.py       # Core Gymnasium exploration environment
│   │   ├── map_generator.py     # Procedural dungeon generator (MST + dead-ends)
│   │   ├── monster.py           # Deterministic monster AI (patrol & chase FSM)
│   │   ├── objects.py           # Custom entities (PointOfInterest, Monster)
│   │   ├── tracking.py          # Strict POMDP exploration & POI discovery tracker
│   │   ├── visibility.py        # Raycasting Line-of-Sight (no wall bleed)
│   │   └── wrappers.py          # Observation Memory Wrapper (egocentric crop, no-leak)
│   ├── eval/                    # Evaluation and benchmarking engine
│   │   └── evaluator.py         # Multi-seed metric aggregation and telemetry reporting
│   ├── rewards.py               # Reward engine, telescoping sum, and O(1) loop suppression
│   ├── train.py                 # Multi-run training orchestration and checkpoint manager
│   ├── ui/                      # Interactive Pygame research GUI
│   │   └── app.py               # Dashboard implementation, dynamic HUD, and tile renderer
│   └── utils/                   # Shared helper utilities
│       ├── config.py            # Config loader and invariant validator
│       └── seeding.py           # Stateless SeedSequence split generator
├── results/                     # Trained model checkpoints and multi-seed evaluation telemetry reports
├── scripts/                     # Operational CLI scripts
│   ├── calibrate_max_steps.py   # Horizon calibration and step budget empirical solver
│   ├── evaluate.py              # Unified policy & baseline benchmark evaluation engine
│   ├── inspect_maps.py          # Batch procedural map generator and visualizer
│   ├── play_builtin.py          # Interactive MiniGrid baseline helper
│   ├── plot_tensorboard.py      # Single-experiment offline curve viewer
│   ├── plot_training_curves.py  # Multi-checkpoint comparative training curve plotter
│   ├── train.py                 # Multi-seed RL training CLI (PPO / RecurrentPPO)
│   ├── view_env.py              # Manual keyboard controller (WASD)
│   ├── view_ui.py               # Interactive research GUI launcher
│   ├── visualize_baseline.py    # Baseline trajectory and coverage curve visualizer
│   ├── visualize_obs.py         # Observation memory wrapper diagnostic visualizer
│   └── visualize_tracking.py    # Exploration tracking and monster trajectory visualizer
├── tests/                       # Automated test suite (119 unit & invariant tests)
│   ├── test_baselines.py        # Baseline agent execution and fairness tests
│   ├── test_config.py           # Configuration invariants and difficulty checks
│   ├── test_evaluator.py        # Evaluation pipeline and metric aggregation tests
│   ├── test_map_generator.py    # Topological connectivity, room reachability, and seeds
│   ├── test_monster.py          # Monster patrol, pursuit, LoS, and swap collisions
│   ├── test_policy.py           # CNN policy forward pass and latent dimension checks
│   ├── test_reset.py            # Seed reproducibility and reset determinism tests
│   ├── test_rewards.py          # Anti-suicide bound, telescoping sum, and loop penalty
│   ├── test_rl_agent.py         # RL agent inference and observation compatibility
│   ├── test_seeding.py          # Disjoint seed splits, RNG isolation, and resume logic
│   ├── test_tracking.py         # Monotonic coverage and POI discovery tests
│   ├── test_train.py            # Training pipeline, checkpoint saving, and callbacks
│   ├── test_ui.py               # UI components, alias resolution, and headless stepping
│   ├── test_visibility.py       # Raycasting Line-of-Sight and wall occlusion tests
│   └── test_wrappers.py         # Memory rotation, egocentric crop, and no-leak tests
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Exact pinned dependency lockfile
└── README.md                    # Operational guide and documentation
```

---

## 4. Installation & Dependency Versions

### Prerequisites
- **Python**: `>= 3.12` (tested with Python 3.12.3)
- **OS**: Linux (Ubuntu 22.04+ recommended) or macOS
- **Hardware**: CPU supported; NVIDIA GPU with CUDA acceleration recommended for training.

### Option A: Installation via `uv` (Recommended)
[`uv`](https://github.com/astral-sh/uv) provides fast virtualenv setup and exact lockfile reproducibility:

```bash
# Clone the repository
git clone <repo_url>
cd minigrid-explorer

# Synchronize virtual environment with pinned dependencies
uv sync
```

### Option B: Installation via standard `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Core Dependency Versions

| Package | Version | Purpose |
|---|---|---|
| `gymnasium` | `>= 1.4.0` | Modern reinforcement learning environment standard |
| `minigrid` | `>= 3.2.0` | Procedural gridworld simulation engine |
| `stable-baselines3` | `>= 2.9.2` | Production-grade PPO implementation |
| `sb3-contrib` | `>= 2.9.2` | RecurrentPPO (LSTM) implementation |
| `torch` | `== 2.13.0` | Deep learning backend (PyTorch) |
| `torchvision` | `== 0.28.0` | Image processing primitives |
| `pygame-gui` | `>= 0.6.14` | Interactive GUI dashboard and widget toolkit |
| `matplotlib` | `>= 3.11.2` | Diagnostic figure rendering and distribution boxplots |
| `networkx` | `>= 3.6.1` | Graph representation for Minimum Spanning Trees (MST) |
| `scipy` | `>= 1.18.1` | Scientific spatial calculations |
| `pandas` | `>= 3.1.0` | Telemetry metric logging and tabulation |
| `tensorboard` | `>= 2.21.0` | Live loss, value, and entropy visualization |
| `wandb` | `>= 0.31.0` | Optional cloud experiment tracking |
| `pytest` | `>= 9.2.0` | Comprehensive unit test runner (119 tests) |
| `ruff` | `>= 0.17.3` | High-performance Python linter and formatter |

---

## 5. MiniGrid Environment & Procedural Map Design

Rather than replacing MiniGrid with an ad-hoc custom simulator, `MiniGrid Explorer` strictly inherits from `minigrid.core.world_object.WorldObj` and `minigrid.minigrid_env.MiniGridEnv`, extending the simulation with custom interactive entities and raycasting visibility.

### 1. Custom World Objects (`explorer/envs/objects.py`)
- **`PointOfInterest` (`WorldObj`)**: Color `yellow`, type `poi`. Represent hidden treasure or targets placed in deep dead ends or rooms. Visible in LoS, non-blocking (walkable).
- **`Monster` (`WorldObj`)**: Color `purple`, type `monster`. Dynamic adversarial agent navigating the grid. Triggers terminal collision upon agent contact.

### 2. Line-of-Sight Raycasting (`explorer/envs/visibility.py`)
Standard MiniGrid permits visibility to "bleed" through corners. We implement a rigorous raycasting Line-of-Sight algorithm based on Bresenham line tracing:
- Rays are cast from the agent's tile $(x, y)$ to all tiles in its $7 \times 7$ field of view.
- A wall tile occludes all tiles lying behind it along the ray angle.
- Prevents the policy from perceiving monsters or rooms across solid boundaries.

### 3. Procedural Dungeon Generation Pipeline (`explorer/envs/map_generator.py`)
1. **Room Placement**: Rejection sampling places non-overlapping rectangular chambers.
2. **MST Corridors**: Delaunay triangulation + Kruskal's Minimum Spanning Tree (MST) guarantees 100% topological connectivity.
3. **Cycles & Junctions**: Adds secondary corridor edges to form loops and alternative escape routes.
4. **Dead Ends**: Explicit branches carved into wall perimeters to test systematic backtracking.
5. **Entity Placement**: POIs and monsters placed with strict minimum-distance constraints from the agent spawn.

### 4. Difficulty Presets & Calibrated Horizon Budgets

Environment difficulty is configured in `configs/env/{easy,medium,hard}.yaml`. Max episode steps ($T_{\text{max}}$) were empirically calibrated via `scripts/calibrate_max_steps.py` across 1,200 episodes on dedicated dev seeds (`3,000,000..3,000,099`) to grant approximately $2.5\times - 3.5\times$ headroom over the 90th-percentile Frontier BFS completion time:

| Parameter | Easy (`easy.yaml`) | Medium (`medium.yaml`) | Hard (`hard.yaml`) | Description |
|---|:---:|:---:|:---:|---|
| **Grid Dimensions ($N \times N$)** | $17 \times 17$ | $25 \times 25$ | $33 \times 33$ | Total maze grid size including outer walls |
| **Room Count Range** | 3 – 4 | 5 – 7 | 8 – 10 | Number of rectangular chambers |
| **Room Dimensions** | $4 \times 4$ to $6 \times 6$ | $4 \times 4$ to $7 \times 7$ | $4 \times 4$ to $8 \times 8$ | Bounding box bounds for room sampling |
| **Points of Interest (POIs)** | 2 | 3 | 5 | Total hidden targets to discover |
| **Monsters ($M$)** | 1 | 1 | 2 | Number of dynamic threats spawned |
| **Monster Period ($\Delta t_{\text{monster}}$)** | 3 | 2 | 2 | Monster moves 1 tile every $N$ agent steps |
| **Detection Radius ($R_{\text{detect}}$)** | 5 | 6 | 7 | Line-of-Sight pursuit trigger threshold |
| **Corridors & Dead Ends** | 2 dead ends, 1 loop | 3 dead ends, 2 loops | 4 dead ends, 3 loops | Structural complexity |
| **Calibrated Max Steps ($T_{\text{max}}$)** | **140** | **390** | **740** | Empirically calibrated horizon budget |
| **Success Threshold ($\tau$)** | $\ge 95\%$ | $\ge 95\%$ | $\ge 95\%$ | Walkable revealed coverage threshold |

---

## 6. Dynamic Threat: Monster Mechanics & Kinematics

Monsters (`explorer/envs/monster.py`) provide deterministic dynamic hostility, forcing the agent to weigh exploration returns against collision risks:

1. **Finite State Machine (FSM)**:
   - **Patrol Mode**: Iterates cyclically through room center waypoints via BFS shortest paths.
   - **Chase Mode**: Triggered when the agent is within $R_{\text{detect}}$ **AND** has unoccluded raycast Line-of-Sight. If Line-of-Sight is broken (e.g. agent turns a corner), the monster pursues the last known location for up to 4 steps before resuming patrol.
2. **100% Deterministic Kinematics**:
   - Tie-breaking in BFS paths follows fixed priority (`Up -> Right -> Down -> Left`).
   - Zero random numbers are generated during `step()`, guaranteeing 100% trajectory reproducibility for any seed.
3. **Collision Detection**:
   - **Same-tile collision**: Agent or monster enters the tile occupied by the other.
   - **Swap collision**: Agent moves $A \rightarrow B$ while monster moves $B \rightarrow A$ on the same tick.
   - Any collision triggers immediate termination: `terminated=True`, `info["end_reason"]="death"`.
4. **Multi-Monster Avoidance**: On Hard difficulty ($M=2$), monsters treat each other as dynamic obstacles, preventing overlap.

---

## 7. Observation Space & Spatial Memory Architecture

Under partial observability (POMDP), raw observations must capture both immediate local hazards and accumulated spatial knowledge without leaking ground-truth map coordinates. `ObservationMemoryWrapper` (`explorer/envs/wrappers.py`) structures agent inputs into a gym `Dict` space:

```
Obs Dict
├── "view":        Box(0.0, 1.0, (7, 7, 5), float32)   # Local Line-of-Sight view
├── "memory":      Box(0.0, 1.0, (15, 15, 4), float32) # Egocentric spatial memory crop
└── "step_ratio":  Box(0.0, 1.0, (1,), float32)        # Normalized horizon progress (t / T)
```

1. **Local View ($7 \times 7 \times 5$, `float32`)**:
   - Egocentric one-hot encoding of currently visible tiles in front of the agent (agent at $[6, 3]$ facing UP).
   - 5 categorical feature channels: `unseen`, `empty_walkable`, `wall`, `poi`, `monster`.
2. **Egocentric Memory Crop ($15 \times 15 \times 4$, `float32`)**:
   - Centered strictly on the agent ($[7, 7]$) and rotated to the agent's current heading.
   - 4 accumulated channels: `revealed_area`, `visited_path`, `known_walls`, `known_pois`.
   - Never exposes global coordinates $(x, y)$, preserving strict POMDP translation/rotation invariance.
3. **Step Ratio ($1,$, `float32`)**:
   - Normalized time progression $t / T_{\text{max}} \in [0.0, 1.0]$, informing the agent of remaining budget.

### Visualizing the Observation Wrapper
Inspect the live ASCII representation and 4-channel decomposition figure:

```bash
uv run python scripts/visualize_obs.py --difficulty easy --seed 42 --steps 30 --save assets/demo/observation_wrapper_demo.png
```

![Observation Memory Wrapper Demo](assets/demo/observation_wrapper_demo.png)

---

## 8. Reward Configuration & Engineering

The reward system (`explorer/rewards.py`) balances multiple operational objectives: encouraging rapid map coverage, locating POIs, avoiding moving threats, and penalizing redundant navigation:

- **Exploration Incentives**: Positive rewards for newly revealed walkable floor tiles and discovering POIs within Line-of-Sight. Re-visiting already known tiles awards no exploration reward, preventing reward farming.
- **Navigation Penalties**: Small step penalty encouraging efficient shortest-path traversal; wall bump penalty suppressing repetitive wall-scraping; and state loop penalty discouraging spinning or trapped cycling.
- **Threat Survival Guarantee**: Severe terminal collision penalty for contact with monsters, calibrated to exceed the maximum possible step penalty across the entire episode budget so that committing suicide is never favored over exploring.
- **Completion Bonus**: Awarded upon reaching $\ge 95\%$ revealed coverage, scaled by remaining steps.

### Configuration Parameters (`configs/reward/default.yaml`)

| Parameter | Default Value | Purpose |
|---|:---:|---|
| `w_reveal` | `0.02` | Reward per newly revealed walkable floor tile |
| `w_poi` | `1.0` | Reward per newly discovered Point of Interest |
| `c_step` | `0.002` | Temporal step penalty encouraging efficient paths |
| `c_bump` | `0.02` | Penalty for walking into walls or boundaries |
| `c_loop` | `0.01` | Penalty for repeating recent states without revealing new tiles |
| `c_death` | `2.0` | Penalty for colliding with a dynamic monster |
| `b_done` | `1.0` | Bonus for reaching $\ge 95\%$ coverage (scaled by remaining time) |
| `loop_window` | `20` | Circular window length for state repetition detection |

---

## 9. Baseline Exploration Algorithms

To establish meaningful benchmark standards, we implement four distinct exploration agents (`explorer/agents/`) spanning naive lower bounds, classical robotics heuristics, and theoretical upper envelopes:

1. **`RandomAgent` (Naive Lower Bound)**: Samples uniformly from $\text{Discrete}(3)$. Lacks spatial memory and serves as the baseline floor.
2. **`WallFollowerAgent` (Reactive Heuristic)**: Implements deterministic left-hand wall following with a two-state machine (`Seek Wall` $\rightarrow$ `Follow Wall`) and immediate obstacle detection.
3. **`FrontierExplorerAgent` (Primary Classical Baseline)**: Implements standard robotics Frontier Exploration (Yamauchi, 1997). Performs Breadth-First Search (BFS) over accumulated memory to target the nearest unrevealed walkable boundary tile, treating active monster locations as transient obstacles.
4. **`OraclePlannerAgent` (Theoretical Upper Bound)**: Operates with privileged ground-truth knowledge of the entire maze and monster coordinates, computing omniscient shortest paths to all unrevealed regions and POIs.

### Visualizing Baseline Trajectories & Running Benchmarks
```bash
# Generate comparative trajectory & coverage figure across all 4 agents on Medium (seed 1000000)
uv run python scripts/visualize_baseline.py --difficulty medium --episodes 5 --plot --plot-output assets/agents/baseline_comparison.png

# Fast trajectory visualization on Easy difficulty
uv run python scripts/visualize_baseline.py --difficulty easy --episodes 10 --verbose

# Run a single agent (e.g. Frontier Explorer) on Hard difficulty
uv run python scripts/visualize_baseline.py --agent frontier --difficulty hard --episodes 5

# Comprehensive 11-metric statistical benchmark across 100 held-out test seeds:
uv run python scripts/evaluate.py --baselines random frontier wall_follower oracle --difficulty medium
```

### Comparative Benchmark Summary (Medium Difficulty)

| Agent | Revealed Cov (%) | Visited Cov (%) | POI Rate (%) | Survival Rate (%) | Steps | Total Reward |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`random`** | $38.4 \pm 8.7$ | $5.1 \pm 1.4$ | $13.3 \pm 18.2$ | $0.0\%$ | $31.2 \pm 11.5$ | $-1.62 \pm 0.38$ |
| **`wall_follower`** | $58.6 \pm 18.2$ | $26.4 \pm 9.1$ | $46.7 \pm 24.5$ | $0.0\%$ | $78.4 \pm 22.0$ | $-0.24 \pm 0.72$ |
| **`frontier`** | **$93.8 \pm 2.6$** | **$31.2 \pm 6.4$** | **$86.7 \pm 16.3$** | **$80.0\%$** | **$112.6 \pm 28.4$** | **$+2.84 \pm 1.15$** |
| **`oracle`** *(privileged)* | $96.5 \pm 1.8$ | $24.8 \pm 3.1$ | $93.3 \pm 13.3$ | $100.0\%$ | $74.2 \pm 14.6$ | $+3.12 \pm 0.85$ |

![Baseline Exploration Comparison on Medium](assets/agents/baseline_comparison.png)

---

## 10. Training RL Policies (Workflows & Flags)

`scripts/train.py` orchestrates multi-seed training with parallel vectorized environments (`SubprocVecEnv`), automated validation callbacks, and checkpointing:

### 1. Standard PPO Training (with Spatial Memory)
```bash
# Production training on Easy difficulty (500k timesteps, 8 parallel envs)
uv run python scripts/train.py --algo ppo --difficulty easy --total-timesteps 500000 --n-envs 8

# Production training on Medium difficulty
uv run python scripts/train.py --algo ppo --difficulty medium --total-timesteps 500000 --n-envs 8

# ⚡ Quick Verification Run (1,000 steps)
uv run python scripts/train.py --algo ppo --difficulty easy --total-timesteps 1000 --n-envs 2 --eval-freq 500 --n-eval-episodes 1 --output-dir results/test_smoke
```

### 2. Recurrent PPO (LSTM Memory)
Trains an LSTM recurrent policy without explicit spatial memory (`--algo recurrent_ppo`):
```bash
# Production run
uv run python scripts/train.py --algo recurrent_ppo --difficulty easy --total-timesteps 500000 --n-envs 8

# ⚡ Quick Verification Run (1,000 steps)
uv run python scripts/train.py --algo recurrent_ppo --difficulty easy --total-timesteps 1000 --n-envs 2 --eval-freq 500 --n-eval-episodes 1 --output-dir results/test_smoke
```

### 3. Controlled Ablation Studies
```bash
# Ablation A: Reactive agent without spatial memory crop (view-only)
uv run python scripts/train.py --algo ppo --difficulty easy --no-memory --total-timesteps 500000

# Ablation B: Disable wall bump penalty (c_bump = 0.0)
uv run python scripts/train.py --algo ppo --difficulty easy --no-bump --exp-name ablation_no_bump

# Ablation C: Disable loop penalty (c_loop = 0.0)
uv run python scripts/train.py --algo ppo --difficulty easy --no-loop --exp-name ablation_no_loop

# Ablation D: Disable both bump and loop penalties
uv run python scripts/train.py --algo ppo --difficulty easy --reward-config no_bump_no_loop
```

### 4. Multi-Seed Training Orchestration
Derive and execute multiple independent training runs from a single root seed to measure variance across runs:
```bash
uv run python scripts/train.py --algo ppo --difficulty easy --seed 2026 --seed-count 3 --total-timesteps 500000
```

### 5. Fault-Tolerant Resuming (`--resume`) & Fine-Tuning (`--init-from`)
- **How `--resume` works**: The training pipeline automatically derives the experiment directory from the configuration parameters (`algo`, `difficulty`, `use_memory`, `reward_config`, `seed`) or `--exp-name`. When `--resume` is supplied with the identical command arguments, the pipeline checks for incomplete runs (missing `DONE` file), loads `latest_model.zip`, restores `run_state.json` (timesteps, best score, resume count), and seamlessly continues training:
  ```bash
  # Resume an interrupted run (must match the original experiment parameters)
  uv run python scripts/train.py --algo ppo --difficulty easy --seed 2026 --seed-count 3 --total-timesteps 500000 --resume
  ```
- **Fine-tuning / Transfer Learning**: Initialize weights from an existing checkpoint on a new difficulty tier:
  ```bash
  uv run python scripts/train.py --algo ppo --difficulty medium --init-from results/checkpoints/ppo_easy_5seeds/run_0_seed_2610532756/best_model.zip
  ```

### 6. Cloud Experiment Tracking (Weights & Biases)
TensorBoard logging is enabled by default in `results/tb_logs/`. To enable cloud synchronization with Weights & Biases:
```bash
uv run python scripts/train.py --algo ppo --difficulty easy --track --wandb-project minigrid-explorer
```

### 7. Training Curve Plotting (`scripts/plot_training_curves.py`)
Extract scalar time-series directly from local TensorBoard event logs (offline, zero W&B dependency), align variable training horizons, and generate publication-quality comparison figures:

```bash
# Compare revealed coverage curves between Memory and No-Memory policies on Easy:
uv run python scripts/plot_training_curves.py --checkpoints ppo_easy_5seeds ppo_nomem_easy_5seeds --metrics coverage

# Compare across multiple metrics (coverage, return, monster deaths) with custom filename:
uv run python scripts/plot_training_curves.py --checkpoints ppo_easy_5seeds ppo_nomem_easy_5seeds recurrent_ppo_easy_5seeds \
    --metrics coverage reward death --output easy_comparison.png

# Plot complete 6-panel evaluation metric suite (2x3 grid):
uv run python scripts/plot_training_curves.py --checkpoints ppo_medium_5seeds ppo_nomem_medium_5seeds --metrics eval

# Quick single-experiment inspection:
uv run python scripts/plot_tensorboard.py --exp-dir results/checkpoints/ppo_easy_5seeds
```
*Outputs: High-resolution PNG figures (300 DPI) saved to `results/plots/`.*

---

## 11. Model Evaluation & Benchmarking Suite

`scripts/evaluate.py` serves as the **single source of truth** for all policy evaluation, multi-seed ensemble aggregation, baseline comparison, and test split benchmarking.

### 1. Evaluating Checkpoints (`scripts/evaluate.py`)
Evaluate one or more trained policy folders or direct `.zip` checkpoint paths:

```bash
# Evaluate Medium 5-seed model on Medium difficulty (default: 100 test seeds 2,000,000..2,000,099):
uv run python scripts/evaluate.py --checkpoint ppo_medium_5seeds --difficulty medium

# Evaluate Easy 5-seed model on 20 test episodes:
uv run python scripts/evaluate.py --checkpoint ppo_easy_5seeds --difficulty easy --n-episodes 20

# Evaluate directly from an explicit checkpoint file path:
uv run python scripts/evaluate.py --checkpoint results/checkpoints/ppo_medium_5seeds/run_0_seed_2610532756/best_model.zip --difficulty medium

# Fast verification mode (evaluates 5 episodes on 1 run per model):
uv run python scripts/evaluate.py --checkpoint ppo_easy_5seeds --difficulty easy --smoke-test
```

### 2. Multi-Model Side-by-Side Comparison (Ensemble $\mu \pm \sigma$)
When evaluating multi-run folders (e.g. `ppo_easy_5seeds` containing `run_0` through `run_4`), the script automatically detects and evaluates all runs (up to `--n-runs`, default: 5), computing ensemble mean and standard deviation across training runs:

```bash
# Compare PPO (with memory), PPO (no memory), and RecurrentPPO (LSTM) on Easy:
uv run python scripts/evaluate.py --checkpoints ppo_easy_5seeds ppo_nomem_easy_5seeds recurrent_ppo_easy_5seeds --difficulty easy

# Compare Memory vs No-Memory models on Medium:
uv run python scripts/evaluate.py --checkpoints ppo_medium_5seeds ppo_nomem_medium_5seeds --difficulty medium
```

### 3. Head-to-Head Comparison with Baselines
Include heuristic/algorithmic baselines (`random`, `frontier`, `wall_follower`, `oracle`, `all`) alongside RL models, or evaluate baselines alone:

```bash
# Compare PPO with Frontier BFS and Random baseline on Medium:
uv run python scripts/evaluate.py --checkpoints ppo_medium_5seeds --baselines frontier random --difficulty medium --n-episodes 30

# Benchmark all 4 baselines together on Easy (no RL model required):
uv run python scripts/evaluate.py --baselines all --difficulty easy --n-episodes 50
```

### 4. Official 100 Held-Out Test Seed Benchmark & Cross-Difficulty Generalization
Executes comprehensive evaluation across the official 100 test seeds (`2,000,000` to `2,000,099` configured in `configs/seeds.yaml`):

```bash
# Official in-distribution benchmark on Easy suite:
uv run python scripts/evaluate.py --checkpoints ppo_easy_5seeds ppo_nomem_easy_5seeds recurrent_ppo_easy_5seeds --difficulty easy

# Official in-distribution benchmark on Medium suite:
uv run python scripts/evaluate.py --checkpoints ppo_medium_5seeds ppo_nomem_medium_5seeds --difficulty medium

# Cross-Difficulty Generalization (Easy-trained models evaluated on larger Medium 25x25 mazes):
uv run python scripts/evaluate.py --checkpoints ppo_easy_5seeds ppo_nomem_easy_5seeds --difficulty medium
```

#### Dual-Table Output Format
Results are printed to terminal and exported to Markdown and JSON in `results/eval/`:
- **Table 1: Primary Exploration & Survival**: Runs, Revealed Coverage (Mean $\pm$ Std), POI Discovery (%), Success Rate ($\ge 95\%$), Monster Deaths (%), Steps, Episodic Return, Policy Latency (ms).
- **Table 2: Navigation Regularization & Telemetry**: Wall Bump Rate (%), Loop Rate (%), Revealed Efficiency (tiles/step), 25% / 50% Coverage Milestones, Simulation FPS.
- **Generated Report Artifacts**:
  - `results/eval/eval_<difficulty>_<timestamp>.md` and `eval_<difficulty>_latest.md`
  - `results/eval/eval_<difficulty>_<timestamp>.json` and `eval_<difficulty>_latest.json`

---

## 12. Interactive Python Research UI (Dashboard)

`MiniGrid Explorer` provides an interactive Pygame dashboard (`explorer/ui/app.py` & `scripts/view_ui.py`) for live visual debugging, parameter manipulation, manual stepping, and telemetry monitoring:

```bash
# Launch interactive dashboard with a specific checkpoint (.zip file):
uv run python scripts/view_ui.py --difficulty medium --checkpoint path/to/specific/best_model.zip

# Launch interactive dashboard with trained RL policy via alias (e.g. 'medium', 'easy', 'nomem', 'recurrent'):
uv run python scripts/view_ui.py --difficulty medium --agent rl --checkpoint medium

# Launch with classical Frontier Explorer:
uv run python scripts/view_ui.py --difficulty easy --agent frontier

# Launch with manual keyboard control:
uv run python scripts/view_ui.py --difficulty easy --agent manual
```

![MiniGrid Explorer Research UI](assets/ui/ui_viewer_demo.png)

### Interactive Dashboard Features
1. **Dynamic Map & Seed Controls**: Switch between Easy ($17 \times 17$), Medium ($25 \times 25$), and Hard ($33 \times 33$) topologies. Randomize seeds (`N`), increment/decrement (`Next`/`Prev`), or reset (`R`).
2. **Multi-Agent Selector**: Toggle dynamically between `Manual (WASD)`, `Frontier Explorer (BFS)`, `Wall Follower`, `Random Agent`, `Oracle Planner`, and `Trained RL Policy`.
3. **Execution Stepping**: `Play / Pause` toggle (`Space`), single-step advancement (`S` / `Right Arrow`), and adjustable simulation FPS (`1x`, `2x`, `5x`, `Max`).
4. **Visual Fog of War**: Real-time Line-of-Sight shading dims unrevealed regions. Press `F` to toggle full global map inspection.
5. **Live Telemetry HUD**: Real-time progress bars for revealed coverage (with 95% completion threshold marker), step budget, POI discovery count, and exact reward component breakdown (`reveal`, `bump`, `loop`, `death`).

### Keyboard Shortcuts Reference

| Key | Action | Description |
|---|---|---|
| `Space` | **Play / Pause** | Toggle continuous simulation playback |
| `S` / `Right Arrow` | **Step Forward** | Advance simulation by exactly 1 environment step |
| `W` / `Up Arrow` | **Move Forward** | Move 1 tile forward (Manual WASD mode) |
| `A` / `Left Arrow` | **Turn Left** | Rotate agent 90° counter-clockwise (Manual mode) |
| `D` | **Turn Right** | Rotate agent 90° clockwise (Manual mode) |
| `Tab` | **Cycle Agent** | Cycle through: Manual $\rightarrow$ Frontier $\rightarrow$ Wall Follower $\rightarrow$ Random $\rightarrow$ Oracle $\rightarrow$ RL |
| `1`, `2`, `3` | **Difficulty** | Switch instantly to Easy (1), Medium (2), or Hard (3) |
| `N` | **Randomize Seed** | Generate a new procedural dungeon layout with a random seed |
| `R` | **Reset Episode** | Reset agent to start position for the current seed |
| `F` | **Toggle Fog** | Toggle Fog-of-War shading on/off for debugging |
| `P` | **Screenshot** | Export current UI frame to disk as a PNG |
| `ESC` | **Quit** | Close the dashboard window |

---

## 13. Diagnostic Visualization Tools

### 1. Batch Procedural Map Generator (`scripts/inspect_maps.py`)
Compare procedural layout diversity across different seeds or difficulty tiers:

```bash
# Compare 4 random seeds side-by-side on Easy difficulty:
uv run python scripts/inspect_maps.py --difficulty easy --seeds 7 42 101 2026 --output assets/maps/map_variety_easy.png

# Compare Easy, Medium, and Hard scaling for seed 42:
uv run python scripts/inspect_maps.py --compare-difficulties --seed 42 --output assets/maps/map_difficulties.png
```

![Procedural Variety Across Seeds](assets/maps/map_variety_easy.png)

### 2. Exploration Tracking & Monster Trajectory Telemetry (`scripts/visualize_tracking.py`)
Inspect the dual-panel diagnostic view comparing ground truth layout vs. revealed fog-of-war area and monster chase paths:

```bash
uv run python scripts/visualize_tracking.py --difficulty easy --seed 42 --steps 60 --output assets/maps/exploration_tracking_demo.png
```

![Exploration Tracking Demo](assets/maps/exploration_tracking_demo.png)

---

## 14. Pre-Trained Model Checkpoints

To enable immediate evaluation, interactive visualization, and reproducible benchmarking **without requiring evaluators or researchers to invest computational resources retraining from scratch**, all converged policy weights across all 5 random seeds and ablation variants are pre-trained and included directly in the repository under `results/checkpoints/`.

You can immediately evaluate any model family with `scripts/evaluate.py` or inspect their behavior visually in real time via `scripts/view_ui.py`:

| Experiment Directory | Model Family | Architecture | Environment / Config |
|---|---|---|---|
| `ppo_easy_5seeds` | **PPO (Memory)** | PPO + Spatial Memory ($15 \times 15$) | Easy ($17 \times 17$), 5 seeds |
| `ppo_nomem_easy_5seeds` | **PPO (No Memory)** | PPO Reactive (View-Only) | Easy ($17 \times 17$), 5 seeds |
| `recurrent_ppo_easy_5seeds` | **RecurrentPPO (LSTM)** | RecurrentPPO + 256-dim LSTM | Easy ($17 \times 17$), 5 seeds |
| `ppo_medium_5seeds` | **PPO (Memory, Med)** | PPO + Spatial Memory ($15 \times 15$) | Medium ($25 \times 25$), 5 seeds |
| `ppo_nomem_medium_5seeds` | **PPO (No Memory, Med)** | PPO Reactive (View-Only) | Medium ($25 \times 25$), 5 seeds |
| `ppo_easy_nomem_no_bump_5seeds` | **Ablation: No Bump** | PPO Reactive ($c_{\text{bump}} = 0$) | Easy ($17 \times 17$), 5 seeds |
| `ppo_easy_nomem_no_loop_5seeds` | **Ablation: No Loop** | PPO Reactive ($c_{\text{loop}} = 0$) | Easy ($17 \times 17$), 5 seeds |
| `ppo_easy_nomem_no_bump_no_loop_5seeds` | **Ablation: Both Off** | PPO Reactive ($c_{\text{bump}} = c_{\text{loop}} = 0$) | Easy ($17 \times 17$), 5 seeds |

### Checkpoint Directory Structure
Inside each experiment folder (e.g. `results/checkpoints/ppo_easy_5seeds/`):
- `run_0_seed_<int>/`, `run_1_seed_<int>/`, ...: Individual training run directories:
  - `best_model.zip`: Checkpoint achieving the highest evaluation coverage on validation seeds.
  - `final_model.zip`: Final policy weights saved at step $T_{\text{total}}$.
  - `run_state.json`: Step counters, loss history, and resume state metadata.
  - `tb_logs/`: Raw TensorBoard scalar logs for learning curve extraction.
- Root directory level:
  - `best_model.zip`: Top-performing run's best checkpoint (for single-model convenience).
  - `run_manifest.json`: Configuration hash, seed list, and git commit hash.
  - `val_summary.json`: Multi-seed aggregated validation performance summary.

---

## 15. Seed Configuration & Split Rigor

To prevent data contamination and guarantee generalization to previously unseen maps, seeds are strictly partitioned in `configs/seeds.yaml`:

```yaml
train:
  start: 0
  count: 100000      # Seeds 0 .. 99,999 (used during RL training)

val:
  start: 1000000
  count: 50          # Seeds 1,000,000 .. 1,000,049 (used for validation checkpoints)

test:
  start: 2000000
  count: 100         # Seeds 2,000,000 .. 2,000,099 (official held-out test suite)

dev:
  start: 3000000
  count: 100         # Seeds 3,000,000 .. 3,000,099 (calibration & parameter tuning)
```

- **Zero Overlap Invariant**:
  $$\text{Train} \cap \text{Val} \cap \text{Test} \cap \text{Dev} \equiv \emptyset$$
  Every training and evaluation run automatically enforces split disjointness via `explorer.utils.config.validate_seed_splits()`.
- **Stateless Seed Derivation**: Worker environment seeds are derived deterministically via `numpy.random.SeedSequence`, ensuring reproducibility across operating systems and CPU architectures.

---

## 16. Automated Testing & Code Quality

The repository maintains an automated test suite enforcing domain invariants (reachability, anti-suicide bounds, Line-of-Sight raycasting, monotonicity, seed determinism):

```bash
# Run complete test suite (119 passing tests)
uv run pytest

# Run specific subsystem test suites
uv run pytest tests/test_baselines.py -v     # Baseline execution and fairness
uv run pytest tests/test_monster.py -v       # Monster patrol, pursuit, and collision
uv run pytest tests/test_rewards.py -v       # Telescoping sum and anti-suicide bound
uv run pytest tests/test_tracking.py -v      # Monotonic coverage and POI discovery
uv run pytest tests/test_ui.py -v            # UI widgets, alias resolution, headless mode
uv run pytest tests/test_wrappers.py -v      # Memory rotation and observation tensors

# Run static code analysis with Ruff
uv run ruff check .
```

---

## 17. Expected Output Files & Artifacts

All operational scripts export structured, machine-readable artifacts into `results/`:

| Output Path | Type | Generating Command | Description |
|---|---|---|---|
| `results/tb_logs/` | Directory | `scripts/train.py` | TensorBoard event files (loss, entropy, value, reward) |
| `results/checkpoints/<run_id>/` | Directory | `scripts/train.py` | Model weights (`best_model.zip`, `final_model.zip`, state JSON) |
| `results/eval/eval_<diff>_<timestamp>.json` | JSON | `scripts/evaluate.py` | Raw evaluation metrics per seed across evaluated models & baselines |
| `results/eval/eval_<diff>_<timestamp>.md` | Markdown | `scripts/evaluate.py` | Formatted markdown telemetry report |
| `results/plots/` | Directory | `scripts/plot_training_curves.py` | Publication-quality multi-checkpoint comparative training curves |
| `assets/agents/baseline_comparison.png` | PNG | `scripts/visualize_baseline.py` | Comparative trajectory and coverage accumulation plot |

---

## 18. Known Codebase & Software Limitations

1. **Pygame UI Initial Loading Latency with RL Checkpoints (Black Screen Lag)**:
   When launching `scripts/view_ui.py` with an RL checkpoint (`.zip`), the Pygame window initially displays a blank/black surface for ~2–4 seconds before rendering the first frame. This delay is caused by PyTorch deserializing model weights and neural graph buffers on the main application thread during startup. Once initialization completes, the simulation loop executes smoothly at the configured target FPS.
2. **Synchronous UI Event & Rendering Loop**:
   The Pygame dashboard executes environment step progression, agent inference, and UI rendering synchronously on the primary application thread. Under maximum simulation speeds (`Max FPS`) on large $33 \times 33$ dungeons, UI event handling (such as button clicks or keyboard polling) can experience minor input latency or frame stutter.
3. **Subprocess VecEnv Memory Duplication in Training**:
   Multi-worker parallel rollout (`SubprocVecEnv` with 8 parallel environments) forks 8 separate Python processes. While this avoids Python's GIL bottleneck for environment step execution, it duplicates memory allocations (~200MB per worker) and relies on synchronous IPC pipes for dictionary observation serialization.
4. **Synchronous File I/O in Evaluation Scripts**:
   During batch evaluation (`scripts/evaluate.py`), episodic trajectory data and metrics across 100 seeds are accumulated in process memory and written synchronously to disk upon benchmark completion, rather than being streamed asynchronously to persistent storage via a dedicated non-blocking worker thread.
