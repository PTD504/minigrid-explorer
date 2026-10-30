# MiniGrid Explorer: Autonomous Exploration under Partial Observability and Dynamic Threats

An advanced Reinforcement Learning framework built on top of [Farama MiniGrid](https://github.com/Farama-Foundation/Minigrid) where an agent learns to autonomously explore procedurally generated dungeon environments featuring rooms, corridors, junctions, dead ends, points of interest (POIs), and moving dynamic threats (monsters).

---

## Project Overview

- **Task**: Autonomous exploration and navigation under strict partial observability (egocentric field of view).
- **Core Objectives**:
  1. Maximize revealed area coverage (`revealed_coverage`) and discover Points of Interest (POIs).
  2. Maintain survival constraints against moving dynamic threats.
  3. Navigate efficiently while avoiding redundant loops, cycles, and wall collisions.
  4. Generalize zero-shot across unseen random seeds and complex map configurations.
- **Action Space**: Compact discrete navigation space: `Discrete(3)` (`0 = turn left`, `1 = turn right`, `2 = move forward`).
- **Reproducibility**: 100% deterministic generation and behavior per seed, with strict disjoint splits between training, validation, and held-out test seed sets.
- **Information Boundaries**: Strictly separates agent-facing POMDP observations from evaluator-facing ground-truth metrics.

---

## Repository Architecture

```text
minigrid-explorer/
├── assets/                      # Curated visuals, media, and demonstration assets
│   ├── maps/                    # Procedural map layouts and seed comparisons
│   ├── agents/                  # Policy rollouts and trajectory captures
│   ├── ui/                      # Interactive GUI screenshots
│   └── demo/                    # Video demos and recording GIFs
├── configs/                     # Centralized YAML configurations
│   ├── env/                     # Environment difficulty presets
│   │   ├── easy.yaml            # 17x17 grid, 3-4 rooms, 2 POIs, 1 monster
│   │   ├── medium.yaml          # 25x25 grid, 5-7 rooms, 3 POIs, 1 monster
│   │   └── hard.yaml            # 33x33 grid, 8-10 rooms, 5 POIs, 2 monsters
│   ├── obs/                     # Observation space configs (crop size, memory)
│   │   └── default.yaml
│   ├── reward/                  # Reward scales, penalties, and invariant weights
│   │   └── default.yaml
│   └── seeds.yaml               # Strict disjoint seed splits (train, val, test)
├── explorer/                    # Core Python package
│   ├── envs/                    # MiniGrid environment extensions
│   │   ├── objects.py           # Custom objects (PointOfInterest, Monster)
│   │   ├── visibility.py        # True Line-of-Sight raycasting (no wall bleed)
│   │   ├── map_generator.py     # Procedural dungeon map generator (MST + dead-ends)
│   │   ├── monster.py           # Deterministic monster manager (patrol & chase)
│   │   ├── tracking.py          # Strict POMDP exploration & POI discovery tracker
│   │   ├── wrappers.py          # Observation Memory Wrapper (egocentric crop, no-leak)
│   │   └── explore_env.py       # Main gymnasium environment
│   └── utils/                   # Shared helper utilities
│       ├── __init__.py
│       └── config.py            # Config loader and invariant validator
├── tests/                       # Automated test suites (pytest)
│   ├── test_config.py           # Seed splits and domain invariant tests
│   ├── test_map_generator.py    # Connectivity, rooms, POIs, determinism tests
│   ├── test_monster.py          # Monster LoS, collision, swap, and determinism tests
│   ├── test_reset.py            # Reset reproducibility and action space tests
│   ├── test_tracking.py         # Monotonic coverage, POI LoS, and termination tests
│   ├── test_visibility.py       # Line-of-Sight and wall occlusion tests
│   └── test_wrappers.py         # Egocentric rotation, memory crop, and no-leak tests
├── scripts/                     # Developer tools and demonstration scripts
│   ├── inspect_maps.py          # Batch procedural map visualizer
│   ├── play_builtin.py          # MiniGrid baseline exploration helper
│   ├── view_env.py              # Interactive manual controller (ManualControl)
│   ├── visualize_obs.py         # Observation wrapper and memory crop inspector
│   └── visualize_tracking.py    # Exploration coverage and monster rollout visualizer
├── pyproject.toml               # Project metadata and dependencies (uv / pip)
├── design.md                    # Detailed architectural specification and decision log
├── notes.md                     # Engineering notes and RCA log
└── README.md                    # Project documentation
```

---

## Current Implementation Status

| Component | Status | Details |
|---|:---:|---|
| **Environment Configuration** | Completed | YAML configs for `easy`, `medium`, `hard`, rewards, and disjoint seed ranges. |
| **Procedural Map Generator** | Completed | Non-overlapping rooms, MST corridors, loops/junctions, dead ends, and POIs. |
| **Connectivity & Invariant Tests**| Completed | 100% BFS reachability verification, determinism checks, anti-suicide checks. |
| **Dynamic Threat (Monster)** | Completed | Deterministic patrol & chase behaviors, raycast LoS, collision & swap mechanics. |
| **Exploration Tracking & POIs** | Completed | Strict POMDP tracking, monotonic coverage metrics, zero privileged leak. |
| **Observation Memory Wrapper** | Completed | Egocentric memory crop (KxKx4), one-hot view (7x7x5), step ratio, zero leak. |
| **Reward Engine & Loop Detection** | Scheduled | Telescoping coverage reward, state-repetition penalty. |
| **Baselines Suite** | Scheduled | Random, Wall-Follower, Frontier Explorer, Oracle. |
| **RL Training Pipeline** | Scheduled | PPO & RecurrentPPO with SB3 / sb3-contrib. |
| **Interactive Python UI** | Scheduled | Pygame / pygame-gui real-time inspector and evaluation GUI. |

---

## Installation & Setup

This repository uses [`uv`](https://github.com/astral-sh/uv) for fast and reproducible package management.

### Prerequisites
- Python `>= 3.12`
- Linux (Ubuntu/Debian recommended)

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone <repo_url>
cd minigrid-explorer

# Install dependencies using uv
uv sync
```

Alternatively, install using standard pip in a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Running Tests & Validation

All domain invariants (connectivity, monster behaviors, tracking monotonicity, anti-suicide constraints, seed determinism) are enforced via automated tests:

```bash
# Run full test suite (28 passing tests)
uv run pytest

# Run specific subsystem tests
uv run pytest tests/test_monster.py -v
uv run pytest tests/test_tracking.py -v

# Run code style & linting checks with Ruff
uv run ruff check .
```

---

## Dynamic Threat: Monster Behavior & Configuration

Monsters provide dynamic adversity in the environment, forcing the agent to navigate cautiously and make decisions under risk.

### 1. Behavioral Mechanics
- **Dual-State State Machine**:
  - **Patrol**: Cycles deterministically through room centers (waypoints) via shortest BFS paths.
  - **Chase**: Triggered when the agent is within `monster_detect_radius` (BFS distance) **AND** has an unoccluded raycast Line-of-Sight (`has_line_of_sight`). If LoS is broken (e.g. agent turns a corner), the monster reverts back to patrolling.
- **Deterministic Navigation**:
  - Uses a fixed tie-breaking priority (`Up -> Right -> Down -> Left`) during BFS path reconstruction.
  - Generates zero random calls during `step()`, guaranteeing 100% trajectory reproducibility for any fixed seed.
- **Collision & Elimination**:
  - **Same-tile collision**: Agent or monster enters the tile occupied by the other.
  - **Swap collision**: Agent moves from tile A to B while monster moves from B to A on the same step.
  - Any collision immediately ends the episode (`terminated=True`, `info["end_reason"]="death"`).
- **Multi-Monster Coordination**:
  - In multi-monster setups (`hard`), monsters treat each other as dynamic obstacles, preventing tile overlap.

### 2. Configuration Parameters (YAML & CLI)

Monster behaviors are configured through environment YAML presets (`configs/env/{easy,medium,hard}.yaml`):

| Parameter | Easy | Medium | Hard | Description |
|---|:---:|:---:|:---:|---|
| `num_monsters` | 1 | 1 | 2 | Total number of dynamic threats spawned |
| `monster_period` | 3 | 2 | 2 | Monster moves 1 tile every N environment steps |
| `monster_detect_radius` | 5 | 6 | 7 | BFS awareness radius for switching to chase mode |
| `monster_min_dist` | 5 | 7 | 9 | Minimum initial BFS spawn distance from agent |

---

## Exploration Tracking & POI Discovery

Exploration progress is tracked using the agent's egocentric field of view without leaking privileged ground-truth information to the policy.

### 1. Information Boundary Architecture
- **Agent Policy Scope**:
  - Receives only egocentric partial observations (`vis_mask` from raycasting).
  - Rewards are proportional to newly revealed walkable tiles (`delta_revealed` or Delta N_t) and newly discovered POIs (`delta_pois` or Delta P_t).
  - Episode termination is strictly caused by monster collisions (`death`) or maximum steps reached (`timeout`).
  - No ground-truth map size (|W|) is leaked to the policy.
- **Evaluator Scope**:
  - Computes objective ground-truth benchmark metrics:
    - `revealed_coverage = |R_t intersect W| / |W|`
    - `visited_coverage = |V_t intersect W| / |W|`
    - `poi_discovery_rate = |discovered_pois| / |total_pois|`
  - Encapsulated into the step `info` dictionary for benchmarking and tensorboard logging.

---

## Visualizing Subsystems: Monster & Coverage Telemetry

To visualize how exploration tracking, fog of war, and monster paths interact during an episode, use `scripts/visualize_tracking.py`:

```bash
# Generate 2-panel diagnostic visualization (saved to assets/maps/exploration_tracking_demo.png)
uv run python scripts/visualize_tracking.py --difficulty easy --seed 42 --steps 60 --output assets/maps/exploration_tracking_demo.png

# Visualize on Medium difficulty with 100 steps
uv run python scripts/visualize_tracking.py --difficulty medium --seed 101 --steps 100 --output assets/maps/tracking_medium.png
```

The resulting figure illustrates:
- **Left Panel (Ground Truth)**: The complete dungeon layout, room structures, POIs, and monster patrol trajectory.
- **Right Panel (Exploration State)**: The actual area revealed by the agent (fog-of-war effect), agent path history, and live coverage percentages.

![Exploration Tracking Demo](assets/maps/exploration_tracking_demo.png)

---

## Observation Memory Wrapper & POMDP Interface

To address Partial Observability (POMDP) without leaking ground-truth data, `ObservationMemoryWrapper` (`explorer/envs/wrappers.py`) converts raw MiniGrid observations into structured tensor representations for the Policy:

- **`view` ($7 \times 7 \times 5$, `float32`)**: Egocentric one-hot encoding of currently visible tiles via raycast line-of-sight (`unseen`, `empty`, `wall`, `poi`, `monster`).
- **`memory` ($15 \times 15 \times 4$, `float32`)**: Egocentric crop centered on the agent ($[7, 7]$) and rotated to the agent's heading, tracking accumulated knowledge (`revealed`, `visited`, `known_wall`, `known_poi`).
- **`step_ratio` ($1,$, `float32`)**: Normalized step progression ($t / T \in [0.0, 1.0]$).

### Visualizing the Observation Wrapper
Run `scripts/visualize_obs.py` to inspect both live terminal ASCII logs and a multi-panel diagnostic plot:

```bash
# Run visual inspection on Easy difficulty (saves to assets/demo/observation_wrapper_demo.png)
uv run python scripts/visualize_obs.py --difficulty easy --seed 42 --steps 30
```

The resulting figure compares:
1. **Global World Map**: True dungeon layout with agent trajectory and spawn point.
2. **Local View ($7 \times 7$)**: Immediate line-of-sight field of view (agent at bottom facing UP).
3. **Memory Crop ($15 \times 15$)**: Egocentric spatial memory with the agent at center $[7, 7]$.
4. **4 Memory Channels**: Individual decomposition of revealed area, visited path trail, known walls, and discovered POIs.

![Observation Memory Wrapper Demo](assets/demo/observation_wrapper_demo.png)

---

## Interactive Exploration & Manual Play

You can directly launch an interactive Pygame window to inspect maps, navigate manually, and test environment dynamics in real time.

```bash
# Launch interactive manual play (default: Easy, Seed 42)
uv run python scripts/view_env.py

# Launch on Medium difficulty with a specific seed
uv run python scripts/view_env.py --difficulty medium --seed 101

# Launch on Hard difficulty with 2 monsters
uv run python scripts/view_env.py --difficulty hard --seed 777

# Custom monster stress test: 2 aggressive fast monsters (moves every 1 step, detect radius 8)
uv run python scripts/view_env.py --difficulty easy --num-monsters 2 --monster-period 1 --monster-detect-radius 8
```

### Keyboard Controls
`scripts/view_env.py` inherits directly from MiniGrid's official `ManualControl` class:

| Key | Action | Description |
|---|---|---|
| `W` / `Up Arrow` | **Move Forward** | Steps 1 tile forward (bumps into walls without penalty in free play) |
| `A` / `Left Arrow` | **Turn Left** | Rotates agent 90 degrees counter-clockwise |
| `D` / `Right Arrow` | **Turn Right** | Rotates agent 90 degrees clockwise |
| `1` | **Easy Mode** | Switches to 17x17 map (3-4 rooms, 2 POIs) |
| `2` | **Medium Mode** | Switches to 25x25 map (5-7 rooms, 3 POIs) |
| `3` | **Hard Mode** | Switches to 33x33 map (8-10 rooms, 5 POIs) |
| `N` | **Next Seed** | Generates a new random procedural map on the fly |
| `Backspace` | **Reset** | Resets the agent to the starting position for the current seed |
| `ESC` | **Quit** | Closes the interactive window |

---

## Visualizing Procedural Maps Across Seeds & Difficulties

To inspect and compare procedural map diversity at a glance without manual clicking, use the batch visualizer `scripts/inspect_maps.py`:

### 1. Compare Multiple Seeds for the Same Difficulty
Generate a composite grid showing 4 distinct procedural seeds side-by-side:
```bash
# Compare 4 seeds on Easy difficulty (saves to results/compare_easy_seeds.png)
uv run python scripts/inspect_maps.py --difficulty easy --seeds 7 42 101 2026 --output results/compare_easy_seeds.png

# Compare seeds on Medium difficulty
uv run python scripts/inspect_maps.py --difficulty medium --seeds 10 20 30 40 --output results/compare_medium_seeds.png
```

![Procedural Variety Across Seeds](assets/maps/map_variety_easy.png)

### 2. Compare Difficulty Scaling Side-by-Side
Generate a side-by-side comparison of **Easy (17x17)**, **Medium (25x25)**, and **Hard (33x33)** using the same seed:
```bash
uv run python scripts/inspect_maps.py --compare-difficulties --seed 42 --output results/compare_difficulties.png
```

![Difficulty Scaling Comparison](assets/maps/map_difficulties.png)

---

## Headless Programmatic Inspection

For headless environments or CI pipelines, execute a lightweight rollout test:
```bash
uv run python -c "
from explorer.envs.explore_env import ExploreEnv

env = ExploreEnv(difficulty='easy')
obs, info = env.reset(seed=42)
print(f'Initialized easy map. Walkable: {len(env.walkable_tiles)}, POIs: {len(env.poi_positions)}')
for _ in range(5):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
print(f'Step successful. Cov: {info[\"revealed_coverage\"]*100:.1f}%, Agent Pos: {env.agent_pos}')
"
```

---

## Procedural Map Generation Guarantees

The `ProceduralMapGenerator` enforces several structural invariants:
1. **Guaranteed Connectivity**: Every walkable tile is reachable from the agent's start position via BFS.
2. **Structural Variety**:
   - **Rooms**: Rectangular chambers generated via rejection sampling with margin separation.
   - **Corridors**: Minimum Spanning Tree (MST) guarantees a fully connected topology. Extra edges add cycles and junctions.
   - **Dead Ends**: Explicit branches carved into wall boundaries to create exploration dead ends.
   - **POIs**: Placed with priority at dead-end tips and rooms, respecting minimum spawn distance from the agent.
3. **Strict Partial Observability**: Environment generation logic is completely decoupled from agent observations; absolute coordinates and ground-truth map layout are never leaked to the policy.
