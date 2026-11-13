# MiniGrid Environment Design & Procedural Generation

## 1. Overview and Design Objectives

To evaluate spatial reasoning rather than layout memorization, each episode synthesizes a procedurally randomized dungeon topology parameterized by an integer seed. The environment satisfies three core requirements:
1. **Guaranteed Reachability:** All walkable floor tiles, POIs, and starting points reside in a single connected graph component verified through flood-fill traversal.
2. **Structural Realism:** Dungeons contain realistic navigable geometries, including multiple rooms, corridors, T- and cross-junctions, dead ends, and multi-path loops.
3. **Deterministic Reproducibility:** Identical seeds generate identical map layouts, entity coordinates, and threat movement trajectories.

---

## 2. Procedural Map Generation

The procedural map generation executes in five logical stages:

1. **Room Partitioning and Placement:**
   A target number of non-overlapping rectangular rooms are sampled within the grid interior via rejection sampling. Dimensions are bounded between specified minimum and maximum sizes, with a 1-tile outer perimeter margin and spacing enforced between rooms to maintain corridor separation.
2. **Corridor Synthesis via Minimum Spanning Tree (MST):**
   Room centroids define vertices in a complete graph where edge weights equal Manhattan distances. Kruskal's algorithm computes a Minimum Spanning Tree (MST) to establish base connectivity, carved as orthogonal L-shaped corridors (50% horizontal-first, 50% vertical-first). Because an MST is strictly acyclic, it guarantees complete room reachability with minimal floor footprint.
3. **Multi-Path Cycles (Extra Corridors) and Dead Ends:**
   * **Extra Corridors (Cycles):** A pure MST contains no alternative pathways; a monster blocking a single corridor would render sections of the map completely inaccessible. To introduce tactical evasion routes, $N_{\text{extra}}$ non-tree candidate edges between rooms are randomly selected and carved as additional L-shaped corridors. This injects topological cycles (loops) into the layout, enabling the agent to flank, circle around, or evade patrolling threats.
   * **Dead Ends:** The generator injects $N_{\text{dead}}$ blind corridor spurs branching from existing walkable tiles into solid interior wall sections. Each branch has a randomized length of 2 to 3 cells (`branch_len` $\in [2, 3]$), carved such that new tiles have strictly one walkable neighbor (no accidental loop generation).
4. **Topological Verification:**
   A breadth-first search (BFS) flood-fill initiates from the agent starting room. If the total number of connected reachable tiles does not match the total carved walkable floor tiles $|\mathcal{W}|$, the configuration is rejected and re-seeded.
5. **Entity Spawning & POI Prioritization:**
   * **Agent Placement:** The agent spawns in the centroid of the initial room.
   * **Point of Interest (POI) Placement:** POIs are placed across reachable tiles subject to a minimum Manhattan distance constraint from the agent spawn. Crucially, the generator **prioritizes placing POIs at the deepest extremities of dead ends**. This establishes a risk-reward dynamic: reaching full coverage and maximum reward forces the agent to commit to blind spurs, heightening danger when monsters are nearby and testing loop recovery during backtracking.
   * **Monster Placement:** Dynamic monsters spawn across walkable tiles meeting a minimum BFS distance threshold from the agent, preventing instant spawn collisions.

---

## 3. Dynamic Threat Mechanics

Dynamic threats follow deterministic movement logic and a two-state behavioral model:

* **Detection Radius ($r_{\text{detect}}$) & State Transitions:**
  At every simulation step matching the monster's movement period ($t \pmod \tau = 0$), the monster evaluates its sensory field relative to the agent using a two-stage condition:
  1. *Line-of-Sight (LoS) Raycasting:* A corner-aware Bresenham raycasting check (`has_line_of_sight`) ensures no opaque wall obstacles obstruct the direct line between monster and agent. Diagonal rays cannot slip through adjacent wall corners.
  2. *Navigable Walking Distance:* If LoS is unobstructed, a breadth-first search (BFS) computes the shortest walkable path to the agent. The agent is detected if and only if the path length satisfies:
     $$d_{\text{BFS}}(\mathbf{p}_{\text{monster}}, \mathbf{p}_{\text{agent}}) \le r_{\text{detect}}$$
  If both conditions hold, the monster enters **`chase`** mode. If either condition fails, the monster remains in or reverts to **`patrol`** mode.
* **Patrol Behavior:**
  In `patrol` mode, the monster cycles deterministically through fixed room waypoints. It computes a BFS shortest path toward its current target room center and advances 1 step every $\tau$ environment ticks. Upon reaching a waypoint, it targets the next room in sequence.
* **Pursuit Behavior:**
  In `chase` mode, the monster computes the shortest BFS path to the agent's current position and advances 1 cell along the path gradient toward the agent. Monsters treat other monsters as impassable obstacles, preventing stacking.
* **Collision Semantics:**
  A lethal collision occurs if a monster occupies the agent's cell or swaps positions across adjacent cells during a single transition step. Collisions trigger instantaneous episode termination ($\text{death}_t = 1$).

---

## 4. Empirical Horizon Calibration & Difficulty Configurations

### 4.1 Calibration Methodology (`scripts/calibrate_max_steps.py`)

The episode horizon budget $T$ is a critical environment parameter:
* If $T$ is too restrictive, stochastic exploration is starved, legitimate backtracking from dead ends is cut short, and artificial truncation skews performance.
* If $T$ is excessively generous, temporal efficiency pressure diminishes, and policies may exploit non-terminating exploratory loops.

To empirically ground $T$, an empirical calibration was conducted across 100 dedicated development seeds (`dev`: seeds 3,000,000 to 3,000,099) using `scripts/calibrate_max_steps.py` with an expanded safety ceiling ($3 \times T$). The study evaluated two reference baselines across two threat regimes:
1. **Agents Evaluated:**
   * **Frontier BFS Explorer (`FrontierExplorerAgent`):** Realistic robotic baseline navigating to unrevealed boundary frontiers without prior map knowledge.
   * **Oracle Planner (`OraclePlannerAgent`):** Privileged baseline with full topological omniscience computing theoretical upper-bound trajectories.
2. **Threat Regimes:**
   * `monster_mode: "off"`: Measures pure topological maze clearance without adversarial disruption.
   * `monster_mode: "invulnerable"`: Retains full dynamic monster patrol and pursuit mechanics while disabling collision lethality, directly isolating evasive detour overhead.

### 4.2 Calibration Results & Step Budget Determination

| Difficulty Tier | Grid Size | Frontier `off` $\text{Mean}$ ($P_{95}$) | Frontier `invulnerable` $\text{Mean}$ ($P_{95}$) | Oracle $\text{Mean}$ ($P_{95}$) | Calibrated Budget ($T$) | Headroom Factor (vs $P_{95}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Easy** | $17 \times 17$ | $34.5$ ($53.1$) | $38.4$ ($61.0$) | $27.9$ ($41.0$) | **140** | **$2.30\times - 2.64\times$** |
| **Medium** | $25 \times 25$ | $98.3$ ($138.1$) | $102.5$ ($148.0$) | $74.5$ ($108.2$) | **390** | **$2.64\times - 2.82\times$** |
| **Hard** | $33 \times 33$ | $201.0$ ($263.0$) | $226.9$ ($376.0$) | $149.1$ ($194.1$) | **740** | **$1.97\times - 2.81\times$** |

*Steps represent the trajectory length required to reach $\ge 95\%$ revealed floor coverage.*

**Key Design Insights:**
1. **Exploration Slack:** Setting $T \in \{140, 390, 740\}$ provides a consistent $\approx 2.0\times - 2.8\times$ multiplier over the 95th percentile ($P_{95}$) completion time of a non-learning heuristic frontier agent ($\approx 3.3\times - 3.9\times$ over mean completion time). This ensures RL policies have ample slack to recover from suboptimal choices and dead ends.
2. **Evasion Overhead Absorption:** Dynamic threats induce an evasive step overhead of $+11.3\%$ (Easy), $+4.3\%$ (Medium), and $+12.9\%$ (Hard). The calibrated horizon comfortably absorbs detour paths forced by patrolling monsters.

---

## 5. Difficulty Tier Summary

The environment configuration across tiers is summarized below:

| Parameter | Easy | Medium | Hard |
|---|:---:|:---:|:---:|
| Grid Dimension ($W \times H$) | $17 \times 17$ | $25 \times 25$ | $33 \times 33$ |
| Room Count Range | $3 - 4$ | $5 - 7$ | $8 - 10$ |
| Room Size Range | $4 \times 4 \dots 6 \times 6$ | $4 \times 4 \dots 7 \times 7$ | $4 \times 4 \dots 8 \times 8$ |
| POI Count | 2 | 3 | 5 |
| Monster Count ($M$) | 1 | 1 | 2 |
| Monster Movement Period ($\tau$) | 3 steps | 2 steps | 2 steps |
| Detection Radius ($r_{\text{detect}}$) | 5 cells | 6 cells | 7 cells |
| Dead Ends ($N_{\text{dead}}$) | 2 | 3 | 4 |
| Extra Corridors ($N_{\text{extra}}$) | 1 | 2 | 3 |
| Maximum Step Budget ($T$) | 140 | 390 | 740 |
| Target Coverage Threshold ($\tau_{\text{cov}}$) | $95\%$ | $95\%$ | $95\%$ |

---

## 6. Seed Partitioning

To avoid data leakage across stages of development and evaluation, seeds are partitioned into strictly disjoint numeric ranges:
- **Training Set (`train`):** Seeds 0 to 99,999 (100,000 distinct procedural dungeon variations).
- **Validation Set (`val`):** Seeds 1,000,000 to 1,000,049 (50 fixed seeds for checkpoint tracking and early stopping).
- **Held-Out Test Set (`test`):** Seeds 2,000,000 to 2,000,099 (100 unseen seeds for final evaluation).
- **Development & Calibration Set (`dev`):** Seeds 3,000,000 to 3,000,099 (100 seeds for baseline benchmarking, horizon calibration, and hyperparameter tuning).
