# Baselines & Privileged Oracles

## 1. Overview and Benchmarking Objectives

To establish rigorous comparative benchmarks for reinforcement learning policies, the framework implements two distinct categories of baseline agents:
1. **Fair Non-Learning Baselines:** Algorithmic and heuristic agents that operate under identical partial observability constraints, action spaces, seed sets, and episode budgets as the learning agents.
2. **Privileged Oracles:** Theoretical upper-bound planners granted omniscient access to the global dungeon topology and ground-truth entity coordinates. These oracles are strictly separated from fair empirical comparisons.

---

## 2. Fair Non-Learning Baselines

### 2.1 Random Walk Baseline
- **Behavioral Formulation:** At each transition step $t$, the policy samples an action uniformly at random from the discrete action set:
  $$\pi_{\text{random}}(a_t \mid o_t) = \frac{1}{|\mathcal{A}|} = \frac{1}{3}, \quad \forall a_t \in \{0, 1, 2\}$$
- **Information Constraints:** Receives standard partial observations $o_t$, but ignores all perceptual inputs.
- **Benchmarking Relevance:** Measures the statistical lower bound of passive Brownian diffusion in grid environments. It establishes the baseline probability of discovering POIs and surviving roaming monsters through pure stochastic chance.

### 2.2 Frontier Breadth-First Search (BFS) Explorer
- **Behavioral Formulation:** A deterministic non-learning exploration algorithm derived from classic robotics SLAM:
  1. The agent maintains an internal occupancy grid of known floor tiles and observed obstacles, constructed strictly from historical line-of-sight visual fields.
  2. At each step, the algorithm identifies "frontier cells"—known traversable floor cells that are directly adjacent to unobserved, unexplored territory.
  3. A Breadth-First Search (BFS) computes the shortest path from the agent's current coordinates to the nearest frontier cell.
  4. The algorithm emits the primitive action ($a \in \{0, 1, 2\}$) required to traverse the first edge along the computed BFS path.
  5. If no frontiers remain reachable, it defaults to exploring unvisited known cells.
- **Information Constraints:** Operates under strict partial observability. It accesses only the accumulated memory map built from past observations. It possesses zero knowledge of unrevealed terrain, hidden POIs, or dynamic monster positions outside line-of-sight.
- **Benchmarking Relevance:** Serves as the primary deterministic heuristic baseline. It measures how effectively an optimal classical path planner can navigate procedural mazes in the absence of dynamic threat anticipation.

### 2.3 Wall Follower (Right-Hand Rule Heuristic)
- **Behavioral Formulation:** A deterministic topological traversal heuristic:
  1. The agent attempts to maintain contact with a wall obstacle along its right-hand flank.
  2. At each step, it probes adjacent cells in priority order: Turn Right and Move Forward (if right cell is open), Move Forward (if front cell is open), Turn Left (if front is blocked).
- **Information Constraints:** Strictly reactive, utilizing only the immediate $3 \times 3$ local neighborhood.
- **Benchmarking Relevance:** Tests whether procedural mazes with loops and junctions can be solved through simple topological boundary-following without spatial mapping or threat awareness.

---

## 3. Privileged Oracle Planner (Theoretical Upper Bound)

### 3.1 Formulation and Privileged Access
The Oracle Planner is granted omniscient access to the global ground-truth environment state $s_t$:
- The complete global floor plan $W \times H$, including all wall obstacles $\mathcal{O}_{\text{wall}}$ and traversable floor cells $\mathcal{W}$ prior to revelation.
- Exact ground-truth coordinates of all hidden Points of Interest $\mathcal{P}$.
- Real-time global coordinates and movement trajectories of all dynamic monsters.

The oracle computes a global shortest-path tree using all-pairs shortest paths or full-graph Dijkstra search, navigating directly to undiscovered POIs and optimal frontier vantage points while maintaining distance from monster paths.

### 3.2 Methodological Separation
As mandated by empirical benchmarking protocols, the Privileged Oracle is **strictly separated** from fair agent comparisons:
- It receives an informational advantage unavailable to real-world autonomous systems.
- It is reported exclusively as an asymptotic theoretical ceiling to quantify the optimality gap of learned policies.

