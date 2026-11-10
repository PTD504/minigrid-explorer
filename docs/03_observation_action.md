# Observation & Action Spaces (Zero-Leakage Architecture)

## 1. Overview and Design Principles

Autonomous spatial navigation requires constructing spatial understanding from a temporal stream of partial observations without relying on global coordinate frames or omniscient map knowledge.

The observation pipeline enforces three core guarantees:
1. **Strict Partial Observability:** Tiles occluded by walls, unvisited rooms, or entities beyond direct line-of-sight are never visible to the agent.
2. **Zero Information Leakage:** Absolute coordinates $(x, y)$, global orientation $d$, dungeon dimensions $(W, H)$, total POI counts, and distance-to-boundary metrics are completely excluded from the observation.
3. **Egocentric Invariance:** Visual observations and spatial memory crops are translated and rotated to align with the agent's current pose, granting rotational and translational invariance.

---

## 2. Action Space Specification

The action space is a minimal discrete set $\mathcal{A} = \{0, 1, 2\}$:

* **Action 0 (Turn Left):** Rotates the facing direction counter-clockwise ($d_{t+1} = (d_t - 1) \bmod 4$). Physical location remains invariant.
* **Action 1 (Turn Right):** Rotates the facing direction clockwise ($d_{t+1} = (d_t + 1) \bmod 4$). Physical location remains invariant.
* **Action 2 (Move Forward):** Attempts displacement along the current orientation vector $(x_t + \Delta x(d_t), y_t + \Delta y(d_t))$. If the target cell is open floor, location updates; if blocked by walls, location is unchanged and a collision penalty is triggered.

Primitive manipulation actions (e.g., picking up or dropping items) are excluded, removing redundant action branches and focusing learning on navigation and evasion.

---

## 3. Observation Space Formulation

The observation $o_t = (\mathbf{v}_t, \mathbf{m}_t, \rho_t)$ comprises three modalities:

### 3.1 Local Egocentric View ($\mathbf{v}_t \in \{0, 1\}^{7 \times 7 \times 5}$)
A $7 \times 7$ grid projected from the agent's forward line-of-sight cone, with the agent positioned at the bottom center $(3, 6)$ facing upward. Occlusion behind interior walls is computed via raycasting.

The 5 one-hot categorical channels encode:
- **Channel 0 (Unseen):** Cells occluded by obstacles or outside the visual cone.
- **Channel 1 (Empty):** Visible, traversable floor space.
- **Channel 2 (Wall):** Visible impassable wall obstacles.
- **Channel 3 (POI):** Visible Points of Interest within direct sight.
- **Channel 4 (Monster):** Visible dynamic adversaries within direct sight.

### 3.2 Accumulated Spatial Memory Crop ($\mathbf{m}_t \in \{0, 1\}^{K \times K \times 4}$, $K=15$)
While the local view provides instantaneous tactical perception for monster evasion, it lacks long-term context. Symmetrical hallways cause state aliasing and cyclic looping for purely reactive policies.

To provide spatial context without leaking global ground-truth maps, the environment accumulates a persistent binary memory map derived **strictly from historical line-of-sight observations**. At each step, an egocentric crop of size $K \times K$ ($K=15$) is extracted, centered at the agent's position $(x_t, y_t)$ and rotated so that the forward direction is persistently aligned with the vertical axis.

The 4 binary channels encode:
- **Channel 0 (Revealed):** Cells previously unveiled by line-of-sight.
- **Channel 1 (Visited):** Cells physically traversed by the agent.
- **Channel 2 (Known Wall):** Wall obstacles previously identified in sight.
- **Channel 3 (Known POI):** Points of Interest previously discovered in sight.

*Zero-Leakage Boundary:* If the $K \times K$ window extends beyond the outer perimeter of the dungeon, out-of-bounds cells are padded with zeros, identical to unexplored interior space. This prevents the policy from inferring its distance to the map boundaries from padding artifacts.

### 3.3 Normalized Temporal Budget ($\rho_t \in [0, 1]$)
A scalar ratio $\rho_t = t / T$ tracking the elapsed fraction of the step budget. This signal provides the temporal context needed to transition from thorough frontier exploration to urgent survival as $t \to T$.

---

## 4. Information Boundary Analysis

The environment maintains an explicit, verifiable separation between privileged ground-truth state variables ($s_t$) and non-privileged policy inputs ($o_t$):

| State / Modality Variable | Present in World State ($s_t$) | Accessible to Policy ($o_t$) | Enforcement Mechanism / Representation |
|---|:---:|:---:|---|
| **Global Coordinates $(x_t, y_t)$** | Yes | **No** | Egocentric centering; origin stripped from observation |
| **Absolute Orientation $d_t \in \{0, 1, 2, 3\}$** | Yes | **No** | Egocentric rotation; agent persistently faces upward |
| **Full Grid Dimensions ($W \times H$)** | Yes | **No** | Cropped to local bounds ($7 \times 7, 15 \times 15$); boundary-invariant |
| **Total Walkable Floor Area ($\|\mathcal{W}\|$)** | Yes | **No** | Excluded; policy cannot infer global completion percentage directly |
| **Total POI Count ($\|\mathcal{P}\|$)** | Yes | **No** | Excluded; POI discovery is purely opportunistic |
| **Unrevealed Map / Maze Topology** | Yes | **No** | Occluded behind walls via corner-aware Bresenham raycasting |
| **Monsters Outside Line-of-Sight** | Yes | **No** | Stripped; only visible within direct $7 \times 7$ visual field |
| **POIs Outside Line-of-Sight** | Yes | **No** | Stripped until discovered; persisted only in accumulated memory |
| **Navigation / Shortest-Path Oracle** | Yes | **No** | No pathfinding or distance hints provided to policy |
| **Out-of-Bounds Padding** | Yes | **No** | Padded with zeros, indistinguishable from unexplored interior space |
| **Local Visual Field ($\mathbf{v}_t$)** | Derived | **Yes** | $7 \times 7 \times 5$ one-hot egocentric categorical grid |
| **Spatial Memory Crop ($\mathbf{m}_t$)** | Derived | **Yes** | $15 \times 15 \times 4$ egocentric rotated binary memory crop |
| **Elapsed Step Ratio ($\rho_t$)** | Derived | **Yes** | 1D scalar $t / T \in [0, 1]$ providing normalized temporal budget |
