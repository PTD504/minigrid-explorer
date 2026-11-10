# Problem Formulation & POMDP Modeling

## 1. Task Definition

The objective is autonomous spatial exploration in procedurally generated 2D grid environments characterized by partial observability, dynamic threats, and a finite step budget. The agent starts at an unobserved position within a randomized dungeon comprising interconnected rooms, corridors, junctions, and dead ends. It must systematically reveal unknown terrain, discover hidden Points of Interest (POIs), and survive encounters with moving adversaries.

The problem presents three core challenges:
1. **Information Asymmetry:** The global layout, obstacle configurations, and POI locations are unknown a priori.
2. **Dynamic Non-Stationarity:** Adversarial entities actively patrol corridors and pursue the agent upon line-of-sight detection, requiring continuous trade-offs between frontier expansion and evasion.
3. **Finite Budget Constraint:** Episodes are bounded by a fixed maximum horizon $T$, penalizing backtracking, idling, and dead-end trapping.

---

## 2. POMDP Mathematical Formulation

We formalize the environment as a discounted Partially Observable Markov Decision Process (POMDP), defined by the 7-tuple:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

### 2.1 State Space $\mathcal{S}$
The true underlying state $s_t \in \mathcal{S}$ is unobserved by the agent. It encapsulates:
- The static grid geometry of dimension $W \times H$, partitioned into traversable floor tiles $\mathcal{W}$, impassable wall boundaries and interior obstacles $\mathcal{O}_{\text{wall}}$, and hidden POI coordinates $\mathcal{P} \subset \mathcal{W}$.
- The physical agent pose $p_t = (x_t, y_t, d_t) \in \mathcal{W} \times \{0, 1, 2, 3\}$, where orientation $d_t$ denotes East, South, West, or North.
- Dynamic threat states $m_t^{(j)} = (x_m^{(j)}, y_m^{(j)}, \text{mode}_j, \tau_j)$ for each monster $j \in \{1, \dots, M\}$, specifying position, behavioral state (patrol vs. pursuit), and movement counter.
- Global visitation and revelation ground-truth matrices $V_t, R_t \in \{0, 1\}^{W \times H}$.
- Global time step $t \in \{0, 1, \dots, T\}$.

### 2.2 Action Space $\mathcal{A}$
The agent operates over a minimal discrete action set:

$$\mathcal{A} = \{0: \text{Turn Left}, 1: \text{Turn Right}, 2: \text{Move Forward}\}$$

- Turning updates orientation deterministically: $d_{t+1} = (d_t \mp 1) \bmod 4$, keeping coordinates invariant.
- Moving forward attempts relocation along the orientation unit vector: $(x_t + \Delta x(d_t), y_t + \Delta y(d_t))$. If obstructed by $\mathcal{O}_{\text{wall}}$, the agent remains in place and records a collision event $\text{bump}_t = 1$.

Irrelevant primitive object interactions (such as picking up or toggling objects) are excluded, removing degenerate action choices and accelerating policy credit assignment.

### 2.3 Transition Dynamics $\mathcal{T}(s_{t+1} \mid s_t, a_t)$
Transitions combine deterministic agent motion with semi-stochastic, reactive threat dynamics:
- The agent moves deterministically according to $a_t$.
- Monsters advance periodically according to their internal timer. If the agent enters their unoccluded line-of-sight within detection radius $r_{\text{detect}}$, the monster pursues the agent along the shortest path. Otherwise, it follows a patrol walk.
- An encounter occurs if an agent and monster occupy the same tile or swap adjacent tiles during the transition step, triggering terminal failure ($\text{death}_t = 1$).

### 2.4 Observation Space $\Omega$ & Emission Function $\mathcal{O}(o_t \mid s_t)$
The agent perceives an egocentric projection $o_t = \mathcal{O}(s_t) = (\mathbf{v}_t, \mathbf{m}_t, \rho_t)$:
- $\mathbf{v}_t \in \{0, 1\}^{7 \times 7 \times 5}$: Local line-of-sight visual cone via raycasting.
- $\mathbf{m}_t \in \{0, 1\}^{K \times K \times 4}$: Egocentric crop of accumulated spatial memory ($K=15$).
- $\rho_t = t / T \in [0, 1]$: Normalized temporal budget ratio.

### 2.5 Reward Function $\mathcal{R}(s_t, a_t, s_{t+1})$
The policy is guided by an exploration reward function defined as:

$$R_t = w_{\text{reveal}} \cdot \Delta N_t + w_{\text{poi}} \cdot \Delta P_t - c_{\text{step}} - c_{\text{bump}} \cdot \mathbb{I}[\text{bump}_t] - c_{\text{loop}} \cdot \mathbb{I}[\text{loop}_t] - c_{\text{death}} \cdot \mathbb{I}[\text{death}_t] + b_{\text{done}} \cdot \left(1 - \frac{t}{T}\right) \cdot \mathbb{I}[\text{done}_t]$$

where:
- $\Delta N_t \in \mathbb{N}_{\ge 0}$: Number of newly revealed walkable floor tiles at time step $t$.
- $w_{\text{reveal}} \in \mathbb{R}^+$: Weight coefficient for newly revealed floor tiles.
- $\Delta P_t \in \mathbb{N}_{\ge 0}$: Number of newly discovered Points of Interest at time step $t$.
- $w_{\text{poi}} \in \mathbb{R}^+$: Weight coefficient for discovered POIs.
- $c_{\text{step}} \in \mathbb{R}^+$: Constant step penalty.
- $c_{\text{bump}} \in \mathbb{R}^+$: Penalty weight for wall collisions.
- $\text{bump}_t \in \{0, 1\}$: Binary flag indicating a forward collision into an obstacle at step $t$.
- $c_{\text{loop}} \in \mathbb{R}^+$: Penalty weight for repeated visited poses.
- $\text{loop}_t \in \{0, 1\}$: Binary flag indicating a repeated state loop at step $t$.
- $c_{\text{death}} \in \mathbb{R}^+$: Penalty weight for fatal threat encounters.
- $\text{death}_t \in \{0, 1\}$: Binary flag indicating a fatal monster collision at step $t$.
- $b_{\text{done}} \in \mathbb{R}^+$: Base reward weight for reaching the exploration target.
- $\text{done}_t \in \{0, 1\}$: Binary flag indicating achievement of the target coverage threshold ($\ge 95\%$).
- $t$: Current environment step index ($t \in \{0, 1, \dots, T\}$).
- $T$: Maximum allowable horizon for the episode.
- $\mathbb{I}[\cdot]$: Indicator function evaluating to 1 if the condition is true and 0 otherwise.

### 2.6 Optimization Objective
The reinforcement learning objective is to optimize a parameterized policy $\pi_\theta$ to maximize expected discounted cumulative return:

$$\max_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R_t \right]$$

where:
- $\theta \in \mathbb{R}^D$: Trainable parameter vector of the policy network.
- $\pi_\theta(a_t \mid o_t)$: Action probability distribution over discrete actions $\mathcal{A}$ given observation $o_t$.
- $\tau = (o_0, a_0, R_0, o_1, a_1, R_1, \dots, o_T)$: Episode trajectory sampled under policy $\pi_\theta$.
- $\gamma \in [0, 1)$: Temporal discount factor.
- $R_t \in \mathbb{R}$: Scalar reward received at time step $t$.
- $T \in \mathbb{N}^+$: Episode termination horizon.
- $J(\theta)$: Expected cumulative discounted return objective.
