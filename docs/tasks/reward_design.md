# Task Specification: Reward Engine & Anti-Reward Hacking

## 1. Executive Summary

This document specifies the mathematical formulation, algorithmic implementation, anti-hacking invariants, and unit testing requirements for the **Reward Engine** (`explorer/rewards.py` or Gymnasium reward integration).

The reward function serves strictly as a **learning proxy** to guide the policy toward the primary mission goals:
1. **Maximizing revealed walkable area coverage** ($\text{revealed\_coverage}$).
2. **Discovering Points of Interest (POIs)**.
3. **Ensuring survival** under dynamic threats (Monsters).
4. **Navigating efficiently** without redundant looping or wall bumping.

---

## 2. Mathematical Definition

At each environment step $t \in \{1, \dots, T\}$, where $T = \text{max\_steps}$, the scalar reward $R_t$ is computed as:

$$
R_t = w_{\text{reveal}} \cdot \Delta N_t + w_{\text{poi}} \cdot \Delta P_t - c_{\text{step}} - c_{\text{bump}} \cdot \mathbb{I}[\text{bump}_t] - c_{\text{loop}} \cdot \mathbb{I}[\text{loop}_t] - c_{\text{death}} \cdot \mathbb{I}[\text{death}_t] + b_{\text{done}} \cdot \left(1 - \frac{t}{T}\right) \cdot \mathbb{I}[\text{coverage\_done}_t]
$$

### Component Breakdown

| Symbol | Parameter | Default Value | Description |
|---|---|:---:|---|
| $\Delta N_t$ | `w_reveal` | `0.02` | Number of **newly revealed walkable tiles** in step $t$ (monotonic telescoping sum). |
| $\Delta P_t$ | `w_poi` | `1.0` | Number of **newly discovered POIs** in line-of-sight in step $t$. |
| $c_{\text{step}}$ | `c_step` | `0.002` | Baseline step penalty to encourage temporal efficiency (max cumulative $\approx 0.5$ on Easy, $0.9$ on Hard). |
| $\mathbb{I}[\text{bump}_t]$ | `c_bump` | `0.02` | Indicator for hitting a wall / grid boundary when taking `forward`. |
| $\mathbb{I}[\text{loop}_t]$ | `c_loop` | `0.01` | Indicator for repeating a recent state $(x, y, \text{dir})$ within `loop_window` without gaining new information ($\Delta N_t = 0 \land \Delta P_t = 0$). |
| $\mathbb{I}[\text{death}_t]$ | `c_death` | `2.0` | Severe terminal penalty when colliding with a monster (same tile or position swap). |
| $\mathbb{I}[\text{coverage\_done}_t]$ | `b_done` | `1.0` | Early completion bonus when reaching $\text{revealed\_coverage} \ge \tau$ ($\tau = 0.95$), scaled by remaining time $(1 - t/T)$. |
| `loop_window` | `loop_window` | `20` | Memory window length for state-repetition tracking. |

All hyperparameters are centralized in `configs/reward/default.yaml` and loaded dynamically via `get_reward_config()`.

---

## 3. Anti-Reward Hacking Guarantees

### 3.1 Strict Anti-Suicide Invariant
- **Vulnerability:** If the cumulative step penalty over the maximum episode length exceeds the penalty for dying ($c_{\text{step}} \cdot T \ge c_{\text{death}}$), the agent learns that committing suicide into the nearest monster is optimal to minimize negative rewards.
- **Guarantee:** We enforce the invariant:
  $$c_{\text{death}} > c_{\text{step}} \cdot T, \quad \forall \text{ difficulty} \in \{\text{easy}, \text{medium}, \text{hard}\}$$
  - Easy ($T=250$): $0.002 \times 250 = 0.50 < 2.0$ (Invariant holds by $4\times$ margin).
  - Medium ($T=350$): $0.002 \times 350 = 0.70 < 2.0$ (Invariant holds by $2.85\times$ margin).
  - Hard ($T=450$): $0.002 \times 450 = 0.90 < 2.0$ (Invariant holds by $2.22\times$ margin).

### 3.2 Telescoping Exploration Property
- $\Delta N_t$ measures strictly newly observed walkable tiles ($\Delta N_t = |R_t \setminus R_{t-1}|$).
- The cumulative sum of exploration rewards over an entire episode satisfies:
  $$\sum_{t=1}^T \Delta N_t = |R_T \cap W| \le |W|$$
- **Immunity:** The agent cannot farm rewards by oscillating between two adjacent cells or repeatedly turning in circles.

### 3.3 State-Repetition Loop Suppression
- A circular buffer of size `loop_window = 20` stores historical state tuples: $(x, y, \text{dir})$.
- If the current $(x_t, y_t, \text{dir}_t)$ exists in the window **AND** $\Delta N_t = 0$ **AND** $\Delta P_t = 0$, the penalty $c_{\text{loop}}$ is applied.
- This prevents the policy from getting trapped in dead-end cycles or turning loops when exploration stalls.

### 3.4 No Privileged Information Leakage
- $\Delta N_t$ and $\Delta P_t$ are derived solely from the agent's observable field of view (`vis_mask`).
- The reward does **NOT** normalize by $|W|$ or total POIs, preventing implicit leakage of true map size.