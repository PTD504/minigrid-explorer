# Reward Engine Design & Anti-Reward Hacking Guarantees

## 1. Overview and Design Objectives

In exploration under sparse objectives, learning policies purely from binary task completion signals suffers from high sample complexity. However, unconstrained reward shaping frequently induces degenerate policy exploitation (reward hacking).

The reward function provides dense learning signals while enforcing formal mathematical invariants against policy collapse, point farming, and behavioral stagnation.

---

## 2. Mathematical Formulation

At step $t \in \{1, \dots, T\}$, the scalar reward $R_t$ is computed as:

$$R_t = w_{\text{reveal}} \cdot \Delta N_t + w_{\text{poi}} \cdot \Delta P_t - c_{\text{step}} - c_{\text{bump}} \cdot \mathbb{I}[\text{bump}_t] - c_{\text{loop}} \cdot \mathbb{I}[\text{loop}_t] - c_{\text{death}} \cdot \mathbb{I}[\text{death}_t] + b_{\text{done}} \cdot \left(1 - \frac{t}{T}\right) \cdot \mathbb{I}[\text{coverage\_done}_t]$$

### 2.1 Component Definitions and Parameter Values

| Symbol | Parameter | Value | Mathematical Definition and Purpose |
|---|---|:---:|---|
| $\Delta N_t$ | `w_reveal` | `+0.02` | Count of newly revealed floor cells in line-of-sight ($\|R_t \setminus R_{t-1}\|$). |
| $\Delta P_t$ | `w_poi` | `+1.00` | Count of newly discovered POIs within line-of-sight. |
| $c_{\text{step}}$ | `c_step` | `0.002` | Uniform per-step penalty encouraging temporal efficiency. |
| $\mathbb{I}[\text{bump}_t]$ | `c_bump` | `0.020` | Indicator function equal to 1 if agent moves forward into a wall. |
| $\mathbb{I}[\text{loop}_t]$ | `c_loop` | `0.010` | Indicator function equal to 1 if agent revisits a recent pose $(x, y, d)$ without acquiring new information. |
| $\mathbb{I}[\text{death}_t]$ | `c_death` | `2.000` | Terminal penalty upon collision with a dynamic monster. |
| $\mathbb{I}[\text{coverage\_done}_t]$ | `b_done` | `+1.00` | Bonus for achieving target coverage ($\ge 95\%$), scaled by remaining time $(1 - t/T)$. |
| $K_{\text{loop}}$ | `loop_window` | `20` | Memory window length for state-repetition tracking. |

---

## 3. Mathematical Anti-Hacking Invariants

### 3.1 Strict Anti-Suicide Invariant
- **Failure Mode:** If the cumulative step penalty over the maximum horizon exceeds the death penalty, the agent learns that committing immediate suicide into a monster is optimal to minimize negative returns:
  $$\sum_{t=1}^T (-c_{\text{step}}) = -c_{\text{step}} \cdot T < -c_{\text{death}} \implies \text{Suicide is optimal}$$
- **Invariant:** We enforce $c_{\text{death}} > c_{\text{step}} \cdot T$ across all difficulty tiers:
  - Easy ($T=140$): $0.002 \times 140 = 0.28 < 2.0$ (Safety margin: $7.1\times$).
  - Medium ($T=390$): $0.002 \times 390 = 0.78 < 2.0$ (Safety margin: $2.56\times$).
  - Hard ($T=750$): $0.002 \times 750 = 1.50 < 2.0$ (Safety margin: $1.33\times$).

Survival is strictly optimal over death under all conditions.

### 3.2 Telescoping Exploration Property
- **Failure Mode:** Rewarding the agent for perceiving open tiles can incentivize cyclic point farming (e.g., repeatedly turning back and forth or pacing between two open cells).
- **Invariant:** The term $\Delta N_t$ measures only newly observed floor cells:
  $$\Delta N_t = |R_t \setminus R_{t-1}|$$
- Because revelation sets are strictly monotonic, the cumulative exploration return over an entire episode telescopes:
  $$\sum_{t=1}^T \Delta N_t = |R_T \cap \mathcal{W}| \le |\mathcal{W}|$$
- Revisiting observed areas yields $\Delta N_t = 0$. Pacing or turning generates negative returns through step penalties.

### 3.3 State-Repetition Loop Suppression
- A circular history buffer tracks recent poses $(x_t, y_t, d_t)$ over a window $K_{\text{loop}} = 20$.
- If the current pose matches an entry in the buffer **AND** no novel information was acquired ($\Delta N_t = 0 \land \Delta P_t = 0$), the penalty $c_{\text{loop}} = 0.01$ is triggered.
- This suppresses infinite limit cycles in symmetric corridors and junction deadlocks.

### 3.4 Wall-Bump Regularization
- Agents operating under partial observability often scrape along walls to orient themselves.
- Applying $c_{\text{bump}} = 0.02$ penalizes invalid forward transitions into obstacles, forcing the policy to learn discrete turns prior to forward movement.

### 3.5 Scale Stationarity
- Rewards are computed without dividing by true map area $|\mathcal{W}|$ or total POI counts. This preserves reward scale stationarity across varying dungeon sizes ($17 \times 17$ to $33 \times 33$).
