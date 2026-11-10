# Evaluation Methodology & Measurement Protocols

## 1. Overview and Evaluation Objectives

To ensure that reported performance reflects genuine spatial intelligence and threat management rather than seed memorization or statistical noise, the evaluation framework enforces a rigorous, reproducible measurement protocol.

The protocol governs seed isolation, checkpoint selection, mathematical metric formulations, information boundaries, and statistical aggregation.

---

## 2. Seed Isolation Protocol

The procedural generation engine generates distinct layout topologies, POI locations, and monster routes parameterized by integer seeds. To guarantee zero data contamination between training, tuning, and evaluation, seeds are partitioned into non-overlapping numeric intervals:

1. **Training Split:** Seeds `0` to `99,999` (100,000 procedural variants). Environment instances draw uniformly from this pool during training rollouts.
2. **Validation Split:** Seeds `1,000,000` to `1,000,049` (50 fixed seeds). Used strictly for periodic checkpoint evaluation during training.
3. **Held-Out Test Split:** Seeds `2,000,000` to `2,000,099` (100 strictly held-out seeds). Reserved exclusively for final benchmark evaluation. No policy ever encounters these layouts during optimization.

---

## 3. Checkpoint Selection Protocol

Rather than selecting the final model checkpoint at the end of training (which may suffer from late-training policy degradation or oscillation), checkpoints are selected via validation performance:
- Policies are evaluated every 10,000 environment steps across the 20 initial validation seeds.
- The model achieving the highest mean evaluation return is saved atomically as `best_model.zip`.
- This checkpoint is locked and utilized for all downstream held-out test evaluations.

---

## 4. Mathematical Metric Definitions

The evaluation suite tracks eleven formal metrics:

### 4.1 Revealed Area Coverage ($\mathcal{C}_{\text{rev}}$)
The proportion of traversable floor tiles unveiled by line-of-sight raycasting by episode termination step $T_{\text{end}}$:
$$\mathcal{C}_{\text{rev}} = \frac{|\mathcal{R}_{T_{\text{end}}} \cap \mathcal{W}|}{|\mathcal{W}|} \times 100\%$$
where $\mathcal{R}_t$ is the set of revealed coordinates and $\mathcal{W}$ is the ground-truth set of all walkable floor cells.

### 4.2 Visited-Cell Coverage ($\mathcal{C}_{\text{vis}}$)
The proportion of traversable floor tiles physically traversed by the agent:
$$\mathcal{C}_{\text{vis}} = \frac{|\mathcal{V}_{T_{\text{end}}} \cap \mathcal{W}|}{|\mathcal{W}|} \times 100\%$$
where $\mathcal{V}_t$ is the set of physical coordinates $(x_\tau, y_\tau)$ occupied by the agent for $\tau \le t$.

### 4.3 Point-of-Interest Discovery Rate ($\mathcal{D}_{\text{poi}}$)
The percentage of hidden Points of Interest brought within line-of-sight:
$$\mathcal{D}_{\text{poi}} = \frac{|\mathcal{P}_{\text{discovered}}|}{|\mathcal{P}_{\text{total}}|} \times 100\%$$

### 4.4 Survival and Episode Outcome Rates
Episodes terminate under one of three mutually exclusive conditions:
- **Completion Rate:** Percentage of episodes reaching the target coverage threshold ($\mathcal{C}_{\text{rev}} \ge 95\%$) prior to timeout or death.
- **Death Rate:** Percentage of episodes terminating due to lethal collision with a dynamic monster.
- **Timeout Rate:** Percentage of episodes reaching the maximum horizon limit ($T$) without achieving 95% coverage or dying.

### 4.5 Threat Interaction Metrics
- **Monster Encounter Rate:** Percentage of total episode steps spent in close proximity to a dynamic adversary:
  $$\text{Encounter Rate} = \frac{1}{T_{\text{total}}} \sum_{t=1}^{T_{\text{total}}} \mathbb{I}[\|p_t - m_t\|_1 \le 1] \times 100\%$$

### 4.6 Wall Collision and Invalid Action Regularization
- **Wall Bump Rate:** Percentage of actions attempting forward movement into wall obstacles or outer grid boundaries:
  $$\text{Bump Rate} = \frac{1}{T_{\text{total}}} \sum_{t=1}^{T_{\text{total}}} \mathbb{I}[\text{bump}_t] \times 100\%$$

### 4.7 Movement Efficiency ($\eta_{\text{rev}}$)
The average rate of new floor area revelation per action executed:
$$\eta_{\text{rev}} = \frac{|\mathcal{R}_{T_{\text{end}}} \cap \mathcal{W}|}{T_{\text{end}}} \quad (\text{tiles revealed / step})$$

### 4.8 Repeated-State and Loop Frequency
- **Loop Rate:** Percentage of steps where the agent revisits a recent pose $(x, y, d)$ within a temporal window of 20 steps without acquiring new information ($\Delta N_t = 0 \land \Delta P_t = 0$):
  $$\text{Loop Rate} = \frac{1}{T_{\text{total}}} \sum_{t=1}^{T_{\text{total}}} \mathbb{I}[\text{loop}_t] \times 100\%$$

### 4.9 Temporal Coverage Milestones
Mean revealed coverage achieved at quarter-intervals of the maximum step budget ($25\% T$, $50\% T$, $75\% T$, and $100\% T$). This measures early-horizon exploration velocity versus late-horizon stagnation.

### 4.10 Cross-Difficulty Generalization
Quantified by evaluating policies trained strictly on one difficulty tier (e.g., Easy $17 \times 17$) directly on unseen higher difficulty tiers (Medium $25 \times 25$, Hard $33 \times 33$) without fine-tuning or retraining.

### 4.11 Inference Latency and Computational Throughput
- **Latency:** Average wall-clock time required for a single forward policy inference pass, measured in milliseconds per step ($\text{ms/step}$).
- **Throughput:** Total environment steps executed per wall-clock second ($\text{FPS}$).

---

## 5. Information Boundary Verification

During evaluation, policies operate under strict information boundaries:
- The evaluator computes ground-truth metrics (e.g., $|\mathcal{W}|$, total POIs, monster coordinates) externally in diagnostic telemetry (`info` dictionary).
- The policy receives exclusively the partial observation tuple $o_t = (\mathbf{v}_t, \mathbf{m}_t, \rho_t)$.
- Zero privileged metrics or global coordinates are fed back into policy inputs.

---

## 6. Statistical Aggregation Protocol

To eliminate single-seed bias:
1. Every model configuration is trained across **5 independent random training seeds** (yielding 5 distinct model checkpoints).
2. Each checkpoint is evaluated across the complete set of **100 held-out test seeds** (`2,000,000` to `2,000,099`).
3. For each model family, results represent the ensemble mean $\pm$ sample standard deviation across the 5 independent training runs:
   $$\bar{\mu} = \frac{1}{K} \sum_{k=1}^K \mu_k, \quad \bar{\sigma} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (\mu_k - \bar{\mu})^2}, \quad K = 5$$
   representing 500 evaluation episodes per policy architecture.

