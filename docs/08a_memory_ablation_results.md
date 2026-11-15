# Causal Memory Channel Ablation Results

## 1. Overview and Experimental Protocol

While standard training-time ablations (comparing **PPO Spatial Memory** against **PPO No-Memory**) measure aggregate performance differences, they do not reveal *which* specific spatial representations a trained memory-augmented policy actually relies on at inference time.

To isolate the causal contribution of each egocentric memory channel without retraining, we performed a **post-hoc channel-blanking evaluation** on the converged Medium-trained ensemble (`ppo_medium_5seeds_500k`, 5 independent training seeds). During evaluation across the 100 held-out Medium test seeds (`2,000,000` to `2,000,099`, $25 \times 25$, $T=390$, $\tau=2$, `deterministic=True`), selected channels of the $15 \times 15 \times 4$ egocentric memory crop (`obs["memory"]`) were zeroed out immediately prior to action selection:

1. **Channel 0 (`MEM_REVEALED`):** Binary mask of tiles previously observed within line-of-sight.
2. **Channel 1 (`MEM_VISITED`):** Binary mask of tiles physically traversed by the agent.
3. **Channel 2 (`MEM_KNOWN_WALL`):** Binary mask of discovered structural walls.
4. **Channel 3 (`MEM_KNOWN_POI`):** Binary mask of discovered Points of Interest.
5. **All Channels (`0, 1, 2, 3`):** Complete suppression of the explicit spatial memory input (forcing the dual-stream CNN to operate purely on the $7 \times 7$ local view and `step_ratio`, plus constant zero-input bias from `memory_cnn`).

---

## 2. Held-Out Test Set Evaluation (Medium Suite, 100 Seeds × 5 Runs)

The table below reports mean performance across the 5 training seeds ($\pm 1\sigma$ across seeds) for each evaluation-time blanking configuration:

| Memory Configuration | Blanked Channel(s) | Revealed Coverage (%) | $\Delta$ Cov | POI Discovery (%) | Success Rate ($\ge 95\%$) | Death Rate (%) | Steps | Mean Return | Wall Bump (%) | Loop Rate (%) | Revealed Eff (tiles/step) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Memory (Baseline)** | None | **$71.83\% \pm 2.59\%$** | $0.00\%$ | **$49.5\%$** | **$33.6\%$** | $27.4\%$ | $221.8$ | **$+1.573$** | $1.68\%$ | $67.44\%$ | $1.221$ |
| **Blank Ch 3 (`MEM_KNOWN_POI`)** | `3` | $70.08\% \pm 1.77\%$ | $-1.75\%$ | $48.9\%$ | $31.8\%$ | $31.4\%$ | $213.9$ | $+1.473$ | $1.44\%$ | $67.22\%$ | **$1.222$** |
| **Blank Ch 1 (`MEM_VISITED`)** | `1` | $65.07\% \pm 3.22\%$ | $-6.76\%$ | $43.0\%$ | $23.2\%$ | $34.2\%$ | $236.0$ | $+0.701$ | $1.23\%$ | $71.32\%$ | $1.039$ |
| **Blank Ch 0 (`MEM_REVEALED`)** | `0` | $52.34\% \pm 14.72\%$ | $-19.49\%$ | $32.7\%$ | $10.4\%$ | $42.2\%$ | $242.2$ | $-0.617$ | **$9.26\%$** | $69.92\%$ | $0.748$ |
| **Blank Ch 2 (`MEM_KNOWN_WALL`)** | `2` | **$35.42\% \pm 8.57\%$** | **$-36.41\%$** | **$17.8\%$** | **$1.4\%$** | **$49.6\%$** | $241.9$ | **$-2.068$** | $2.51\%$ | **$83.54\%$** | **$0.580$** |
| **Blank All Memory** | `0, 1, 2, 3` | $61.33\% \pm 6.12\%$ | $-10.50\%$ | $40.0\%$ | $12.2\%$ | **$23.8\%$** | $310.5$ | $+0.302$ | **$0.20\%$** | **$59.88\%$** | $0.558$ |

---

## 3. Causal Attribution & Key Findings

The empirical hierarchy of channel importance is strictly ordered:
$$\text{MEM\_KNOWN\_WALL (Ch 2)} \gg \text{MEM\_REVEALED (Ch 0)} > \text{MEM\_VISITED (Ch 1)} \gg \text{MEM\_KNOWN\_POI (Ch 3)}$$

### 3.1 Structural Wall Memory (`MEM_KNOWN_WALL`) is the Primary Navigational Backbone
Zeroing out Channel 2 triggers a catastrophic collapse across every performance dimension:
* Revealed coverage drops by **$-36.41$ percentage points** ($71.83\% \to 35.42\%$), and episode completion rate nearly vanishes ($33.6\% \to 1.4\%$).
* Loop rate surges to **$83.54\%$** (up from $67.44\%$), while mortality nearly doubles from $27.4\%$ to **$49.6\%$**.
* This confirms that the spatial memory policy relies predominantly on accumulated wall geometry to orient itself across rooms and corridors beyond the $7 \times 7$ field of view.

### 3.2 Frontier Boundary Tracking (`MEM_REVEALED`) Drives Exploration Progress
Removing Channel 0 reduces revealed coverage by **$-19.49$ percentage points** ($71.83\% \to 52.34\%$) and induces severe cross-seed instability ($\pm 14.72\%$ standard deviation). Simultaneously, the wall bump rate jumps **$5.5\times$** ($1.68\% \to 9.26\%$), indicating that without explicit knowledge of which regions have already been revealed, the agent loses frontier directionality and collides repeatedly while probing boundaries.

### 3.3 Visited Trajectory (`MEM_VISITED`) Provides Moderate Anti-Looping Support
Blanking Channel 1 produces a moderate degradation ($-6.76\%$ coverage, $-10.4\%$ success rate, and $+3.88\%$ loop rate), demonstrating that the agent uses its historical path footprint to avoid redundant traversal, though it can partially compensate using `MEM_REVEALED` and `MEM_KNOWN_WALL`.

### 3.4 Stored POI Memory (`MEM_KNOWN_POI`) is Virtually Unused
Zeroing out Channel 3 yields virtually identical behavior to the unablated baseline ($70.08\%$ vs. $71.83\%$ coverage; $1.222$ vs. $1.221$ tiles/step revealed efficiency; $67.22\%$ vs. $67.44\%$ loop rate). Because POI discovery rewards are granted *immediately* upon entering the agent's $7 \times 7$ line-of-sight, persisting discovered POIs in the $15 \times 15$ spatial memory provides no actionable navigation signal once seen.

### 3.5 Why Blanking All Channels ($61.33\%$) Outperforms Blanking Walls Alone ($35.42\%$)
A notable non-monotonic effect emerges when comparing single-channel vs. full-memory blanking:
* **Full Blanking (`0, 1, 2, 3`):** When all memory channels are zeroed, the `memory_cnn` branch outputs a constant zero-input bias, effectively reverting the policy to a conservative, view-driven fallback ($61.33\%$ coverage, $310.5$ steps, $0.20\%$ bump rate, $23.8\%$ death rate, $0.558$ tiles/step).
* **Partial Structural Blanking (`2` or `0` alone):** When `MEM_KNOWN_WALL` (or `MEM_REVEALED`) is zeroed while the remaining channels remain active, the resulting feature map presents a **contradictory spatial state** (e.g., revealed floor tiles with no bounding walls). This out-of-distribution channel mismatch induces severe spatial hallucination in the convolutional encoder, trapping the agent in high-loop ($83.54\%$) or high-bump ($9.26\%$) failure modes.

