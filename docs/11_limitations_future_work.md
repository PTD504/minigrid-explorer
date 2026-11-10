# Known Limitations & Future Directions

## 1. Known Limitations

While the evaluated reinforcement learning framework achieves strong empirical performance across procedural benchmarks, several clear limitations remain in the current implementation and experimental setup:

### 1.1 Limited Training Step Budget & Lack of Dedicated Hard Training
- **Brief Horizon on Medium Maps:** The Medium models were trained for only $500\text{k}$ environment steps. Given that Medium maps ($25 \times 25$) feature more than double the floor area of Easy maps and a $50\%$ faster monster ($\tau=2$), 500k steps is relatively short to reach full policy convergence.
- **No Direct Training on Hard Maps:** Due to computational budget constraints, no models were trained directly on the Hard suite ($33 \times 33$, 8–12 rooms, 2 simultaneous monsters). We only evaluated zero-shot transfer from Easy-trained and Medium-trained checkpoints onto Hard. While Medium-trained models achieved $68.4\%$ coverage zero-shot, policies trained end-to-end on Hard maps could develop specialized coordinated evasion strategies against dual threats.

### 1.2 Limited Number of Random Training Seeds (5 Seeds)
- Each model configuration was trained across 5 independent random seeds. While 5 seeds is standard for preliminary benchmarks, deep reinforcement learning in procedurally generated environments exhibits substantial run-to-run variance.
- A sample size of 5 seeds is insufficient to compute tight statistical confidence intervals (such as Interquartile Mean with $95\%$ bootstrap intervals) or to definitively rule out rare policy failure outliers.

### 1.3 Fixed Receptive Field of Egocentric Memory Crops
- The spatial memory crop operates over a fixed window size ($15 \times 15$). While this covers most of the layout on Easy ($17 \times 17$) and Medium ($25 \times 25$) maps, it covers less than $21\%$ of the total area on Hard maps ($33 \times 33$).
- The policy remains blind to global loops and distant room connections outside this $15 \times 15$ window, preventing long-horizon spatial planning.

### 1.4 Strict Line-of-Sight Sensory Boundary (Blind Corners)
- The agent detects dynamic monsters only when an unobstructed visual ray connects the agent to the monster.
- In layouts with blind corners and L-shaped corridor junctions, the agent receives zero advance warning of approaching threats until stepping directly into an adjacent cell. This sensory boundary is the primary cause of the elevated mortality observed in Hard environments ($45.2\% - 69.0\%$).

### 1.5 Binary Memory Representation Without Time Decay
- The spatial memory grid uses binary indicator channels (Revealed, Visited, Known Obstacles, Known POIs).
- The channels do not track temporal recency (i.e. how many steps ago a cell was observed). If a monster crosses a cell and departs, the memory system cannot convey whether the threat is recent or long gone.

---

## 2. Promising Future Directions

### 2.1 Scaling Compute Budget & Direct Hard-Map Training
- **Extended Training Budgets:** Scale training on Medium environments to $>1\text{M}$ or $2\text{M}$ steps to allow policies to fully optimize path efficiency and threat evasion.
- **End-to-End Hard Map Training:** Allocate $2\text{M} - 5\text{M}$ steps to train policies directly on the Hard benchmark ($33 \times 33$, 2 fast monsters, $T=740$). This will allow the actor-critic network to learn complex multi-agent evasion loops that cannot be discovered from smaller environments alone.

### 2.2 Expanding Seed Ensembles & Rigorous Statistical Evaluation
- Increase the number of training runs to $10 - 20$ independent seeds per model family.
- Incorporate standardized statistical benchmarking tools (such as `rliable`) to report Interquartile Mean (IQM), Performance Profiles, and stratified bootstrap confidence intervals, ensuring publication-grade statistical significance.

### 2.3 Automated Domain Randomization (ADR) & Curriculum Learning
- Generalization experiments showed that training on Medium maps reduced Hard mortality by $17.6\%$ compared to Easy-trained models.
- Future work should formalize this into an Automated Domain Randomization (ADR) curriculum that smoothly scales grid dimensions ($17 \to 25 \to 33$), corridor tortuosity, room counts, and monster speeds ($\tau=3 \to 2$) as the agent's rolling coverage surpasses predefined thresholds.

### 2.4 Multi-Modal Sensing: Indirect / Acoustic Threat Perception
- Augment the agent's observation space with simulated acoustic or vibration sensors (e.g. a 1D scalar or direction vector indicating monster footsteps through walls, attenuated by distance).
- This would allow the policy to detect approaching adversaries before turning blind corners, directly mitigating narrow-corridor ambush traps.

### 2.5 Risk-Sensitive & Distributional RL
- Standard PPO optimizes the expected cumulative return $\mathbb{E}[R]$. Under catastrophic death penalties, expected-value optimization can make the policy overly optimistic near room exits.
- Applying Distributional RL (e.g. Quantile Regression PPO) or Conditional Value-at-Risk (CVaR) objectives would encourage risk-averse navigation policies that prioritize safe retreat paths whenever predictive variance indicates potential entrapment.
