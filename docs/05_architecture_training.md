# RL Algorithm Selection & Neural Policy Architecture

## 1. Algorithmic Paradigm Selection

Autonomous exploration in procedurally generated environments under partial observability and dynamic threats imposes distinct optimization challenges:
1. **Non-Markovian Observations:** In POMDPs, individual observations $o_t$ do not satisfy the Markov property ($P(o_{t+1} \mid o_t, a_t) \neq P(s_{t+1} \mid s_t, a_t)$).
2. **Severe Asymmetric Penalties:** Colliding with a monster yields a catastrophic penalty ($c_{\text{death}} = -2.0$), while individual exploration rewards are small and incremental ($w_{\text{reveal}} = +0.02$).
3. **Long Horizon Credit Assignment:** Exploration episodes span hundreds of steps ($T = 140 \dots 750$), requiring low-variance advantage estimation across lengthy trajectories.

---

### 1.1 Comparative Analysis of Candidate Paradigms

We analyzed five reinforcement learning paradigms against these constraints:

#### 1. Deep Q-Networks (DQN / Rainbow) — Off-Policy Value-Based
- **Mechanism:** Minimizes Temporal Difference (TD) Bellman error over an Experience Replay Buffer $\mathcal{D}$:
  $$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
- **Failure Modes:**
  - *Deadly Triad & Non-Markovian Stale Buffer:* Off-policy replay combines function approximation, bootstrapping, and an off-policy distribution (the Deadly Triad). In POMDPs, stored historical transitions contain stale monster trajectories and incomplete map fragments that do not reflect the current policy distribution, leading to severe state-aliasing bias and Q-value divergence.
  - *Threat Distribution Shift:* Historical death transitions where monsters cornered early policies disproportionately distort target Q-values, causing the agent to freeze in corners to avoid perceived universal threat.
  - *Slow Credit Propagation:* Standard 1-step or n-step TD updates propagate delayed exploration rewards backward slowly across long horizons ($T > 300$).
- **Verdict:** Rejected.

#### 2. Soft Actor-Critic (SAC) — Off-Policy Maximum Entropy
- **Mechanism:** Optimizes return concurrently with policy entropy:
  $$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t)} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t))]$$
- **Failure Modes:**
  - *Action Space Mismatch:* SAC is formulated for continuous action spaces. Adapting to discrete actions requires computing expectations over all $|A|$ discrete actions across double Q-critics, which introduces substantial gradient variance on compact grid navigation actions.
  - *Temperature Sensitivity:* Under severe terminal death penalties, auto-tuning the temperature parameter $\alpha$ frequently causes the entropy term to dominate or collapse, destabilizing exploration.
- **Verdict:** Rejected.

#### 3. Advantage Actor-Critic (A2C / A3C) — Synchronous On-Policy Policy Gradient
- **Mechanism:** Updates policy parameters along the unclipped policy gradient:
  $$\nabla_\theta J(\theta) = \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$$
- **Failure Mode:**
  - *Catastrophic Policy Collapse under Death Penalties:* When an agent encounters a monster, the resulting penalty ($c_{\text{death}} = -2.0$) produces a large negative advantage $\hat{A}_t \ll 0$. Because A2C lacks trust-region step bounding, a single batch containing fatal encounters can push action probabilities to near-zero, inducing irreversible policy collapse and entropy destruction.
- **Verdict:** Rejected.

#### 4. Proximal Policy Optimization (PPO) — Selected Primary Framework
- **Mechanism:** Constrains parameter updates within a trust region via a clipped surrogate objective:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
  where $r_t(\theta) = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)}$ and $\epsilon = 0.20$.
- **Why PPO Succeeds:**
  - *Gradient Bounding:* When catastrophic death events occur, clipping prevents outsized negative updates, ensuring policy parameters remain within a stable trust manifold.
  - *Generalized Advantage Estimation ($\text{GAE-}\lambda$):* Uses $\lambda = 0.95$ and $\gamma = 0.99$ to balance bias and variance across extended exploration horizons ($T = 140 \dots 750$).
  - *Explicit Entropy Regularization:* An explicit entropy coefficient ($\beta = 0.015$) prevents premature policy collapse into safe but stagnant local loops.
  - *High On-Policy Vectorized Throughput:* Synchronous parallel rollouts gather fresh transitions across distinct procedural layouts, avoiding buffer staleness entirely.
- **Verdict:** Selected as the primary optimization algorithm.

#### 5. RecurrentPPO (PPO + LSTM) — Ablation Benchmark
- **Mechanism:** Embeds an LSTM cell between the feature extractor and actor/critic heads, maintaining a temporal hidden state $h_t = \text{LSTM}(h_{t-1}, f(o_t))$.
- **Research Role:**
  - Serves as the primary comparative ablation to test whether an **explicit spatial memory crop** with 2D convolutional inductive bias outperforms an **implicit recurrent hidden state** trained via Backpropagation Through Time (BPTT).
- **Verdict:** Selected as the comparative ablation baseline.

---

## 2. Policy Neural Architecture

Standard vision backbones designed for Atari (such as `NatureCNN`) use large $8 \times 8$ kernels that cannot process compact $7 \times 7$ observations. Furthermore, standard networks flatten non-standard inputs, discarding 2D geometric structure.

We design a multi-modal feature extractor that processes visual perception, accumulated spatial memory, and temporal status through distinct dedicated pathways:

1. **Local View CNN Pathway:**
   Processes the $7 \times 7 \times 5$ local visual field. A $3 \times 3$ convolution with stride 1 (no pooling) outputs 32 channels of spatial dimension $5 \times 5$, flattened into an 800-dimensional latent vector. This preserves high-resolution local obstacle and entity layouts.
2. **Spatial Memory CNN Pathway:**
   Processes the $15 \times 15 \times 4$ egocentric spatial memory crop. Two cascaded $3 \times 3$ convolutions with stride 1 (channels $4 \to 32 \to 64$) without pooling expand the receptive field to $5 \times 5$. The resulting $11 \times 11 \times 64$ tensor is flattened into a 7,744-dimensional representation capturing multi-room connectivity. Setting memory usage to false ablates this entire branch.
3. **Temporal Modulation Pathway:**
   The normalized budget scalar $\rho_t = t / T$ bypasses convolutions and is concatenated directly into the latent vector, enabling time-conditioned policy behavior.
4. **Fusion Trunk & Output Heads:**
   The concatenated latent vector (dimension 8,545 with memory, or 801 without memory) projects through a dense linear layer to a 256-dimensional embedding with ReLU activation.
   - **Actor Head ($\pi_\theta$):** A 2-layer MLP ($256 \to 64 \to 64$) with Tanh activations outputs a 3-dimensional categorical distribution over $\mathcal{A}$.
   - **Critic Head ($V_\phi$):** A symmetric 2-layer MLP ($256 \to 64 \to 64$) with Tanh activations outputs a scalar baseline state-value estimate $V(o_t)$.

---

## 3. Training and Optimization Protocol

### 3.1 Optimization Parameters
- **Parallel Sampling:** 8 parallel environment workers collecting transitions across distinct procedural layouts simultaneously.
- **Rollout Horizon:** 128 environment steps per worker ($128 \times 8 = 1,024$ total transitions per iteration).
- **Mini-Batch Optimization:** Rollout transitions are partitioned into mini-batches of size 64 (16 gradient steps per epoch) over 4 optimization epochs per iteration.
- **Learning Rate Schedule:** Initial learning rate $\alpha_0 = 3 \times 10^{-4}$, decayed linearly to 0 across the training budget ($500,000$ steps).
- **Discount & GAE:** $\gamma = 0.99$, $\lambda = 0.95$.
- **Entropy & Clipping:** Entropy coefficient $\beta = 0.015$, clipping parameter $\epsilon = 0.20$, gradient norm bound $\|g\|_2 \le 0.50$.

### 3.2 Normalization Invariant
Running mean/variance observation normalization is intentionally omitted. Input channels are natively bounded binary masks $\in \{0, 1\}$, the temporal budget $\rho_t \in [0, 1]$ is bounded, and rewards are constrained by design invariants. Avoiding running normalization eliminates non-stationary distribution shift when reloading checkpoints across evaluation suites.

### 3.3 Stability Monitoring and Validation
Training convergence is monitored through four quantitative indicators:
- **Policy & Value Loss:** Verifying stable value error reduction without gradient spikes.
- **Entropy Decay:** Confirming smooth decay from uniform distribution ($\ln(3) \approx 1.098$) to a focused policy ($\approx 0.35 - 0.50$).
- **Approximate KL Divergence:** Verifying that updates respect trust-region bounds ($\text{approx\_kl} \le 0.03$).
- **Explained Variance:** Confirming the critic accurately predicts returns ($\text{explained\_var} > 0.70$ during late training).
- **Validation Protocol:** Checkpoints are evaluated every 10,000 steps across 20 fixed validation seeds. The model achieving the highest validation return is saved as the evaluation checkpoint.