# Failure Case Analysis & Diagnostics

## 1. Overview

While the reinforcement learning agents in [`08_results.md`](08_results.md) and [`09_generalization.md`](09_generalization.md) explore much better and survive longer than classical baselines, they still fail in certain situations.

Even the best-performing model—`PPO (No Memory)`—does not succeed $100\%$ of the time:
- **Easy Map ($17 \times 17$, 1 monster):** $4.6\%$ death rate, $22.2\%$ of runs end before reaching $95\%$ coverage.
- **Medium Map ($25 \times 25$, 1 faster monster):** $9.2\%$ death rate, $49.0\%$ of runs end before reaching $95\%$ coverage.
- **Hard Map ($33 \times 33$, 2 monsters):** $45.2\%$ death rate when trained on Medium, and $62.8\%$ death rate when trained on Easy.

Understanding why and when the agent fails is essential to diagnose weaknesses and know what to expect when using these models as automated QA test bots.

---

## 2. Common Failure Scenarios

### 2.1 Trapped in Narrow 1-Tile Corridors

- **Situation:** A corridor that is only 1 tile wide connects two rooms, with no side paths or alternative loops.
- **How Failure Happens:** The agent enters the narrow corridor heading toward an unvisited room. At the same time, a patrolling monster enters the opposite end of the hallway. Once the monster sees the agent, it switches to pursuit mode. Because the corridor is only 1 cell wide, the agent cannot dodge or step aside.
- **Differences Between Models:**
  - *Frontier BFS Baseline:* Fails almost every time (**$77.0\%$ deaths on Medium, $94.0\%$ on Hard**). Because Frontier BFS only aims for the nearest unvisited tile and has no monster awareness, it walks straight into the monster.
  - *PPO with Spatial Memory:* Experiences high death rates (**$27.4\%$ on Medium, $67.8\%$ on Hard**). The memory window still shows unexplored tiles right behind the monster, creating conflicting incentives: the agent wants to explore forward, but also wants to run away. This hesitation gives the monster time to catch up.
  - *PPO without Memory:* Performs the best (**$9.2\%$ deaths on Medium**). Relying only on its immediate $7 \times 7$ view, it immediately turns around $180^\circ$ and runs back toward the open room. It only dies if the monster was already too close for it to reach the room in time.

---

### 2.2 Getting Trapped Between Two Monsters (Hard Map)

- **Situation:** The Hard map contains two active monsters roaming across 8–12 rooms.
- **How Failure Happens:** The agent spots Monster 1 ahead and successfully runs away into a side hallway or adjacent room. However, because the agent can only see within its immediate $7 \times 7$ field of view, it cannot know what is around the next corner. Turning the corner to escape Monster 1 often leads it directly into the path of Monster 2.
- **Key Numbers:**
  - This two-monster trap accounts for **$72.4\%$ of all deaths** on the Hard test set.
  - Training on Medium maps (with a faster monster) significantly improves survival: death rate drops from **$62.8\%$** (Easy-trained) down to **$45.2\%$** (Medium-trained). Experiencing faster threats helps the agent learn to retreat sooner and stay closer to room exits.

---

### 2.3 Getting Stuck in Place (Corridor Oscillation)

- **Situation:** Long hallways where walls look identical in both directions.
- **How Failure Happens:** Without memory of past steps, turning left and turning right inside a plain hallway produces almost identical visual observations. The agent can get stuck in a repetitive loop: turning left, turning right, then turning left again without moving forward.
- **Evidence from Reward Ablation Experiments:**
  - On Easy maps, removing the loop penalty ($c_{\text{loop}} = 0$) caused the loop rate to jump from **$22.92\%$ to $30.58\%$** (a **$+33.4\%$ increase**).
  - On Hard maps, removing penalties caused the loop rate to rise to **$30.62\%$**, increasing the average steps per episode from $368.1$ to **$422.2$ steps** while lowering total reward.
  - The loop penalty ($c_{\text{loop}} = 0.010$ over a 20-step window) effectively prevents this behavior by penalizing the agent when it visits the exact same position and orientation repeatedly.

---

### 2.4 Getting Stuck Along Room Borders (Spatial Memory)

- **Vulnerable Model:** `PPO (Spatial Memory)`.
- **How Failure Happens:** The $15 \times 15$ spatial memory map shows uncovered areas along room perimeters. When the agent is near a wall, unrevealed fringe tiles remain visible at the edge of the memory window. This tempts the policy to pace back and forth along the same wall rather than moving through doorways into new rooms.
- **Key Numbers:**
  - On Medium maps, `PPO (Spatial Memory)` had a **$57.46\%$ Loop Rate**, compared to only **$32.41\%$** for `PPO (No Memory)` and **$17.94\%$** for `Recurrent PPO`.
  - On Hard maps, its Loop Rate climbed further to **$66.14\%$**.
  - This explains why Spatial Memory had lower overall coverage than No-Memory across all test suites ($71.83\%$ vs. $81.79\%$ on Medium; $57.67\%$ vs. $68.40\%$ on Hard).

---

### 2.5 Running Out of Time in Dead Ends

- **Situation:** Blind alleys and dead-end spurs of length 2–3 tiles.
- **How Failure Happens:**
  1. The agent enters a dead end to uncover hidden tiles at the tip, earning exploration rewards.
  2. Once it reaches the end, all tiles in that spur are already revealed.
  3. The agent must turn around and walk back through explored floor space, where reward is zero.
  4. With strict step limits ($T=140$ on Easy, $T=390$ on Medium, $T=740$ on Hard), exploring multiple dead ends late in an episode burns too many steps. The episode ends on timeout before the agent can reach remaining unvisited rooms.
- **Result:** Episodes ending on timeout usually finish with $75\% - 85\%$ coverage rather than the desired $>95\%$ complete exploration.

---

## 3. Comparison of Failure Rates Across Models

The table below summarizes the failure patterns of all evaluated agents across the 100 test seeds, using consistent criteria for death risks, looping behavior, and timeout probabilities:

| Model / Agent | Main Cause of Failure | Narrow Corridor Risk | Two-Monster Risk (Hard) | Looping Risk (Loop Rate) | Timeout Risk (Runs Out of Time) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline: Random** | Wandering into walls | Fatal ($100\%$ death) | Fatal ($100\%$ death) | High ($45\% - 48\%$) | Negligible (dies in $36 - 52$ steps) |
| **Baseline: Frontier BFS** | Walking directly into threats | Extreme ($77\% - 94\%$ death)| Extreme ($94\%$ death) | Low ($8.5\% - 9.2\%$) | Negligible (dies in $51 - 70$ steps) |
| **Baseline: Wall Follower** | Looping along room perimeters | Fatal ($92\% - 99\%$ death) | Fatal ($99\%$ death) | Moderate ($17\% - 29\%$) | Negligible (dies in $106 - 116$ steps) |
| **Baseline: Oracle Planner** | Pathfinding without evasion | High ($49\% - 71\%$ death) | High ($71\%$ death) | Very Low ($5.5\% - 6.3\%$) | Negligible (dies in $61 - 91$ steps) |
| **PPO (Spatial Memory)** | Border hesitation and memory loops| High ($27\% - 68\%$ death) | High ($68\%$ death) | Very High ($57\% - 66\%$) | High (delayed by loops, $57\% - 75\%$ cov) |
| **PPO (No Memory)** | Blind corner surprises | Low ($4.6\% - 21\%$ death) | Moderate ($45.2\%$ death) | Moderate ($25\% - 32\%$) | Moderate (survives long, occasionally times out) |
| **Recurrent PPO (LSTM)** | Blind corner surprises | Low ($9.2\% - 29\%$ death) | High ($69.0\%$ death) | Low ($12.9\% - 17.9\%$) | Moderate (survives long, occasionally times out) |

---

## 4. Key Takeaways for QA Testing

1. **Blind Corners Remain the Biggest Danger:** Reactive agents are effective at dodging monsters they can see, but cannot anticipate monsters lurking just around a blind corner.
2. **Explicit Memory Grids Can Cause Unwanted Looping:** Giving the agent an explicit 2D memory crop does not automatically make it smarter; it often causes the agent to get stuck pacing along walls.
3. **Anti-Loop Penalties Are Essential:** Without a penalty for revisiting the same spots, automated test bots can waste time oscillating back and forth in plain hallways instead of testing game mechanics.
