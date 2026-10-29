import sys

import gymnasium as gym

env_id = sys.argv[1] if len(sys.argv) > 1 else "MiniGrid-FourRooms-v0"
env = gym.make(env_id)
obs, info = env.reset(seed=0)

print("observation_space:", env.observation_space)
print("action_space:", env.action_space)
print("image shape:", obs["image"].shape, "| direction:", obs["direction"])
print("channel 0 (object id) 7x7:\n", obs["image"][..., 0].T)
u = env.unwrapped
print("grid size:", u.width, u.height, "| max_steps:", u.max_steps)

for _ in range(20):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        break
print("after 20 random steps: reward =", r, "| step_count =", u.step_count)