"""
Navigation.py — Training script for the Banana Collector Navigation project.

Run from the project root directory:
    python Navigation.py

The Unity environment must be downloaded and placed in the project root.
See README.md for download links and setup instructions.
"""

import os
import sys
import platform
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display required)
import matplotlib.pyplot as plt
from collections import deque

from dqn_agent import Agent

# ── 1. Locate the Unity environment for the current OS ───────────────────────
def find_env():
    """Return the Unity environment binary path for the current platform."""
    system = platform.system()
    candidates = {
        "Darwin": ["Banana.app"],                              # macOS
        "Linux":  [
            "Banana_Linux/Banana.x86_64",
            "Banana_Linux/Banana.x86",
        ],
    }
    paths = candidates.get(system, [])
    for p in paths:
        if os.path.exists(p):
            return p
    print(
        "ERROR: Could not find the Unity environment binary.\n"
        "Please download it and place it in the project root.\n"
        "See README.md for download links."
    )
    sys.exit(1)

if __name__ == "__main__":
    from unityagents import UnityEnvironment
    env_path = find_env()
    print(f"Loading environment from: {env_path}")

    env        = UnityEnvironment(file_name=env_path)
    brain_name = env.brain_names[0]
    brain      = env.brains[brain_name]

    # ── 2. Inspect the environment ────────────────────────────────────────────────
    env_info    = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state       = env_info.vector_observations[0]
    state_size  = len(state)

    print(f"\nState  space size : {state_size}")
    print(f"Action space size : {action_size}")
    print(f"Number of agents  : {len(env_info.agents)}\n")

    # ── 3. Create the agent ───────────────────────────────────────────────────────
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    # ── 4. Training loop ──────────────────────────────────────────────────────────

    def dqn(
        n_episodes   = 2000,
        max_t        = 1000,
        eps_start    = 1.0,
        eps_end      = 0.01,
        eps_decay    = 0.995,
        score_target = 13.0,
        score_window = 100,
    ):
        """Double DQN training loop.

        Params
        ======
            n_episodes   (int):   maximum number of training episodes
            max_t        (int):   maximum number of timesteps per episode
            eps_start  (float):   starting value of epsilon (ε)
            eps_end    (float):   minimum value of epsilon
            eps_decay  (float):   multiplicative decay factor applied each episode
            score_target (float): rolling average score considered "solved"
            score_window (int):   number of episodes in the rolling average
        """
        scores        = []                            # all episode scores
        scores_window = deque(maxlen=score_window)    # rolling window
        eps           = eps_start

        for i_episode in range(1, n_episodes + 1):
            env_info  = env.reset(train_mode=True)[brain_name]
            state     = env_info.vector_observations[0]
            score     = 0

            for _ in range(max_t):
                action     = agent.act(state, eps)
                env_info   = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward     = env_info.rewards[0]
                done       = env_info.local_done[0]

                agent.step(state, action, reward, next_state, done)

                state  = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay * eps)

            avg = np.mean(scores_window)
            print(f"\rEpisode {i_episode:4d}  Avg Score (last {score_window}): {avg:.2f}", end="")

            if i_episode % score_window == 0:
                print(f"\rEpisode {i_episode:4d}  Avg Score (last {score_window}): {avg:.2f}")

            if avg >= score_target:
                print(
                    f"\n✓ Solved in {i_episode - score_window:d} episodes!"
                    f"  Average Score: {avg:.2f}"
                )
                torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
                print("Model weights saved to checkpoint.pth")
                break

        return scores


    scores = dqn()

    # ── 5. Plot rewards ───────────────────────────────────────────────────────────
    rolling_avg = [
        np.mean(scores[max(0, i - 99) : i + 1]) for i in range(len(scores))
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(scores)), scores, alpha=0.4, color="steelblue", label="Episode score")
    ax.plot(np.arange(len(scores)), rolling_avg, color="orange", linewidth=2, label="100-ep average")
    ax.axhline(y=13.0, color="red", linestyle="--", linewidth=1.5, label="Target (+13)")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Score")
    ax.set_title("Double DQN — Navigation (Banana Collector)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("scores.png", dpi=150)
    print("Reward plot saved to scores.png")

    env.close()
