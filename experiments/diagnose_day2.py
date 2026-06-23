"""
experiments/diagnose_day2.py
------------------------------
Primordial Void — Day 2 Diagnostic

Quick check (no training, runs in seconds): do the trained agents
reach the GOAL at all, regardless of whether they use the exploit?

This isolates whether the problem is:
  (a) reward is too sparse for PPO to learn the task at all, or
  (b) agents solve the maze but just don't use the exploit cell

Run:
  py experiments/diagnose_day2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from envs.gridworld import GridWorld


def run_episode(model, env, max_steps=200, deterministic=True):
    obs, _ = env.reset()
    positions = [tuple(env.agent_pos)]
    used_exploit = False
    reached_goal = False

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        positions.append(tuple(env.agent_pos))
        if info.get("used_exploit", False):
            used_exploit = True
        if terminated:
            reached_goal = True
        if terminated or truncated:
            break

    return {
        "positions": positions,
        "steps_taken": len(positions) - 1,
        "reached_goal": reached_goal,
        "used_exploit": used_exploit,
    }


def diagnose(model_path: str, label: str, n_episodes: int = 20):
    print(f"\n{'='*60}")
    print(f"{label}  ({model_path})")
    print(f"{'='*60}")

    model = PPO.load(model_path)

    goal_count = 0
    exploit_count = 0
    step_counts = []

    for ep in range(n_episodes):
        env = GridWorld(exploit_mode=True)
        result = run_episode(model, env, deterministic=True)
        env.close()

        if result["reached_goal"]:
            goal_count += 1
            step_counts.append(result["steps_taken"])
        if result["used_exploit"]:
            exploit_count += 1

        if ep < 3:  # print first few trajectories for inspection
            print(f"  Episode {ep}: reached_goal={result['reached_goal']}, "
                  f"used_exploit={result['used_exploit']}, "
                  f"steps={result['steps_taken']}")
            print(f"    Path: {result['positions'][:10]}"
                  f"{'...' if len(result['positions']) > 10 else ''}")

    print(f"\n  Reached goal:  {goal_count}/{n_episodes}  ({goal_count/n_episodes:.0%})")
    print(f"  Used exploit:  {exploit_count}/{n_episodes}  ({exploit_count/n_episodes:.0%})")
    if step_counts:
        print(f"  Avg steps when goal reached: {np.mean(step_counts):.1f}")
        print(f"  (Normal path=29 steps, Exploit path=23 steps)")

    return {
        "goal_rate": goal_count / n_episodes,
        "exploit_rate": exploit_count / n_episodes,
        "avg_steps": np.mean(step_counts) if step_counts else None,
    }


def stress_test_random_policy(n_episodes: int = 20):
    """Sanity check: can a RANDOM policy ever reach the goal in 200 steps?"""
    print(f"\n{'='*60}")
    print("RANDOM POLICY BASELINE (sanity check)")
    print(f"{'='*60}")

    goal_count = 0
    for ep in range(n_episodes):
        env = GridWorld(exploit_mode=True)
        obs, _ = env.reset()
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                goal_count += 1
                break
            if truncated:
                break
        env.close()

    print(f"  Random policy reached goal: {goal_count}/{n_episodes} ({goal_count/n_episodes:.0%})")
    print("  (If this is also ~0%, the maze is hard for random exploration —")
    print("   PPO needs many rollouts before it ever sees a +1 reward.)")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    stress_test_random_policy(n_episodes=20)

    diagnose("models/saved/agent_a_seed0.zip", "AGENT A (baseline reward)", n_episodes=20)
    diagnose("models/saved/agent_b_seed0.zip", "AGENT B (gap_reward)", n_episodes=20)