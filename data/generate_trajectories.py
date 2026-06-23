"""
data/generate_trajectories.py
------------------------------
Primordial Void — Trajectory Generator

Generates two datasets:
  1. correct_trajectories.npy  — BFS optimal paths (no exploit)   → trains intent model
  2. exploit_trajectories.npy  — BFS paths using exploit shortcut  → used in validation

Each trajectory is saved as a list of (observation, action) pairs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.gridworld import GridWorld


# ------------------------------------------------------------------ #
#  Action inference from position delta
# ------------------------------------------------------------------ #
def _pos_to_action(pos_from: tuple, pos_to: tuple) -> int:
    """Convert a move from pos_from to pos_to into an action integer."""
    dr = pos_to[0] - pos_from[0]
    dc = pos_to[1] - pos_from[1]
    mapping = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    return mapping.get((dr, dc), 1)  # default down if stuck


# ------------------------------------------------------------------ #
#  Generate one trajectory following a BFS path
# ------------------------------------------------------------------ #
def _generate_one_trajectory(env: GridWorld, use_exploit: bool) -> list:
    """
    Run one episode following the BFS shortest path.
    Returns list of (observation, action) tuples.
    """
    obs, _ = env.reset()
    path = env.bfs_path(use_exploit=use_exploit)

    if len(path) < 2:
        return []

    trajectory = []
    current_pos_idx = 0

    for step_idx in range(len(path) - 1):
        pos_from = path[step_idx]
        pos_to   = path[step_idx + 1]
        action   = _pos_to_action(pos_from, pos_to)

        trajectory.append((obs.copy(), action))
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    return trajectory


# ------------------------------------------------------------------ #
#  Main generation function
# ------------------------------------------------------------------ #
def generate_datasets(
    n_correct: int = 200,
    n_exploit: int = 200,
    save_dir: str = None,
    verbose: bool = True,
) -> dict:
    """
    Generate correct and exploit trajectory datasets.

    Returns a dict with keys:
      'correct'  : list of trajectories (no exploit used)
      'exploit'  : list of trajectories (exploit used)
    """
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "trajectories"
        )
    os.makedirs(save_dir, exist_ok=True)

    # --- Correct trajectories (GridWorld-C, no exploit) ---
    env_c = GridWorld(exploit_mode=False)
    correct_trajectories = []
    for i in range(n_correct):
        traj = _generate_one_trajectory(env_c, use_exploit=False)
        if traj:
            correct_trajectories.append(traj)
    env_c.close()

    # --- Exploit trajectories (GridWorld-E, uses shortcut) ---
    env_e = GridWorld(exploit_mode=True)
    exploit_trajectories = []
    for i in range(n_exploit):
        traj = _generate_one_trajectory(env_e, use_exploit=True)
        if traj:
            exploit_trajectories.append(traj)
    env_e.close()

    # --- Save ---
    correct_path = os.path.join(save_dir, "correct_trajectories.npy")
    exploit_path = os.path.join(save_dir, "exploit_trajectories.npy")
    np.save(correct_path, np.array(correct_trajectories, dtype=object), allow_pickle=True)
    np.save(exploit_path, np.array(exploit_trajectories, dtype=object), allow_pickle=True)

    if verbose:
        print(f"Generated {len(correct_trajectories)} correct trajectories → {correct_path}")
        print(f"Generated {len(exploit_trajectories)} exploit trajectories → {exploit_path}")
        if correct_trajectories:
            print(f"  Correct path length: {len(correct_trajectories[0])} steps")
        if exploit_trajectories:
            print(f"  Exploit path length: {len(exploit_trajectories[0])} steps  (shorter = exploit worked)")

    return {
        "correct": correct_trajectories,
        "exploit": exploit_trajectories,
    }


if __name__ == "__main__":
    print("=" * 50)
    print("Primordial Void — Generating trajectories")
    print("=" * 50)
    datasets = generate_datasets(verbose=True)
    print("\nDone. Trajectories saved to data/trajectories/")