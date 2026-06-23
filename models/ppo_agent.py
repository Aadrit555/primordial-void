"""
models/ppo_agent.py
---------------------
Primordial Void — PPO Training Wrapper

Trains two types of agents on the same environment:

  "baseline"   → Agent A: standard PPO, raw environment reward
  "gap_reward" → Agent B: standard PPO, but reward = intent gap score
                 (environment wrapped with GapRewardWrapper)

Both agents use IDENTICAL hyperparameters — the only difference is
the reward signal. This isolates the effect of the gap score.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from core.reward_wrapper import GapRewardWrapper
from core.gap_score import estimate_score_bounds
from models.intent_model import IntentModel


# ------------------------------------------------------------------ #
#  Default PPO hyperparameters (from configs/config.yaml)
# ------------------------------------------------------------------ #
DEFAULT_PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,   # entropy bonus — encourages exploration in sparse-reward mazes
    verbose=0,
)


# ------------------------------------------------------------------ #
#  Build the environment (with or without gap reward wrapper)
# ------------------------------------------------------------------ #
def make_env(env_fn, agent_type: str, intent_model: IntentModel = None,
              score_bounds: tuple = None):
    """
    env_fn: a callable that returns a fresh environment instance.
    agent_type: "baseline" or "gap_reward"
    """
    env = env_fn()

    if agent_type == "gap_reward":
        if intent_model is None or score_bounds is None:
            raise ValueError("gap_reward agent requires intent_model and score_bounds")
        min_score, max_score = score_bounds
        env = GapRewardWrapper(env, intent_model, min_score, max_score)

    env = Monitor(env)
    return env


# ------------------------------------------------------------------ #
#  Exploit discovery rate — evaluation metric
# ------------------------------------------------------------------ #
def evaluate_exploit_discovery_rate(model, env_fn, n_episodes: int = 10) -> float:
    """
    Run the trained policy greedily for n_episodes.
    Returns the fraction of episodes where the agent used the exploit.

    Requires the env's step() info dict to contain "used_exploit" (GridWorld does).
    """
    discovered = 0

    for _ in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset()
        used_exploit = False
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if info.get("used_exploit", False):
                used_exploit = True
            done = terminated or truncated

        env.close()
        if used_exploit:
            discovered += 1

    return discovered / n_episodes


# ------------------------------------------------------------------ #
#  Main training function
# ------------------------------------------------------------------ #
def train_agent(
    env_fn,
    agent_type: str,
    total_timesteps: int,
    seed: int,
    run_name: str,
    intent_model: IntentModel = None,
    score_bounds: tuple = None,
    save_dir: str = "models/saved",
    eval_callback_every: int = None,
    verbose: bool = True,
):
    """
    Train a PPO agent.

    agent_type: "baseline" or "gap_reward"
    intent_model / score_bounds: required if agent_type == "gap_reward"

    Returns the trained model.
    """
    env = make_env(env_fn, agent_type, intent_model, score_bounds)

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        **DEFAULT_PPO_KWARGS,
    )

    if verbose:
        print(f"  Training {run_name} (agent_type={agent_type}, seed={seed})...")

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_name}.zip")
    model.save(save_path)

    if verbose:
        print(f"  Saved → {save_path}")

    env.close()
    return model


# ------------------------------------------------------------------ #
#  Convenience — train both agent types for one seed
# ------------------------------------------------------------------ #
def train_both_agents(
    env_fn,
    intent_model: IntentModel,
    score_bounds: tuple,
    total_timesteps: int,
    seed: int,
    eval_episodes: int = 10,
    verbose: bool = True,
):
    """
    Trains Agent A (baseline) and Agent B (gap_reward) for one seed.
    Returns a dict with both models and their exploit discovery rates.
    """
    if verbose:
        print(f"\n--- Seed {seed} ---")

    # Agent A — baseline (raw reward)
    model_a = train_agent(
        env_fn, "baseline", total_timesteps, seed,
        run_name=f"agent_a_seed{seed}",
        verbose=verbose,
    )
    rate_a = evaluate_exploit_discovery_rate(model_a, env_fn, n_episodes=eval_episodes)

    # Agent B — gap reward
    model_b = train_agent(
        env_fn, "gap_reward", total_timesteps, seed,
        run_name=f"agent_b_seed{seed}",
        intent_model=intent_model,
        score_bounds=score_bounds,
        verbose=verbose,
    )
    rate_b = evaluate_exploit_discovery_rate(model_b, env_fn, n_episodes=eval_episodes)

    if verbose:
        print(f"  Agent A (baseline)   exploit discovery rate: {rate_a:.2f}")
        print(f"  Agent B (gap_reward) exploit discovery rate: {rate_b:.2f}")

    return {
        "model_a": model_a,
        "model_b": model_b,
        "rate_a": rate_a,
        "rate_b": rate_b,
    }
