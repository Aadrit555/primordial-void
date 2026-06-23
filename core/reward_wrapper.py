"""
core/reward_wrapper.py
------------------------
Primordial Void — Gap Reward Wrapper

A Gymnasium RewardWrapper that replaces the environment's native reward
with the normalized intent gap score.

This is the mechanism that turns "find the exploit" into something
an RL agent can be trained on with standard PPO.

How it works:
  1. Every step, record (observation, action) into the current trajectory
  2. Compute gap_score() over the trajectory so far
  3. Normalize it to [0, 1] using pre-computed min/max bounds
  4. Return that normalized value AS the reward — replacing env reward entirely
  5. On episode end, reset the trajectory buffer
"""

import gymnasium as gym
import numpy as np

from core.gap_score import compute_step_kl, normalize_gap_score
from models.intent_model import IntentModel


class GapRewardWrapper(gym.RewardWrapper):

    def __init__(
        self,
        env: gym.Env,
        intent_model: IntentModel,
        min_score: float,
        max_score: float,
    ):
        super().__init__(env)
        self.intent_model = intent_model
        self.min_score = min_score
        self.max_score = max_score

        # Track running sum of per-step KL divergences and step count.
        # gap_score over a trajectory is the MEAN of per-step KL values,
        # so we can update it incrementally in O(1) per step instead of
        # recomputing over the whole trajectory (O(n) -> O(n^2) per episode).
        self._kl_sum = 0.0
        self._step_count = 0
        # Track the observation before the action is taken
        self.previous_observation = None

    # ------------------------------------------------------------------ #
    #  Reset — clear the running KL accumulator
    # ------------------------------------------------------------------ #
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._kl_sum = 0.0
        self._step_count = 0
        self.previous_observation = obs
        return obs, info

    # ------------------------------------------------------------------ #
    #  Step — intercept and replace the reward
    # ------------------------------------------------------------------ #
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Compute the KL divergence for THIS step only (O(1) — single forward pass)
        step_kl = compute_step_kl(self.previous_observation, action, self.intent_model)

        # Update running mean incrementally
        self._kl_sum += step_kl
        self._step_count += 1
        gap_score = self._kl_sum / self._step_count

        normalized_reward = normalize_gap_score(gap_score, self.min_score, self.max_score)

        # Update previous observation for next step
        self.previous_observation = observation

        # Reset accumulator at episode end
        if terminated or truncated:
            self._kl_sum = 0.0
            self._step_count = 0

        # Replace the reward entirely
        return observation, normalized_reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Required by RewardWrapper base class (unused — we override step())
    # ------------------------------------------------------------------ #
    def reward(self, reward):
        return reward