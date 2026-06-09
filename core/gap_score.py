"""
core/gap_score.py
------------------
Primordial Void — Intent Gap Score

THE core invention of the paper.

Measures how far a trajectory's behavior is from what the intent model
(designer's proxy) would do — using KL divergence at each step.

High gap score = agent is doing something the designer didn't intend.
Low gap score  = agent is doing exactly what the designer expected.

Functions:
  compute_gap_score(trajectory, intent_model)          → float (mean KL)
  compute_gap_score_sequence(trajectory, intent_model) → list of per-step KL
  normalize_gap_score(score, min_score, max_score)     → float in [0, 1]
"""

import numpy as np
from models.intent_model import IntentModel

EPSILON = 1e-8


def compute_gap_score(trajectory: list, intent_model: IntentModel) -> float:
    """
    Compute the mean intent gap score for a full trajectory.

    trajectory: list of (observation, action) tuples — one complete episode.

    For each step:
      1. Get intent model's action probability distribution Q = π_intent(obs)
      2. Build agent's one-hot distribution P (1.0 on taken action, 0.0 elsewhere)
      3. Compute KL(P || Q) = sum(P * log(P / Q))
         where Q is clipped by epsilon to avoid log(0)

    Returns the mean KL divergence across all steps (a single float).
    """
    if not trajectory:
        return 0.0

    kl_values = compute_gap_score_sequence(trajectory, intent_model)
    return float(np.mean(kl_values))


def compute_gap_score_sequence(trajectory: list, intent_model: IntentModel) -> list:
    """
    Compute per-step KL divergence values for a trajectory.

    Returns a list of floats, one per step.
    Useful for identifying exactly which steps constitute the exploit.
    """
    kl_values = []

    for (obs, action) in trajectory:
        # Q: intent model's probability distribution over actions
        Q = intent_model.get_action_probs(obs).astype(np.float64)
        Q = np.clip(Q, EPSILON, 1.0)   # avoid log(0)

        # P: agent's one-hot distribution (certainty on the taken action)
        action_dim = len(Q)
        P = np.zeros(action_dim, dtype=np.float64)
        P[action] = 1.0

        # KL divergence: KL(P || Q) = sum(P * log(P / Q))
        # Only the term where P[i] > 0 contributes (0 * log(0) = 0)
        kl = float(P[action] * np.log(P[action] / Q[action]))
        kl_values.append(kl)

    return kl_values


def normalize_gap_score(score: float, min_score: float, max_score: float) -> float:
    """
    Min-max normalize a gap score to [0, 1].

    This normalized value is what gets fed as the RL reward signal —
    keeps reward on a consistent scale regardless of environment.
    """
    if max_score <= min_score:
        return 0.0
    normalized = (score - min_score) / (max_score - min_score)
    return float(np.clip(normalized, 0.0, 1.0))


def estimate_score_bounds(
    trajectories: list,
    intent_model: IntentModel,
) -> tuple:
    """
    Estimate min and max gap scores from a set of trajectories.
    Used to calibrate the normalization range before RL training.

    Returns (min_score, max_score).
    """
    scores = [compute_gap_score(traj, intent_model) for traj in trajectories]
    if not scores:
        return 0.0, 1.0
    return float(np.min(scores)), float(np.max(scores))