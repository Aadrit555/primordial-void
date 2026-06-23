"""
experiments/day2_train.py
----------------------------
Primordial Void — Day 2 Experiment

Wires the intent gap score in as an RL reward signal and runs the
first head-to-head comparison:

  Agent A (baseline)   — standard PPO, raw environment reward
  Agent B (gap_reward) — standard PPO, reward = intent gap score

Steps:
  1. Load (or train) the intent model from Day 1
  2. Estimate gap score bounds for normalization
  3. Train Agent A and Agent B across N seeds (small N for a quick run)
  4. Evaluate exploit discovery rate for each
  5. Run Mann-Whitney + Cohen's d
  6. Plot Figure 2

Run (quick smoke test — ~2-5 min):
  python experiments/day2_train.py --timesteps 20000 --seeds 0 1

Run (paper-quality — slow, ~hours):
  python experiments/day2_train.py --timesteps 500000 --seeds 0 1 2 3 4 5 6 7 8 9
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.gridworld import GridWorld
from data.generate_trajectories import generate_datasets
from models.intent_model import IntentModel
from core.gap_score import estimate_score_bounds
from models.ppo_agent import train_both_agents
from analysis.plot import plot_exploit_discovery_comparison
from analysis.stats import summarize_results


def get_or_train_intent_model(verbose=True):
    """Load the Day 1 intent model if it exists, otherwise train a fresh one."""
    model_path = "models/saved/intent_model.pt"

    if os.path.exists(model_path):
        if verbose:
            print(f"[1/6] Loading existing intent model from {model_path}")
        return IntentModel.load(model_path)

    if verbose:
        print("[1/6] No saved intent model found — training a new one...")
    datasets = generate_datasets(n_correct=200, n_exploit=200, verbose=verbose)
    intent_model = IntentModel(obs_dim=66, action_dim=4, hidden_size=64)
    intent_model.train_on_trajectories(datasets["correct"], epochs=50, verbose=verbose)
    intent_model.save(model_path)
    return intent_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000,
                         help="Total PPO training timesteps per agent (default 20000 for quick run)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                         help="Random seeds to run (default: 0 1)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                         help="Episodes used to compute exploit discovery rate")
    args = parser.parse_args()

    print("=" * 60)
    print("PRIMORDIAL VOID — Day 2 Experiment")
    print("Gap Reward vs Raw Reward — Head-to-Head")
    print("=" * 60)
    print(f"\nConfig: timesteps={args.timesteps}, seeds={args.seeds}, "
          f"eval_episodes={args.eval_episodes}\n")

    # ── Step 1: Intent model ────────────────────────────────────────
    intent_model = get_or_train_intent_model(verbose=True)

    # ── Step 2: Estimate gap score bounds ───────────────────────────
    print("\n[2/6] Estimating gap score bounds for normalization...")
    datasets = generate_datasets(n_correct=50, n_exploit=50, verbose=False)
    all_trajs = datasets["correct"] + datasets["exploit"]
    min_score, max_score = estimate_score_bounds(all_trajs, intent_model)
    # Add a small margin so unseen trajectories aren't clipped at exactly 0 or 1
    score_range = max_score - min_score
    min_score -= 0.05 * score_range
    max_score += 0.05 * score_range
    print(f"  Gap score bounds: [{min_score:.4f}, {max_score:.4f}]")

    # ── Step 3: Environment factory ─────────────────────────────────
    def env_fn():
        return GridWorld(exploit_mode=True)

    # ── Step 4: Train both agents across seeds ──────────────────────
    print(f"\n[3/6] Training Agent A (baseline) and Agent B (gap_reward) "
          f"across {len(args.seeds)} seed(s)...")

    baseline_rates = []
    gap_rates = []

    for seed in args.seeds:
        result = train_both_agents(
            env_fn=env_fn,
            intent_model=intent_model,
            score_bounds=(min_score, max_score),
            total_timesteps=args.timesteps,
            seed=seed,
            eval_episodes=args.eval_episodes,
            verbose=True,
        )
        baseline_rates.append(result["rate_a"])
        gap_rates.append(result["rate_b"])

    # ── Step 5: Statistical comparison ──────────────────────────────
    print("\n[4/6] Running statistical comparison...")
    summarize_results(baseline_rates=baseline_rates, gap_rates=gap_rates)

    # ── Step 6: Plot Figure 2 ────────────────────────────────────────
    print("\n[5/6] Plotting Figure 2...")
    fig_path = plot_exploit_discovery_comparison(
        baseline_rates, gap_rates, save_dir="figures", env_name="GridWorld"
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n[6/6] Summary")
    print("=" * 60)
    print(f"  Agent A (baseline)   discovery rates: {baseline_rates}")
    print(f"  Agent B (gap_reward) discovery rates: {gap_rates}")
    mean_a = np.mean(baseline_rates)
    mean_b = np.mean(gap_rates)
    if mean_b > mean_a:
        print(f"\n  Agent B finds the exploit MORE often than Agent A "
              f"({mean_b:.2f} vs {mean_a:.2f}).")
        print("  This is the core Day 2 result — the gap reward works.")
    elif mean_b == mean_a:
        print("\n  No difference yet — try more timesteps or more seeds.")
    else:
        print("\n  Agent A found it more often — check gap score bounds "
              "or increase --timesteps.")
    print(f"\n  Figure saved to {fig_path}")
    print("=" * 60)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()