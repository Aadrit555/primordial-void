"""
experiments/day1_validate.py
-----------------------------
Primordial Void — Day 1 Validation Experiment

This is the Week 1 proof-of-concept:
  "Does the gap score signal actually separate exploit from normal behavior?"

Steps:
  1. Generate trajectories (correct + exploit)
  2. Train intent model on correct trajectories
  3. Score all trajectories with gap score
  4. Run Mann-Whitney U test
  5. Plot Figure 1 — the separation chart

Run:
  python experiments/day1_validate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_trajectories import generate_datasets
from models.intent_model import IntentModel
from core.gap_score import compute_gap_score


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def cohens_d(a: list, b: list) -> float:
    """Effect size between two groups."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2)
        / (na + nb - 2)
    )
    return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-8)


def plot_figure1(
    exploit_scores: list,
    normal_scores: list,
    save_dir: str = "figures",
):
    """
    Figure 1 — Gap Score Distributions: Exploit vs Normal Trajectories.
    Two overlapping histograms with vertical mean lines.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(min(exploit_scores), min(normal_scores)),
        max(max(exploit_scores), max(normal_scores)),
        30,
    )

    ax.hist(exploit_scores, bins=bins, alpha=0.6, color="#FF5722",
            label="Exploit trajectories", edgecolor="white", linewidth=0.5)
    ax.hist(normal_scores,  bins=bins, alpha=0.6, color="#2196F3",
            label="Normal trajectories",  edgecolor="white", linewidth=0.5)

    # Mean lines
    ax.axvline(np.mean(exploit_scores), color="#FF5722", linestyle="--",
               linewidth=2, label=f"Exploit mean: {np.mean(exploit_scores):.3f}")
    ax.axvline(np.mean(normal_scores),  color="#2196F3", linestyle="--",
               linewidth=2, label=f"Normal mean:  {np.mean(normal_scores):.3f}")

    ax.set_xlabel("Intent Gap Score (KL divergence)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Figure 1 — Gap Score Distribution: Exploit vs Normal Trajectories\n"
                 "Primordial Void", fontsize=12, pad=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "figure1.pdf")
    png_path = os.path.join(save_dir, "figure1.png")
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Figure 1 saved → {png_path}")
    plt.close()
    return png_path


# ------------------------------------------------------------------ #
#  Main experiment
# ------------------------------------------------------------------ #
def run_validation():
    print("=" * 60)
    print("PRIMORDIAL VOID — Day 1 Validation Experiment")
    print("=" * 60)

    # ── Step 1: Generate trajectories ──────────────────────────────
    print("\n[1/5] Generating trajectories...")
    datasets = generate_datasets(n_correct=200, n_exploit=200, verbose=True)
    correct_trajs = datasets["correct"]
    exploit_trajs = datasets["exploit"]

    # ── Step 2: Train intent model ─────────────────────────────────
    print("\n[2/5] Training intent model on correct trajectories...")
    intent_model = IntentModel(obs_dim=66, action_dim=4, hidden_size=64)
    loss_history = intent_model.train_on_trajectories(
        correct_trajs,
        epochs=50,
        learning_rate=1e-3,
        verbose=True,
    )
    accuracy = intent_model.evaluate_accuracy(correct_trajs)
    print(f"  Training accuracy: {accuracy:.1%}")

    # Save model
    model_path = "models/saved/intent_model.pt"
    intent_model.save(model_path)

    # ── Step 3: Score all trajectories ─────────────────────────────
    print("\n[3/5] Computing gap scores...")
    exploit_scores = [compute_gap_score(t, intent_model) for t in exploit_trajs]
    normal_scores  = [compute_gap_score(t, intent_model) for t in correct_trajs]
    print(f"  Exploit trajectories scored: {len(exploit_scores)}")
    print(f"  Normal  trajectories scored: {len(normal_scores)}")

    # ── Step 4: Statistical tests ───────────────────────────────────
    print("\n[4/5] Running statistical tests...")
    u_stat, p_value = stats.mannwhitneyu(
        exploit_scores, normal_scores, alternative="greater"
    )
    d = cohens_d(exploit_scores, normal_scores)

    print(f"  Exploit mean gap score : {np.mean(exploit_scores):.4f} ± {np.std(exploit_scores):.4f}")
    print(f"  Normal  mean gap score : {np.mean(normal_scores):.4f} ± {np.std(normal_scores):.4f}")
    print(f"  Mann-Whitney U         : {u_stat:.1f}")
    print(f"  p-value                : {p_value:.6f}  {'✓ SIGNIFICANT' if p_value < 0.05 else '✗ not significant'}")
    print(f"  Cohen's d effect size  : {d:.3f}  ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'})")

    # ── Step 5: Plot Figure 1 ───────────────────────────────────────
    print("\n[5/5] Plotting Figure 1...")
    fig_path = plot_figure1(exploit_scores, normal_scores, save_dir="figures")

    # ── Render example trajectories ────────────────────────────────
    print("\n[Bonus] Rendering example trajectories...")
    from envs.gridworld import GridWorld

    env_e = GridWorld(exploit_mode=True)
    exploit_path = env_e.bfs_path(use_exploit=True)
    env_e.render_trajectory(
        trajectory=exploit_path,
        highlight_cells=[env_e.exploit_cell],
        title="Exploit Trajectory (uses passable wall)",
        save_path="figures/exploit_trajectory.png",
    )
    env_e.close()

    env_c = GridWorld(exploit_mode=False)
    normal_path = env_c.bfs_path(use_exploit=False)
    env_c.render_trajectory(
        trajectory=normal_path,
        highlight_cells=[],
        title="Normal Trajectory (intended path)",
        save_path="figures/normal_trajectory.png",
    )
    env_c.close()

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)
    if p_value < 0.05 and d > 0.5:
        print("✓ PASS — Gap score cleanly separates exploit from normal trajectories.")
        print("  The intent gap signal is REAL. Proceed to Day 2.")
    else:
        print("✗ REVIEW — Separation is weak. Check intent model training.")
    print(f"\n  Figures saved to figures/")
    print("=" * 60)

    return {
        "exploit_scores": exploit_scores,
        "normal_scores":  normal_scores,
        "p_value":        p_value,
        "cohens_d":       d,
        "figure":         fig_path,
    }


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_validation()