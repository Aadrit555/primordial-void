"""
analysis/plot.py
------------------
Primordial Void — Figure Generation

All paper figures are produced by functions in this module.
Consistent style: white background, no top/right spines, font size 12.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
#  Figure 1 — Gap score distributions (Day 1)
# ------------------------------------------------------------------ #
def plot_gap_score_distributions(exploit_scores, normal_scores, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(min(exploit_scores), min(normal_scores)),
        max(max(exploit_scores), max(normal_scores)),
        30,
    )

    ax.hist(exploit_scores, bins=bins, alpha=0.6, color="#FF5722",
            label="Exploit trajectories", edgecolor="white", linewidth=0.5)
    ax.hist(normal_scores, bins=bins, alpha=0.6, color="#2196F3",
            label="Normal trajectories", edgecolor="white", linewidth=0.5)

    ax.axvline(np.mean(exploit_scores), color="#FF5722", linestyle="--", linewidth=2,
                label=f"Exploit mean: {np.mean(exploit_scores):.3f}")
    ax.axvline(np.mean(normal_scores), color="#2196F3", linestyle="--", linewidth=2,
                label=f"Normal mean: {np.mean(normal_scores):.3f}")

    ax.set_xlabel("Intent Gap Score (KL divergence)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Figure 1 — Gap Score Distribution: Exploit vs Normal Trajectories\nPrimordial Void",
                  fontsize=12, pad=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    png_path = os.path.join(save_dir, "figure1.png")
    plt.savefig(os.path.join(save_dir, "figure1.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 1 saved -> {png_path}")
    return png_path


# ------------------------------------------------------------------ #
#  Figure 2 — Exploit discovery rate: baseline vs gap reward (Day 2)
# ------------------------------------------------------------------ #
def plot_exploit_discovery_comparison(baseline_rates, gap_rates, save_dir="figures",
                                       env_name="GridWorld"):
    """
    Two box plots side by side with individual seed points overlaid.

    baseline_rates: list of exploit discovery rates for Agent A (one per seed)
    gap_rates:      list of exploit discovery rates for Agent B (one per seed)
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    data = [baseline_rates, gap_rates]
    labels = ["Agent A\n(baseline reward)", "Agent B\n(gap-reward)"]
    colors = ["#2196F3", "#FF5722"]

    bp = ax.boxplot(
        data,
        labels=labels,
        widths=0.45,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # Overlay individual seed points
    rng = np.random.default_rng(0)
    for i, group in enumerate(data, start=1):
        jitter = rng.uniform(-0.06, 0.06, size=len(group))
        ax.scatter(np.full(len(group), i) + jitter, group,
                    color=colors[i - 1], alpha=0.8, zorder=3, s=40,
                    edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Exploit Discovery Rate", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Figure 2 — Exploit Discovery Rate: Baseline vs Gap-Reward Agent\n{env_name}",
                  fontsize=12, pad=12)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    png_path = os.path.join(save_dir, "figure2.png")
    plt.savefig(os.path.join(save_dir, "figure2.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 2 saved -> {png_path}")
    return png_path


# ------------------------------------------------------------------ #
#  Generic trajectory rendering (delegates to env.render_trajectory)
# ------------------------------------------------------------------ #
def plot_trajectory(env, trajectory, highlight_cells=None, title="Trajectory",
                     save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{title.lower().replace(' ', '_')}.png")
    env.render_trajectory(
        trajectory=trajectory,
        highlight_cells=highlight_cells or [],
        title=title,
        save_path=save_path,
    )
    return save_path


# ------------------------------------------------------------------ #
#  Figure 3 — Transfer similarity (Week 3, placeholder for now)
# ------------------------------------------------------------------ #
def plot_transfer_similarity(graph_a, graph_b, exploit_nodes_a, high_gap_nodes_b,
                              save_dir="figures"):
    import networkx as nx
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, G, highlight_nodes, label, color in zip(
        axes, [graph_a, graph_b], [exploit_nodes_a, high_gap_nodes_b],
        ["GridWorld-E", "Resource Environment"], ["#FF5722", "#FFB74D"]
    ):
        pos = nx.spring_layout(G, seed=0)
        node_colors = [color if n in highlight_nodes else "#90A4AE" for n in G.nodes()]
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=120,
                edge_color="#CFD8DC", with_labels=False, arrows=True)
        ax.set_title(label, fontsize=11)

    fig.suptitle("Figure 3 — Structural Comparison: GridWorld-E vs Resource Environment",
                  fontsize=12)
    plt.tight_layout()
    png_path = os.path.join(save_dir, "figure3.png")
    plt.savefig(os.path.join(save_dir, "figure3.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 3 saved -> {png_path}")
    return png_path