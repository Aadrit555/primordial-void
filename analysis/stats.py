"""
analysis/stats.py
-------------------
Primordial Void — Shared Statistical Functions

Used across all experiments (Day 1 validation, ablation, transfer)
to keep statistical reporting consistent.
"""

import numpy as np
from scipy import stats


def mann_whitney_test(group_a: list, group_b: list, alternative: str = "two-sided") -> dict:
    """
    Run Mann-Whitney U test between two groups.

    Returns dict with keys: u_statistic, p_value, significant (bool, threshold 0.05)
    """
    u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative=alternative)
    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def cohens_d(group_a: list, group_b: list) -> float:
    """
    Cohen's d effect size between two groups.

    Formula: (mean_a - mean_b) / pooled_std
    """
    a = np.array(group_a, dtype=np.float64)
    b = np.array(group_b, dtype=np.float64)

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0

    pooled_var = (
        (na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)
    ) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0

    return float((np.mean(a) - np.mean(b)) / pooled_std)


def effect_size_label(d: float) -> str:
    """Human-readable label for Cohen's d magnitude."""
    ad = abs(d)
    if ad >= 0.8:
        return "large"
    elif ad >= 0.5:
        return "medium"
    elif ad >= 0.2:
        return "small"
    return "negligible"


def bootstrap_ci(data: list, n_iterations: int = 10000, ci: float = 0.95, seed: int = 0) -> tuple:
    """
    Bootstrap confidence interval on the mean.

    Returns (lower, upper).
    """
    rng = np.random.default_rng(seed)
    data = np.array(data, dtype=np.float64)
    means = np.empty(n_iterations)

    for i in range(n_iterations):
        sample = rng.choice(data, size=len(data), replace=True)
        means[i] = np.mean(sample)

    alpha = (1 - ci) / 2
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1 - alpha))
    return lower, upper


def summarize_results(
    exploit_scores: list = None,
    normal_scores: list = None,
    baseline_rates: list = None,
    gap_rates: list = None,
):
    """
    Prints a formatted summary table of all statistics used in the paper.
    Any argument can be None — only provided groups are summarized.
    """
    print("=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)

    if exploit_scores is not None and normal_scores is not None:
        mw = mann_whitney_test(exploit_scores, normal_scores, alternative="greater")
        d = cohens_d(exploit_scores, normal_scores)
        print("\nGap Score — Exploit vs Normal")
        print(f"  Exploit mean : {np.mean(exploit_scores):.4f} ± {np.std(exploit_scores):.4f}")
        print(f"  Normal  mean : {np.mean(normal_scores):.4f} ± {np.std(normal_scores):.4f}")
        print(f"  Mann-Whitney : U={mw['u_statistic']:.1f}, p={mw['p_value']:.6f} "
              f"({'significant' if mw['significant'] else 'not significant'})")
        print(f"  Cohen's d    : {d:.3f} ({effect_size_label(d)})")

    if baseline_rates is not None and gap_rates is not None:
        mw = mann_whitney_test(gap_rates, baseline_rates, alternative="greater")
        d = cohens_d(gap_rates, baseline_rates)
        print("\nExploit Discovery Rate — Agent B (gap) vs Agent A (baseline)")
        print(f"  Agent A mean : {np.mean(baseline_rates):.3f} ± {np.std(baseline_rates):.3f}")
        print(f"  Agent B mean : {np.mean(gap_rates):.3f} ± {np.std(gap_rates):.3f}")
        print(f"  Mann-Whitney : U={mw['u_statistic']:.1f}, p={mw['p_value']:.6f} "
              f"({'significant' if mw['significant'] else 'not significant'})")
        print(f"  Cohen's d    : {d:.3f} ({effect_size_label(d)})")

    print("=" * 60)