"""
main.py
-------
End-to-end runner for the e-commerce contextual bandit experiment.

Usage:
    python main.py

Produces:
    results/summary_table.csv        — CTR / Gini / Purchase Rate for all combos
    results/ctr_comparison.png       — bar chart: CTR per (algo, reward)
    results/gini_comparison.png      — bar chart: Gini index per (algo, reward)
    results/learning_curves.png      — avg reward over time per (algo, reward)
    results/cumulative_rewards.png   — cumulative reward over time
    results/recommendation_hist.png  — item recommendation distributions
    results/purchase_rate.png        — bar chart: purchase rate per (algo, reward)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "raw")
RES_DIR  = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

# ── project imports ──────────────────────────────────────────────────────────
from src.preprocessing       import load_and_prepare, get_arm_list, get_context_dim
from algorithms              import RandomBandit, LinUCB, DivLinUCB
from evaluation.simulator    import run_all_experiments

# ── plot style ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10")
ALGO_COLORS = {
    "Random":    "#6c757d",
    "LinUCB":    "#0d6efd",
    "DivLinUCB": "#198754",
}
REWARD_LABELS = {
    "reward_binary":    "R1 Binary",
    "reward_weighted":  "R2 Weighted",
    "reward_maxstage":  "R3 Max-Stage",
}

# ─────────────────────────────────────────────────────────────────────────────
# Config — tweak these to adjust the experiment
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = dict(
    n_items      = 50,     # number of items (arms)
    n_components = 10,     # PCA context dimensionality
    min_events   = 5,      # min events per item to be included
    alpha        = 2.62,   # LinUCB / DivLinUCB exploration parameter
    beta         = 5000.0, # DivLinUCB diversity strength (needs to be high for large datasets)
    random_state = 42,
)


# =============================================================================
# Main
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # 0. Download data if not already present
    # ------------------------------------------------------------------
    from download_data import download
    download()

    # ------------------------------------------------------------------
    # 1. Preprocessing
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  E-COMMERCE CONTEXTUAL BANDIT — REWARD COMPARISON")
    print("="*60)

    trials = load_and_prepare(
        data_dir     = DATA_DIR,
        n_items      = CONFIG["n_items"],
        n_components = CONFIG["n_components"],
        min_events   = CONFIG["min_events"],
        random_state = CONFIG["random_state"],
    )

    arm_pool   = get_arm_list(trials)
    ctx_dim    = get_context_dim(trials)
    n_features = ctx_dim   # context dim = 2 * n_components (user + item)

    print(f"Arms (items)   : {len(arm_pool)}")
    print(f"Context dim    : {n_features}")
    print(f"Total trials   : {len(trials):,}")
    print(f"Event breakdown:\n{trials['event_type'].value_counts()}\n")

    # ------------------------------------------------------------------
    # 2. Algorithm setup
    # ------------------------------------------------------------------
    algorithms = [
        RandomBandit(name="Random"),
        LinUCB(n_features=n_features, alpha=CONFIG["alpha"], name="LinUCB"),
        DivLinUCB(n_features=n_features, alpha=CONFIG["alpha"],
                  beta=CONFIG["beta"], name="DivLinUCB"),
    ]

    reward_cols = ["reward_binary", "reward_weighted", "reward_maxstage"]

    # ------------------------------------------------------------------
    # 3. Run experiments
    # ------------------------------------------------------------------
    results = run_all_experiments(
        trials          = trials,
        warmup_fraction = 0.5,
        algorithms  = algorithms,
        reward_cols = reward_cols,
        arm_pool    = arm_pool,
        seed        = CONFIG["random_state"],
        verbose     = True,
    )

    # ------------------------------------------------------------------
    # 4. Summary table
    # ------------------------------------------------------------------
    summary_cols = [
        "algorithm", "reward", "n_trials",
        "mean_reward", "total_reward", "ctr",
        "purchase_rate", "gini_index", "match_rate",
    ]
    summary_df = pd.DataFrame([
        {k: r[k] for k in summary_cols} for r in results
    ])
    summary_df["reward"] = summary_df["reward"].map(REWARD_LABELS)
    summary_df.to_csv(os.path.join(RES_DIR, "summary_table.csv"), index=False)

    print("\n── Summary Table ──────────────────────────────────────────")
    print(summary_df.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    _plot_bar_metric(results, "mean_reward",   "Mean Reward (chosen arm)",    "ctr_comparison.png")
    _plot_bar_metric(results, "gini_index",    "Gini Index (↓ = more diverse)", "gini_comparison.png")
    _plot_bar_metric(results, "purchase_rate", "Purchase Rate",               "purchase_rate.png")
    _plot_learning_curves(results)
    _plot_cumulative_rewards(results)
    _plot_recommendation_histograms(results, arm_pool)

    print(f"\nAll results saved to: {RES_DIR}/")
    print("Done ✓\n")


# =============================================================================
# Plot helpers
# =============================================================================

def _result_label(r: dict) -> str:
    return f"{r['algorithm']}\n{REWARD_LABELS[r['reward']]}"


def _plot_bar_metric(results: list, metric: str, ylabel: str, filename: str):
    """Grouped bar chart comparing a single metric across all (algo, reward) combos."""
    algos   = ["Random", "LinUCB", "DivLinUCB"]
    rewards = ["reward_binary", "reward_weighted", "reward_maxstage"]
    x       = np.arange(len(rewards))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, algo in enumerate(algos):
        vals = [
            next((r[metric] for r in results
                  if r["algorithm"] == algo and r["reward"] == rw), 0.0)
            for rw in rewards
        ]
        bars = ax.bar(
            x + i * width, vals, width,
            label=algo, color=ALGO_COLORS[algo], alpha=0.88, edgecolor="white"
        )
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7.5
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Algorithm and Reward Function")
    ax.legend(title="Algorithm")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def _plot_learning_curves(results: list):
    """Average reward over time (smoothed) — one subplot per reward function."""
    rewards = ["reward_binary", "reward_weighted", "reward_maxstage"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, rw in zip(axes, rewards):
        for r in results:
            if r["reward"] != rw:
                continue
            curve = r["avg_reward_curve"]
            if not curve:
                continue
            ax.plot(curve, label=r["algorithm"],
                    color=ALGO_COLORS[r["algorithm"]], linewidth=1.6)

        ax.set_title(REWARD_LABELS[rw])
        ax.set_xlabel("Accepted Trials")
        ax.set_ylabel("Avg Reward (rolling)")
        ax.legend(fontsize=8)

    fig.suptitle("Learning Curves — Average Reward Over Time", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: learning_curves.png")


def _plot_cumulative_rewards(results: list):
    """Cumulative reward curves — one subplot per reward function."""
    rewards = ["reward_binary", "reward_weighted", "reward_maxstage"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, rw in zip(axes, rewards):
        for r in results:
            if r["reward"] != rw:
                continue
            curve = r["cumulative_reward_curve"]
            if not curve:
                continue
            ax.plot(curve, label=r["algorithm"],
                    color=ALGO_COLORS[r["algorithm"]], linewidth=1.6)

        ax.set_title(REWARD_LABELS[rw])
        ax.set_xlabel("Accepted Trials")
        ax.set_ylabel("Cumulative Reward")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"
        ))

    fig.suptitle("Cumulative Reward Over Time", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "cumulative_rewards.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: cumulative_rewards.png")


def _plot_recommendation_histograms(results: list, arm_pool: list):
    """
    Distribution of recommendation counts across items.
    Mirrors Figure 4 / Fig 5-7 from Semenov et al.
    Shows Random vs LinUCB vs DivLinUCB for the Binary reward.
    """
    target_reward = "reward_binary"
    algos = ["Random", "LinUCB", "DivLinUCB"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, algo in zip(axes, algos):
        r = next(
            (x for x in results
             if x["algorithm"] == algo and x["reward"] == target_reward),
            None
        )
        if r is None:
            continue

        counts = [r["arm_counts"].get(a, 0) for a in arm_pool]
        ax.bar(range(len(arm_pool)), sorted(counts, reverse=True),
               color=ALGO_COLORS[algo], alpha=0.85, edgecolor="none")
        ax.set_title(f"{algo}  (Gini={r['gini_index']:.3f})")
        ax.set_xlabel("Item Rank (by rec. count)")
        ax.set_ylabel("Recommendation Count")

    fig.suptitle(
        "Item Recommendation Distribution — R1 Binary Reward\n"
        "(lower Gini = more uniform = more diverse)",
        fontsize=12, y=1.03
    )
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "recommendation_hist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: recommendation_hist.png")


# =============================================================================

if __name__ == "__main__":
    main()