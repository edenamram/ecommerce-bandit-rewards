"""
metrics.py
----------
Evaluation metrics for the e-commerce bandit experiment.

Metrics implemented:
  1. CTR (Click-Through Rate)         — fraction of trials with reward > 0
  2. Purchase Rate                    — fraction of trials with a transaction
  3. Gini Index                       — recommendation diversity (lower = more diverse)
  4. Cumulative Reward                — total reward accumulated over time
  5. Average Reward                   — mean reward per trial (learning curve)
"""

import numpy as np
from collections import Counter
from typing import List, Dict


# ---------------------------------------------------------------------------
# 1. CTR
# ---------------------------------------------------------------------------

def ctr(rewards: List[float]) -> float:
    """
    Fraction of trials where the agent received a positive reward.

    Equation 5 from Semenov et al.:
        CTR = #clicks / #recommendations
    Here 'click' = any event that yields reward > 0.
    """
    if len(rewards) == 0:
        return 0.0
    return float(np.mean(np.array(rewards) > 0))


# ---------------------------------------------------------------------------
# 2. Purchase Rate
# ---------------------------------------------------------------------------

def purchase_rate(event_types: List[str]) -> float:
    """Fraction of recommended trials that resulted in a transaction."""
    if len(event_types) == 0:
        return 0.0
    n_purchases = sum(1 for e in event_types if e == "transaction")
    return n_purchases / len(event_types)


# ---------------------------------------------------------------------------
# 3. Gini Index (recommendation diversity)
# ---------------------------------------------------------------------------

def gini_index(recommendation_counts: Dict[int, int]) -> float:
    """
    Measures concentration of recommendations across items.
    Lower Gini = more uniform / diverse recommendations.
    Higher Gini = a few items dominate.

    Equation 6 from Semenov et al.:
        Gini = Σ_{a1,a2 ∈ A} |rc(a1) - rc(a2)| / (2 * |A| * Σ rc(a))

    Parameters
    ----------
    recommendation_counts : {item_id: count} for all items that were ever
                            in the pool (including those with count=0).
    """
    counts = np.array(list(recommendation_counts.values()), dtype=float)
    total = counts.sum()
    if total == 0 or len(counts) <= 1:
        return 0.0

    # Compute all pairwise absolute differences efficiently
    abs_diff_sum = np.sum(np.abs(counts[:, None] - counts[None, :]))
    return float(abs_diff_sum / (2 * len(counts) * total))


# ---------------------------------------------------------------------------
# 4. Cumulative Reward
# ---------------------------------------------------------------------------

def cumulative_reward(rewards: List[float]) -> np.ndarray:
    """
    Running cumulative sum of rewards.
    Returns an array of the same length as rewards.
    """
    return np.cumsum(rewards)


# ---------------------------------------------------------------------------
# 5. Average Reward (rolling window)
# ---------------------------------------------------------------------------

def average_reward(rewards: List[float], window: int = 500) -> np.ndarray:
    """
    Rolling mean of rewards over a sliding window.
    Useful for plotting the learning curve.
    """
    r = np.array(rewards, dtype=float)
    if len(r) < window:
        return np.cumsum(r) / (np.arange(len(r)) + 1)
    kernel = np.ones(window) / window
    smoothed = np.convolve(r, kernel, mode="valid")
    # Pad beginning with expanding mean so the array is same length
    prefix = np.cumsum(r[: window - 1]) / (np.arange(1, window) )
    return np.concatenate([prefix, smoothed])


# ---------------------------------------------------------------------------
# 6. Summary helper
# ---------------------------------------------------------------------------

def summarise(
    rewards: List[float],
    arm_counts: Dict[int, int],
    event_types: List[str],
    algorithm_name: str,
    reward_name: str,
) -> dict:
    """
    Produce a single summary dict for one (algorithm, reward) combination.
    """
    return {
        "algorithm":    algorithm_name,
        "reward":       reward_name,
        "n_trials":     len(rewards),
        "ctr":          round(ctr(rewards), 4),
        "purchase_rate":round(purchase_rate(event_types), 4),
        "gini_index":   round(gini_index(arm_counts), 4),
        "total_reward": round(float(np.sum(rewards)), 2),
        "mean_reward":  round(float(np.mean(rewards)), 4) if rewards else 0.0,
    }
