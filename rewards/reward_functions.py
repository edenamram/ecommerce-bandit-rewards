"""
reward_functions.py
-------------------
The three reward signals for the e-commerce bandit experiment.

Each function takes a single event row (pd.Series or dict) and returns
a scalar reward.  They are also exposed as named constants so the
simulator can select them by string key.
"""

from typing import Union
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# R1 — Binary Reward
# ---------------------------------------------------------------------------

def binary_reward(row: Union[pd.Series, dict]) -> float:
    """
    Returns 1.0 only if the event is a purchase (transaction), else 0.0.

    Pros: Clean, sparse signal — directly optimises purchases.
    Cons: Very rare positive signal; slow learning in cold start.
    """
    return 1.0 if row["event_type"] == "transaction" else 0.0


# ---------------------------------------------------------------------------
# R2 — Weighted Funnel Reward
# ---------------------------------------------------------------------------

# Stage weights reflect how much each step contributes to purchase intent.
FUNNEL_WEIGHTS = {
    "view":        0.1,   # user saw the item
    "addtocart":   0.5,   # strong purchase intent
    "transaction": 1.0,   # actual purchase
}

def weighted_reward(row: Union[pd.Series, dict]) -> float:
    """
    Assigns a fractional reward based on the funnel stage reached.

    Pros: Denser signal — every interaction contributes.
    Cons: May over-reward browsing behaviour without purchases.
    """
    return FUNNEL_WEIGHTS.get(row["event_type"], 0.0)


# ---------------------------------------------------------------------------
# R3 — Max-Stage Reward
# ---------------------------------------------------------------------------

# Pre-computed per (visitor, item) — stored in the trials DataFrame.
# At evaluation time we simply read the column.

def maxstage_reward(row: Union[pd.Series, dict]) -> float:
    """
    Returns the reward corresponding to the *deepest* funnel stage the
    visitor reached for this item (regardless of current event type).

    This rewards the algorithm for recommending items that lead to a
    purchase at *any* point in the session, giving credit for the full
    conversion path.

    Pros: Captures the full conversion intent of the visit.
    Cons: Look-ahead bias in strict online settings (acceptable for
          offline evaluation with historical logs).
    """
    return float(row.get("reward_maxstage", 0.0))


# ---------------------------------------------------------------------------
# Registry — used by simulator and main.py
# ---------------------------------------------------------------------------

REWARD_FUNCTIONS = {
    "binary":    (binary_reward,   "reward_binary"),
    "weighted":  (weighted_reward, "reward_weighted"),
    "maxstage":  (maxstage_reward, "reward_maxstage"),
}
"""
Dict mapping reward name → (callable, dataframe_column_name).
The column name is what was pre-computed during preprocessing and stored
in the trials DataFrame, which the simulator uses directly for speed.
"""
