"""
simulator.py
------------
Offline evaluation for the e-commerce contextual bandit experiment.

Strategy for RetailRocket (implicit feedback, no candidate pool):

  Phase 1 — Warm-up (first warmup_fraction of data):
    All algorithms learn by updating on every displayed item.

  Phase 2 — Evaluation (remaining data):
    Each algorithm makes a REAL choice from the full arm pool.
    Metrics are computed on the CHOSEN item using its historical
    mean reward and whether it was ever purchased.
    The algorithm continues updating on the displayed item.

Key metrics tracked on the CHOSEN arm:
  - mean_reward    : expected reward of chosen items
  - purchase_rate  : fraction of chosen items ever purchased (binary)
  - gini_index     : concentration of the algorithm's choices
  - match_rate     : how often algorithm agreed with the log
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List

from evaluation.metrics import summarise, average_reward, cumulative_reward


class OfflineSimulator:
    def __init__(
        self,
        trials: pd.DataFrame,
        algorithm,
        reward_col: str,
        arm_pool: List[int],
        warmup_fraction: float = 0.5,
        seed: int = 42,
    ):
        self.trials = trials
        self.algorithm = algorithm
        self.reward_col = reward_col
        self.arm_pool = arm_pool
        self.warmup_fraction = warmup_fraction
        self.seed = seed

        # Per-item statistics computed from the FULL dataset
        self.item_mean_reward = (
            trials.groupby("itemid")[reward_col].mean().to_dict()
        )
        # Whether each item was ever purchased (for purchase_rate on chosen arm)
        self.item_ever_purchased = (
            (trials.groupby("itemid")["event_type"]
             .apply(lambda x: (x == "transaction").any()))
            .to_dict()
        )

    def run(self, verbose: bool = True) -> dict:
        np.random.seed(self.seed)
        self.algorithm.reset()

        n_warmup = int(len(self.trials) * self.warmup_fraction)

        chosen_rewards:    List[float] = []
        chosen_purchased:  List[float] = []
        chosen_counts      = defaultdict(int, {a: 0 for a in self.arm_pool})
        match_count        = 0

        iterator = tqdm(
            self.trials.itertuples(index=False),
            total=len(self.trials),
            desc=f"{self.algorithm.name:<14} | {self.reward_col:<18}",
            disable=not verbose,
        )

        for i, row in enumerate(iterator):
            context       = row.context
            displayed_arm = int(row.itemid)
            displayed_rew = float(getattr(row, self.reward_col))

            if i < n_warmup:
                # Warm-up: learn only, no metrics
                self.algorithm.update(displayed_arm, context, displayed_rew)
            else:
                # Evaluation: make a real choice
                chosen_arm = self.algorithm.select_arm(context, self.arm_pool)

                # Reward = historical mean reward of chosen item
                chosen_rew = self.item_mean_reward.get(chosen_arm, 0.0)
                # Purchase = was this item ever bought?
                was_purchased = float(
                    self.item_ever_purchased.get(chosen_arm, False)
                )

                chosen_rewards.append(chosen_rew)
                chosen_purchased.append(was_purchased)
                chosen_counts[chosen_arm] += 1

                if chosen_arm == displayed_arm:
                    match_count += 1

                # Always update on displayed item
                self.algorithm.update(displayed_arm, context, displayed_rew)

        n_eval = len(self.trials) - n_warmup
        match_rate = match_count / n_eval if n_eval > 0 else 0.0

        # Build summary using chosen-arm metrics
        from evaluation.metrics import ctr, gini_index, cumulative_reward, average_reward

        summary = {
            "algorithm":     self.algorithm.name,
            "reward":        self.reward_col,
            "n_trials":      len(chosen_rewards),
            "mean_reward":   round(float(np.mean(chosen_rewards)) if chosen_rewards else 0, 4),
            "total_reward":  round(float(np.sum(chosen_rewards)), 2),
            "ctr":           round(ctr(chosen_rewards), 4),
            "purchase_rate": round(float(np.mean(chosen_purchased)) if chosen_purchased else 0, 4),
            "gini_index":    round(gini_index(dict(chosen_counts)), 4),
            "match_rate":    round(match_rate, 4),
        }

        return {
            **summary,
            "reward_trace":            chosen_rewards,
            "arm_counts":              dict(chosen_counts),
            "avg_reward_curve":        average_reward(chosen_rewards).tolist(),
            "cumulative_reward_curve": cumulative_reward(chosen_rewards).tolist(),
        }


def run_all_experiments(
    trials: pd.DataFrame,
    algorithms: list,
    reward_cols: List[str],
    arm_pool: List[int],
    warmup_fraction: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> List[dict]:
    results = []
    total = len(algorithms) * len(reward_cols)
    print(f"\n{'='*60}")
    print(f"Running {total} experiments  "
          f"({len(algorithms)} algorithms × {len(reward_cols)} rewards)")
    print(f"  Warm-up: {warmup_fraction*100:.0f}%  |  Eval: {(1-warmup_fraction)*100:.0f}%")
    print(f"{'='*60}\n")

    for algo in algorithms:
        for rcol in reward_cols:
            sim = OfflineSimulator(
                trials=trials, algorithm=algo, reward_col=rcol,
                arm_pool=arm_pool, warmup_fraction=warmup_fraction, seed=seed,
            )
            result = sim.run(verbose=verbose)
            results.append(result)
            print(
                f"  {algo.name:<14} | {rcol:<18} | "
                f"MeanReward={result['mean_reward']:.4f}  "
                f"Purchase={result['purchase_rate']:.4f}  "
                f"Gini={result['gini_index']:.4f}  "
                f"Match={result['match_rate']:.4f}"
            )
    print()
    return results