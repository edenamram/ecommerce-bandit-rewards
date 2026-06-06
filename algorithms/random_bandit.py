"""
random_bandit.py
----------------
Uniformly random arm selection — used as the sanity-check baseline.
"""

import numpy as np


class RandomBandit:
    """
    Picks an arm uniformly at random from the available set.
    No learning, no context — pure random baseline.
    """

    def __init__(self, name: str = "Random"):
        self.name = name

    def select_arm(self, context: np.ndarray, available_arms: list) -> int:
        """Return a random arm from available_arms (ignores context)."""
        return np.random.choice(available_arms)

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """No-op: random bandit does not learn."""
        pass

    def reset(self) -> None:
        """No-op."""
        pass
