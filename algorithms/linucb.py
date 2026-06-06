"""
linucb.py
---------
Linear Upper Confidence Bound (LinUCB) contextual bandit.

Reference:
  Li et al. (2010). A contextual-bandit approach to personalized news
  article recommendation. WWW 2010.

Each arm 'a' maintains:
  X_a  — feature co-variance matrix  (d×d), initialised to identity
  b_a  — reward-weighted feature sum (d×1), initialised to zeros
  theta_a = X_a^{-1} b_a            (estimated coefficient vector)

At each trial the algorithm selects the arm that maximises:
  p_{t,a} = theta_a^T f_{t,a}  +  alpha * sqrt(f_{t,a}^T X_a^{-1} f_{t,a})
             [exploitation]            [exploration]
"""

import numpy as np
from typing import Dict


class LinUCB:
    """
    Disjoint LinUCB: each arm has its own independent parameter set.

    Parameters
    ----------
    n_features : int
        Dimensionality of the context (feature) vector.
    alpha : float
        Exploration parameter.  Higher → more exploration.
        Li et al. suggest alpha = 1 + sqrt(ln(2/delta)/2) for a
        (1 - delta) confidence bound.  alpha=2.62 ↔ delta=0.01.
    name : str
        Label used in plots/logs.
    """

    def __init__(self, n_features: int, alpha: float = 2.62, name: str = "LinUCB"):
        self.n_features = n_features
        self.alpha = alpha
        self.name = name

        # Per-arm state — lazily initialised on first encounter
        self._X: Dict[int, np.ndarray] = {}   # d×d covariance
        self._b: Dict[int, np.ndarray] = {}   # d×1 reward accumulator

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_arm(self, context: np.ndarray, available_arms: list) -> int:
        """
        Choose the arm with the highest UCB score.

        Parameters
        ----------
        context       : feature vector for the current trial (length n_features)
        available_arms: list of arm IDs to choose from

        Returns
        -------
        int : selected arm ID
        """
        f = context.reshape(-1, 1)          # (d, 1)
        best_arm, best_score = None, -np.inf

        for arm in available_arms:
            self._init_arm(arm)
            X_inv = np.linalg.inv(self._X[arm])
            theta = X_inv @ self._b[arm]

            exploit = float((theta.T @ f).item())
            explore = self.alpha * float(np.sqrt((f.T @ X_inv @ f).item()))
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update arm parameters after observing a reward.

        Parameters
        ----------
        arm     : arm ID that was played
        context : feature vector used at selection time
        reward  : observed scalar reward
        """
        self._init_arm(arm)
        f = context.reshape(-1, 1)
        self._X[arm] += f @ f.T
        self._b[arm] += reward * f

    def reset(self) -> None:
        """Clear all learned parameters (for multiple experiment runs)."""
        self._X.clear()
        self._b.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_arm(self, arm: int) -> None:
        """Lazily initialise identity matrix and zero vector for a new arm."""
        if arm not in self._X:
            self._X[arm] = np.eye(self.n_features)
            self._b[arm] = np.zeros((self.n_features, 1))
