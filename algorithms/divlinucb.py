"""
divlinucb.py
------------
Diversity-Boosting Linear UCB (DivLinUCB).

Reference:
  Semenov et al. (2022). Diversity in news recommendations using
  contextual bandits. Expert Systems With Applications, 195, 116478.

Algorithm 1 from the paper, adapted for e-commerce:
  - Arms = items (products)
  - At each trial the expected payoff is penalised by a cost term that
    grows with how frequently an item has already been recommended.

Selection score (Eq. 3–4 from paper):
                       theta_a^T f_{t,a}
  D_{t,a} = ─────────────────────────────────────────────  +  alpha * UCB_term
             1 + beta * (rc_a / T_hat)^n

where
  rc_a   = number of times item a has been recommended so far
  T_hat  = total number of recommendations made so far
  beta   = diversity strength  (beta=0 → identical to LinUCB)
  n      = exponent (set to 2 following the paper)
"""

import numpy as np
from typing import Dict


class DivLinUCB:
    """
    Diversity-boosting LinUCB (Semenov et al. 2022), adapted for
    e-commerce recommendation.

    Parameters
    ----------
    n_features : int
        Dimensionality of the context vector.
    alpha : float
        UCB exploration parameter (same role as in LinUCB).
    beta : float
        Diversity strength.  beta=0 collapses to standard LinUCB.
        Larger beta → stronger diversity, lower CTR trade-off.
    n_exp : int
        Exponent applied to the normalised recommendation count (paper uses 2).
    name : str
        Label for plots and logs.
    """

    def __init__(
        self,
        n_features: int,
        alpha: float = 2.62,
        beta: float = 100.0,
        n_exp: int = 2,
        name: str = "DivLinUCB",
    ):
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.n_exp = n_exp
        self.name = name

        # Per-arm covariance matrices and reward accumulators (like LinUCB)
        self._X: Dict[int, np.ndarray] = {}
        self._b: Dict[int, np.ndarray] = {}

        # Diversity tracking
        self._rc: Dict[int, int] = {}   # recommendation count per arm
        self._T_hat: int = 1            # total recommendations (start at 1 to avoid /0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_arm(self, context: np.ndarray, available_arms: list) -> int:
        """
        Choose arm with the highest diversity-adjusted UCB score (Eq. 3–4).

        Parameters
        ----------
        context       : feature vector for the current trial
        available_arms: candidate arm IDs

        Returns
        -------
        int : selected arm ID
        """
        f = context.reshape(-1, 1)
        best_arm, best_score = None, -np.inf

        for arm in available_arms:
            self._init_arm(arm)
            X_inv = np.linalg.inv(self._X[arm])
            theta = X_inv @ self._b[arm]

            exploit = float((theta.T @ f).item())
            explore = self.alpha * float(np.sqrt((f.T @ X_inv @ f).item()))

            # Cost term: penalises over-recommended arms (Eq. 4)
            rc_norm = self._rc[arm] / self._T_hat
            cost = 1.0 + self.beta * (rc_norm ** self.n_exp)

            # Diversity-adjusted score (Eq. 3): scale exploitation by cost
            score = exploit / cost + explore

            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update arm parameters and diversity counters.

        Parameters
        ----------
        arm     : arm ID that was played
        context : feature vector used at selection time
        reward  : observed scalar reward
        """
        self._init_arm(arm)
        f = context.reshape(-1, 1)

        # LinUCB update
        self._X[arm] += f @ f.T
        self._b[arm] += reward * f

        # Diversity counter update (lines 12–13 of Algorithm 1)
        self._rc[arm] += 1
        self._T_hat += 1

    def reset(self) -> None:
        """Clear all state (for fresh experiment runs)."""
        self._X.clear()
        self._b.clear()
        self._rc.clear()
        self._T_hat = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_arm(self, arm: int) -> None:
        if arm not in self._X:
            self._X[arm] = np.eye(self.n_features)
            self._b[arm] = np.zeros((self.n_features, 1))
            self._rc[arm] = 0
