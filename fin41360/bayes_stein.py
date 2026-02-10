"""
Bayes–Stein-style shrinkage utilities for FIN41360.

We implement:
- A Jorion (1986)-style Bayes–Stein shrinkage of the mean vector towards a
  simple target (here the cross-sectional grand mean), using the covariance
  matrix and sample length.
- A simple covariance shrinkage towards a scalar multiple of the identity,
  with a user-chosen shrinkage intensity.

EXPLAIN: There are many possible shrinkage schemes. We pick:
- Jorion-style mean shrinkage because it directly matches the course readings
  and explicitly trades off estimation error vs prior (grand mean).
- A mild convex combination with an identity target for the covariance because
  (i) our T >> N here, so only light regularisation is needed, and
  (ii) it improves conditioning without imposing a strong structural prior
  (such as constant correlation) that might be harder to justify empirically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BayesSteinResult:
    """Container for Bayes–Stein mean shrinkage output."""

    mu_bs: np.ndarray
    shrinkage_intensity: float
    target_mean: float


def bayes_stein_means(
    mu: np.ndarray,
    Sigma: np.ndarray,
    T: int,
) -> BayesSteinResult:
    """
    Jorion (1986) Bayes–Stein shrinkage of the mean vector towards the grand mean.

    Parameters
    ----------
    mu : np.ndarray, shape (N,)
        Sample mean vector (net returns).
    Sigma : np.ndarray, shape (N, N)
        Sample covariance matrix (net returns).
    T : int
        Sample length (number of time observations).

    Returns
    -------
    BayesSteinResult
        Contains shrunk means, shrinkage intensity, and target mean.

    Notes
    -----
    Following the spirit of Jorion (1986):
    - Target is the cross-sectional grand mean: m_bar.
    - Shrinkage intensity is proportional to the ratio of average variance to
      cross-sectional dispersion of means, adjusted for T and N.
    """
    mu = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)
    N = mu.shape[0]

    if T <= N + 2:
        # EXPLAIN: In small samples relative to dimension, the classic
        # Jorion formula can misbehave; in that case we skip shrinkage.
        return BayesSteinResult(mu_bs=mu.copy(), shrinkage_intensity=0.0, target_mean=float(mu.mean()))

    # Grand mean target
    m_bar = float(mu.mean())
    ones = np.ones(N)
    Sigma_inv = np.linalg.pinv(Sigma)

    # Cross-sectional dispersion term q = (mu - m_bar 1)' Σ^{-1} (mu - m_bar 1)
    diff = mu - m_bar * ones
    q = float(diff @ Sigma_inv @ diff)

    # Average variance s^2 = trace(Sigma) / N
    s2 = float(np.trace(Sigma)) / N

    # Jorion-style shrinkage factor; clip to [0, 1]
    # EXPLAIN: The (N+2)/(T - N - 2) term accounts for dimensionality vs sample size.
    num = (N + 2) * s2
    den = max(q, 1e-12)
    delta = num / (den * max(T - N - 2, 1))
    delta = float(np.clip(delta, 0.0, 1.0))

    mu_bs = (1.0 - delta) * mu + delta * m_bar * ones
    return BayesSteinResult(mu_bs=mu_bs, shrinkage_intensity=delta, target_mean=m_bar)


def shrink_covariance_identity(
    Sigma: np.ndarray,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """
    Shrink the covariance matrix towards a scalar multiple of the identity.

    Parameters
    ----------
    Sigma : np.ndarray, shape (N, N)
        Sample covariance matrix.
    shrinkage : float, default 0.1
        Shrinkage intensity λ in [0, 1]:
        Σ_bs = (1 - λ) Σ + λ * σ̄^2 I,
        where σ̄^2 is the average variance (trace(Σ)/N).

    Returns
    -------
    Sigma_bs : np.ndarray, shape (N, N)
        Shrunk covariance matrix.

    Justification
    -------------
    - Identity target (scaled by average variance) preserves the overall risk
      level while damping off-diagonal noise.
    - With monthly data from 1980–2025 (T ≈ 552) and only N = 30 industries,
      the sample covariance is reasonably well-estimated; we therefore choose
      a **modest** shrinkage λ (default 0.1) to improve conditioning without
      imposing a strong structural view on correlations.
    - Alternatives would include constant-correlation or Ledoit–Wolf-type
      optimally chosen λ; these are more complex to implement and may not
      materially change results in this relatively high T / low N setting.
    """
    Sigma = np.asarray(Sigma)
    N = Sigma.shape[0]
    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square.")

    lam = float(np.clip(shrinkage, 0.0, 1.0))
    avg_var = float(np.trace(Sigma)) / N
    target = avg_var * np.eye(N)
    Sigma_bs = (1.0 - lam) * Sigma + lam * target
    return Sigma_bs

