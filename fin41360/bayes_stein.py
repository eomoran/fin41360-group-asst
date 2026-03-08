"""
Bayes–Stein-style shrinkage utilities for FIN41360.

We implement:
- Canonical Jorion-style Bayes-Stein shrinkage of the mean vector towards an
  explicit target, using the sample covariance and sample length.
- A simple covariance shrinkage towards a scalar multiple of the identity,
  with a user-chosen shrinkage intensity.
- A Ledoit-Wolf covariance shrinkage estimator (data-driven shrinkage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BayesSteinResult:
    """Container for Bayes–Stein mean shrinkage output."""

    mu_bs: np.ndarray
    shrinkage_intensity: float
    target_mean: float
    target_kind: str


def bayes_stein_means(
    mu: np.ndarray,
    Sigma: np.ndarray,
    T: int,
    target: str = "grand_mean",
) -> BayesSteinResult:
    """
    Canonical Jorion-style Bayes-Stein shrinkage of the mean vector.

    Parameters
    ----------
    mu : np.ndarray, shape (N,)
        Sample mean vector (net returns).
    Sigma : np.ndarray, shape (N, N)
        Sample covariance matrix (net returns).
    T : int
        Sample length (number of time observations).
    target : {"grand_mean", "gmv"}, default "grand_mean"
        Shrinkage target for the prior mean.

    Returns
    -------
    BayesSteinResult
        Contains shrunk means, shrinkage intensity, and target mean.

    Notes
    -----
    Uses the standard Jorion-style shrinkage factor
    phi = (N + 2) / ((N + 2) + T * q),
    where q = (mu - mu_target)' Sigma^{-1} (mu - mu_target).
    """
    mu = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)
    N = mu.shape[0]
    target_kind = str(target).strip().lower()

    if T <= N + 2:
        fallback_target = float(mu.mean())
        return BayesSteinResult(
            mu_bs=mu.copy(),
            shrinkage_intensity=0.0,
            target_mean=fallback_target,
            target_kind=target_kind,
        )

    ones = np.ones(N)
    Sigma_inv = np.linalg.pinv(Sigma)

    if target_kind == "grand_mean":
        target_mean = float(mu.mean())
    elif target_kind == "gmv":
        w_gmv_unnorm = Sigma_inv @ ones
        denom = float(ones @ w_gmv_unnorm)
        if abs(denom) < 1e-12:
            raise ValueError("GMV target is undefined because the normalization denominator is near zero.")
        w_gmv = w_gmv_unnorm / denom
        target_mean = float(mu @ w_gmv)
    else:
        raise ValueError("target must be one of {'grand_mean', 'gmv'}")

    diff = mu - target_mean * ones
    q = float(diff @ Sigma_inv @ diff)
    phi = float(np.clip((N + 2) / ((N + 2) + T * max(q, 0.0)), 0.0, 1.0))

    mu_bs = (1.0 - phi) * mu + phi * target_mean * ones
    return BayesSteinResult(
        mu_bs=mu_bs,
        shrinkage_intensity=phi,
        target_mean=target_mean,
        target_kind=target_kind,
    )


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
    - Alternatives include constant-correlation targets and data-driven methods
      such as Ledoit-Wolf (implemented in `shrink_covariance_ledoit_wolf`).
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


def shrink_covariance_ledoit_wolf(
    returns_net: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Estimate covariance using Ledoit-Wolf shrinkage.

    Parameters
    ----------
    returns_net : np.ndarray, shape (T, N)
        Net returns matrix.

    Returns
    -------
    (Sigma_lw, lambda_lw)
        Sigma_lw : np.ndarray, shape (N, N)
            Ledoit-Wolf covariance estimate.
        lambda_lw : float
            Estimated shrinkage intensity from the fitted model.
    """
    try:
        from sklearn.covariance import LedoitWolf
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for Ledoit-Wolf covariance shrinkage. "
            "Install with: pip install scikit-learn"
        ) from e

    x = np.asarray(returns_net, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns_net must be a 2D array of shape (T, N).")
    if x.shape[0] < 2:
        raise ValueError("returns_net must contain at least 2 observations.")

    lw = LedoitWolf().fit(x)
    Sigma_lw = np.asarray(lw.covariance_, dtype=float)
    lambda_lw = float(getattr(lw, "shrinkage_", np.nan))
    return Sigma_lw, lambda_lw
