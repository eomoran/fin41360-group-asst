"""
Closed-form mean–variance frontier utilities for FIN41360.

This module provides NumPy-based functions to:
- Compute sample means and covariance matrices of returns.
- Compute the global minimum-variance (GMV) portfolio.
- Compute the tangency portfolio for a given risk-free rate.
- Generate points on the unconstrained (short-selling allowed) MV frontier.

All functions assume *net* returns by default (e.g. 0.02 = 2%),
but convenience helpers are provided for gross returns (R = 1 + r).

EXPLAIN: Keeping all closed-form formulas in one place makes it easier
to cross-check against the lecture notes and adapt them later (e.g. to
Bayes–Stein inputs) without touching notebook code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FrontierCoefficients:
    """
    Store the classic A, B, C, D coefficients for the MV frontier.

    A = 1' Σ^{-1} 1
    B = 1' Σ^{-1} μ
    C = μ' Σ^{-1} μ
    D = AC - B^2
    """

    A: float
    B: float
    C: float
    D: float


def compute_moments_from_net(returns_net: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sample mean vector and covariance matrix from *net* returns.

    Parameters
    ----------
    returns_net : np.ndarray, shape (T, N)
        Net returns (e.g. 0.02 for 2%) for N assets over T periods.

    Returns
    -------
    mu : np.ndarray, shape (N,)
        Sample mean of net returns.
    Sigma : np.ndarray, shape (N, N)
        Sample covariance matrix of net returns.
    """
    if returns_net.ndim != 2:
        raise ValueError("returns_net must be 2D array of shape (T, N)")

    mu = returns_net.mean(axis=0)
    # rowvar=False => each column is a variable (asset)
    Sigma = np.cov(returns_net, rowvar=False, ddof=1)
    return mu, Sigma


def compute_moments_from_gross(returns_gross: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper to compute moments from *gross* returns (R = 1 + r).

    Parameters
    ----------
    returns_gross : np.ndarray, shape (T, N)
        Gross returns (R = 1 + r).

    Returns
    -------
    mu : np.ndarray, shape (N,)
        Sample mean of net returns.
    Sigma : np.ndarray, shape (N, N)
        Sample covariance of net returns.
    """
    returns_net = returns_gross - 1.0
    return compute_moments_from_net(returns_net)


def frontier_coefficients(mu: np.ndarray, Sigma: np.ndarray) -> Tuple[FrontierCoefficients, np.ndarray]:
    """
    Compute (A, B, C, D) and Σ^{-1} given mean vector and covariance matrix.

    Returns
    -------
    coeffs : FrontierCoefficients
    Sigma_inv : np.ndarray
        Inverse (or pseudo-inverse) of Σ, shape (N, N).
    """
    mu = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)

    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square (N x N)")
    if Sigma.shape[0] != mu.shape[0]:
        raise ValueError("Sigma and mu dimension mismatch")

    # Use pseudo-inverse for numerical stability in case of near-singularity.
    Sigma_inv = np.linalg.pinv(Sigma)
    ones = np.ones_like(mu)

    A = float(ones @ Sigma_inv @ ones)
    B = float(ones @ Sigma_inv @ mu)
    C = float(mu @ Sigma_inv @ mu)
    D = A * C - B * B

    if D <= 0:
        # EXPLAIN: In theory D > 0; if not, the inputs are pathological
        # (e.g. perfect collinearity). We still return the coefficients,
        # but downstream code should be cautious in interpreting results.
        print("Warning: D <= 0 in frontier_coefficients; frontier may be ill-defined.")

    return FrontierCoefficients(A=A, B=B, C=C, D=D), Sigma_inv


def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """
    Global minimum-variance (GMV) portfolio weights (no risk-free asset).

    Parameters
    ----------
    Sigma : np.ndarray, shape (N, N)
        Covariance matrix of net returns.

    Returns
    -------
    w_gmv : np.ndarray, shape (N,)
        GMV portfolio weights, summing to 1.
    """
    Sigma = np.asarray(Sigma)
    N = Sigma.shape[0]
    Sigma_inv = np.linalg.pinv(Sigma)
    ones = np.ones(N)
    w = Sigma_inv @ ones
    w = w / (ones @ w)
    return w


def tangency_weights(mu: np.ndarray, Sigma: np.ndarray, rf: float) -> np.ndarray:
    """
    Tangency portfolio weights for a given risk-free rate.

    Assumes unconstrained weights (short-selling allowed) and uses
    the classic formula proportional to Σ^{-1}(μ - r_f 1).

    Parameters
    ----------
    mu : np.ndarray, shape (N,)
        Mean net returns.
    Sigma : np.ndarray, shape (N, N)
        Covariance of net returns.
    rf : float
        Risk-free rate (net, same units as mu, e.g. monthly).

    Returns
    -------
    w_tan : np.ndarray, shape (N,)
        Tangency portfolio weights, summing to 1.
    """
    mu = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)
    N = mu.shape[0]

    Sigma_inv = np.linalg.pinv(Sigma)
    ones = np.ones(N)
    excess = mu - rf * ones

    w_unnorm = Sigma_inv @ excess
    w_sum = float(w_unnorm.sum())
    if np.isclose(w_sum, 0.0):
        raise ValueError("Tangency weights sum to ~0; check inputs (mu, rf, Sigma).")
    w = w_unnorm / w_sum
    return w


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> Tuple[float, float]:
    """
    Compute expected return and standard deviation for a given weight vector.

    Parameters
    ----------
    weights : np.ndarray, shape (N,)
        Portfolio weights.
    mu : np.ndarray, shape (N,)
        Mean net returns.
    Sigma : np.ndarray, shape (N, N)
        Covariance of net returns.

    Returns
    -------
    (mean_return, volatility)
        Both in the same units as mu (e.g. monthly). Volatility is the
        standard deviation of portfolio return; no annualisation is applied.
    """
    w = np.asarray(weights).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)

    mean_ret = float(w @ mu)
    var = float(w @ Sigma @ w)
    vol = float(np.sqrt(var)) if var >= 0 else np.nan
    return mean_ret, vol


def efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_points: int = 100,
    mu_min: Optional[float] = None,
    mu_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate points on the unconstrained MV frontier using closed-form formulas.

    Parameters
    ----------
    mu : np.ndarray, shape (N,)
        Mean net returns.
    Sigma : np.ndarray, shape (N, N)
        Covariance of net returns.
    n_points : int, default 100
        Number of target-mean points along the frontier.
    mu_min, mu_max : float, optional
        Minimum and maximum target mean returns. If None, they are chosen
        slightly beyond the min/max of the individual asset means.

    Returns
    -------
    target_means : np.ndarray, shape (n_points,)
    vols : np.ndarray, shape (n_points,)
    weights : np.ndarray, shape (n_points, N)
    """
    mu = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)
    N = mu.shape[0]

    coeffs, Sigma_inv = frontier_coefficients(mu, Sigma)
    ones = np.ones(N)

    if mu_min is None:
        mu_min = float(mu.min()) - 0.001
    if mu_max is None:
        mu_max = float(mu.max()) + 0.001

    target_means = np.linspace(mu_min, mu_max, n_points)
    weights = np.zeros((n_points, N))
    vols = np.zeros(n_points)

    A, B, C, D = coeffs.A, coeffs.B, coeffs.C, coeffs.D
    invSigma1 = Sigma_inv @ ones
    invSigmaMu = Sigma_inv @ mu

    for i, m in enumerate(target_means):
        # Closed-form weights for a given target mean m:
        alpha = (C - B * m) / D
        beta = (A * m - B) / D
        w = alpha * invSigma1 + beta * invSigmaMu
        weights[i, :] = w

        # Compute volatility for this portfolio
        _, vol = portfolio_stats(w, mu, Sigma)
        vols[i] = vol

    return target_means, vols, weights

