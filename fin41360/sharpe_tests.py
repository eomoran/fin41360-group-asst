"""
Sharpe ratio equality tests for FIN41360 Scope 7.

Implements:
- Jobson–Korkie (1981) test for equality of two Sharpe ratios (statistic as in
  Jorion (1985), footnote 20, p. 271). Assumes i.i.d. normal returns; same
  formula is used for two independent samples (e.g. in-sample vs out-of-sample).
- Ledoit–Wolf (2008) robust test using circular block bootstrap, which does
  not require i.i.d. or normality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _sharpe_ratio(rets: np.ndarray, rf: float = 0.0) -> float:
    """Sample Sharpe ratio (excess mean / vol). rets: net or excess returns, 1D."""
    rets = np.asarray(rets).reshape(-1)
    excess = rets - rf
    mu = np.mean(excess)
    sig = np.std(excess, ddof=1)
    if sig <= 0:
        return np.nan
    return float(mu / sig)


def _block_bootstrap_indices(T: int, block_size: int) -> np.ndarray:
    """Circular block bootstrap: return T indices into [0..T-1]."""
    n_blocks = int(np.ceil(T / block_size))
    starts = np.random.randint(0, T, size=n_blocks)
    idx = []
    for b in range(n_blocks):
        for j in range(block_size):
            if len(idx) >= T:
                break
            idx.append((starts[b] + j) % T)
    return np.array(idx[:T], dtype=int)


@dataclass
class JobsonKorkieResult:
    """Result of Jobson–Korkie test for H0: SR1 = SR2."""

    sharpe1: float
    sharpe2: float
    statistic: float  # z-statistic
    pvalue_two_sided: float


def jobson_korkie_test(
    returns1: np.ndarray,
    returns2: np.ndarray,
    rf1: float = 0.0,
    rf2: float = 0.0,
) -> JobsonKorkieResult:
    """
    Jobson–Korkie (1981) test for equality of two Sharpe ratios.

    Statistic as in Jorion (1985), footnote 20, p. 271. Assumes i.i.d. normal
    returns. For two independent samples (e.g. in-sample vs out-of-sample),
    the asymptotic variance of (SR1 - SR2) is V1 + V2 with
    V_i = (1/T_i) * (1 + 0.5*SR_i^2) under normality.

    Parameters
    ----------
    returns1, returns2 : np.ndarray, shape (T1,) and (T2,)
        Net (or excess) return series. If excess, set rf1=rf2=0.
    rf1, rf2 : float
        Risk-free rate (net) for each period; excess = returns - rf.

    Returns
    -------
    JobsonKorkieResult
    """
    r1 = np.asarray(returns1).reshape(-1)
    r2 = np.asarray(returns2).reshape(-1)
    T1, T2 = len(r1), len(r2)

    sr1 = _sharpe_ratio(r1, rf1)
    sr2 = _sharpe_ratio(r2, rf2)
    if np.isnan(sr1) or np.isnan(sr2):
        return JobsonKorkieResult(
            sharpe1=sr1, sharpe2=sr2, statistic=np.nan, pvalue_two_sided=np.nan
        )

    # Asymptotic variance of SR_i (under normality): (1/T_i)*(1 + 0.5*SR_i^2)
    V1 = (1.0 / T1) * (1.0 + 0.5 * sr1**2)
    V2 = (1.0 / T2) * (1.0 + 0.5 * sr2**2)
    se = np.sqrt(V1 + V2)
    if se <= 0:
        return JobsonKorkieResult(
            sharpe1=sr1, sharpe2=sr2, statistic=np.nan, pvalue_two_sided=np.nan
        )
    z = (sr1 - sr2) / se
    from scipy import stats

    pval = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return JobsonKorkieResult(
        sharpe1=sr1, sharpe2=sr2, statistic=float(z), pvalue_two_sided=float(pval)
    )


@dataclass
class LedoitWolfResult:
    """Result of Ledoit–Wolf (2008) bootstrap test for H0: SR1 = SR2."""

    sharpe1: float
    sharpe2: float
    difference: float  # SR1 - SR2
    ci_low: float
    ci_high: float
    pvalue_two_sided: float  # reject H0 if 0 not in CI
    n_boot: int


def ledoit_wolf_test(
    returns1: np.ndarray,
    returns2: np.ndarray,
    rf1: float = 0.0,
    rf2: float = 0.0,
    n_boot: int = 1000,
    block_size: Optional[int] = None,
    confidence: float = 0.95,
) -> LedoitWolfResult:
    """
    Ledoit–Wolf (2008) robust test for equality of two Sharpe ratios.

    Uses circular block bootstrap so that the test does not require i.i.d. or
    normality. Reject H0: SR1 = SR2 if zero is not in the bootstrap CI for
    (SR1 - SR2).

    Parameters
    ----------
    returns1, returns2 : np.ndarray, shape (T1,) and (T2,)
        Net (or excess) return series.
    rf1, rf2 : float
        Risk-free rate (net) for each period.
    n_boot : int
        Number of bootstrap replications.
    block_size : int, optional
        Block length for circular block bootstrap. If None, use default
        (max(1, int(sqrt(T)) for each series; for different T1/T2 we use
        a common choice based on the shorter series).
    confidence : float
        Nominal coverage for the bootstrap CI (e.g. 0.95).

    Returns
    -------
    LedoitWolfResult
    """
    r1 = np.asarray(returns1).reshape(-1)
    r2 = np.asarray(returns2).reshape(-1)
    T1, T2 = len(r1), len(r2)

    sr1 = _sharpe_ratio(r1, rf1)
    sr2 = _sharpe_ratio(r2, rf2)
    delta = sr1 - sr2

    if block_size is None:
        block_size = max(1, int(np.sqrt(min(T1, T2))))

    deltas = np.zeros(n_boot)
    for b in range(n_boot):
        # Circular block bootstrap for each series independently
        i1 = _block_bootstrap_indices(T1, block_size)
        i2 = _block_bootstrap_indices(T2, block_size)
        sr1_b = _sharpe_ratio(r1[i1], rf1)
        sr2_b = _sharpe_ratio(r2[i2], rf2)
        deltas[b] = sr1_b - sr2_b

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(deltas, 100.0 * alpha / 2))
    ci_high = float(np.percentile(deltas, 100.0 * (1.0 - alpha / 2)))
    # Two-sided p-value: proportion of bootstrap distribution on opposite side of 0 from point estimate
    if delta >= 0:
        pval = 2.0 * np.mean(deltas <= 0)
    else:
        pval = 2.0 * np.mean(deltas >= 0)
    pval = min(1.0, max(0.0, float(pval)))

    return LedoitWolfResult(
        sharpe1=sr1,
        sharpe2=sr2,
        difference=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        pvalue_two_sided=pval,
        n_boot=n_boot,
    )


@dataclass
class FrontierReplicationResult:
    """
    Result of checking whether portfolio weights can be replicated as
    (1 - alpha) * w_tan_oos + alpha * w_gmv_oos (on the OOS frontier).
    """

    alpha: float
    r_squared: float
    residual_norm: float
    w_fit: np.ndarray


def frontier_replication_alpha(
    w_in_sample: np.ndarray,
    w_tan_oos: np.ndarray,
    w_gmv_oos: np.ndarray,
) -> FrontierReplicationResult:
    """
    Solve for alpha such that w_in_sample ≈ (1 - alpha) * w_tan_oos + alpha * w_gmv_oos
    in least-squares sense, and report R² fit.

    Used to assess whether an in-sample portfolio "remains on the OOS frontier"
    (replicable as a combination of OOS tangency and OOS GMV). If R² is close to 1,
    the portfolio is approximately on the OOS efficient frontier.

    Parameters
    ----------
    w_in_sample : np.ndarray, shape (N,)
        Portfolio weights to replicate (e.g. in-sample TAN).
    w_tan_oos : np.ndarray, shape (N,)
        Tangency portfolio weights estimated on OOS data.
    w_gmv_oos : np.ndarray, shape (N,)
        GMV portfolio weights estimated on OOS data.

    Returns
    -------
    FrontierReplicationResult
        alpha : (1 - alpha) * TAN + alpha * GMV; residual_norm = ||w_in_sample - w_fit||;
        r_squared = 1 - (residual_norm² / SS_tot) with SS_tot = ||w_in_sample - mean(w)||².
    """
    w_in_sample = np.asarray(w_in_sample).reshape(-1)
    w_tan_oos = np.asarray(w_tan_oos).reshape(-1)
    w_gmv_oos = np.asarray(w_gmv_oos).reshape(-1)
    if not (w_in_sample.shape == w_tan_oos.shape == w_gmv_oos.shape):
        raise ValueError("w_in_sample, w_tan_oos, w_gmv_oos must have the same length")

    d = w_tan_oos - w_gmv_oos
    y = w_in_sample - w_gmv_oos
    dtd = float(np.dot(d, d))
    if dtd <= 0:
        # TAN and GMV coincide; any alpha gives same w_fit
        one_minus_alpha = 0.0
        alpha = 1.0
    else:
        one_minus_alpha = float(np.dot(d, y) / dtd)
        alpha = 1.0 - one_minus_alpha

    w_fit = one_minus_alpha * w_tan_oos + alpha * w_gmv_oos
    residual = w_in_sample - w_fit
    residual_norm_sq = float(np.dot(residual, residual))
    residual_norm = np.sqrt(residual_norm_sq)

    w_centered = w_in_sample - np.mean(w_in_sample)
    ss_tot = float(np.dot(w_centered, w_centered))
    if ss_tot <= 0:
        r_squared = np.nan
    else:
        r_squared = float(1.0 - (residual_norm_sq / ss_tot))

    return FrontierReplicationResult(
        alpha=alpha,
        r_squared=r_squared,
        residual_norm=residual_norm,
        w_fit=w_fit,
    )
