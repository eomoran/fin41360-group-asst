"""
Reusable workflow helpers for mean-variance frontier analysis.

These functions keep notebooks thin by centralizing repeated mechanics:
- GMV/tangency statistics
- frontier curve generation
- CML line generation
- in-sample vs out-of-sample Sharpe summaries
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .mv_frontier import efficient_frontier, gmv_weights, portfolio_stats, tangency_weights


def gmv_tan_stats(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> dict[str, Any]:
    """
    Compute GMV and tangency weights and summary stats for one frontier.
    """
    w_gmv = gmv_weights(Sigma)
    mu_gmv, vol_gmv = portfolio_stats(w_gmv, mu, Sigma)

    w_tan = tangency_weights(mu, Sigma, rf=rf)
    mu_tan, vol_tan = portfolio_stats(w_tan, mu, Sigma)
    sharpe_tan = (mu_tan - rf) / vol_tan if vol_tan > 0 else np.nan

    return {
        "weights": {"gmv": w_gmv, "tan": w_tan},
        "stats": {
            "gmv": {"mean": mu_gmv, "vol": vol_gmv, "excess_mean": mu_gmv - rf},
            "tan": {
                "mean": mu_tan,
                "vol": vol_tan,
                "excess_mean": mu_tan - rf,
                "sharpe": sharpe_tan,
            },
        },
    }


def build_frontier_curve(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_points: int = 200,
    mu_min: float | None = None,
    mu_max: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Build one frontier curve in (vol, mean) space.
    """
    means, vols, weights = efficient_frontier(
        mu, Sigma, n_points=n_points, mu_min=mu_min, mu_max=mu_max
    )
    return {"means": means, "vols": vols, "weights": weights}


def build_mu_range(mu_vectors: list[np.ndarray], pad: float = 0.001) -> tuple[float, float]:
    """
    Build a shared [mu_min, mu_max] range from one or more mean vectors.
    """
    mins = [float(np.min(v)) for v in mu_vectors]
    maxs = [float(np.max(v)) for v in mu_vectors]
    return min(mins) - pad, max(maxs) + pad


def build_cml(
    sharpe_tan: float,
    vol_max: float,
    rf: float = 0.0,
    n_points: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build CML points: mean = rf + sharpe_tan * vol.
    """
    vol = np.linspace(0.0, vol_max, n_points)
    mean = rf + sharpe_tan * vol
    return vol, mean


def summarize_sharpe_is_oos(
    returns_is: np.ndarray,
    returns_oos: np.ndarray,
    rf_is: float | np.ndarray = 0.0,
    rf_oos: float | np.ndarray = 0.0,
) -> dict[str, float]:
    """
    Compute in-sample and out-of-sample Sharpe ratios.
    """

    def _sharpe(rets: np.ndarray, rf_val: float | np.ndarray) -> float:
        excess = rets - rf_val
        vol = np.std(excess, ddof=1)
        return float(np.mean(excess) / vol) if vol > 0 else np.nan

    sr_is = _sharpe(np.asarray(returns_is), rf_is)
    sr_oos = _sharpe(np.asarray(returns_oos), rf_oos)
    return {"sr_is": sr_is, "sr_oos": sr_oos, "delta": sr_oos - sr_is}
