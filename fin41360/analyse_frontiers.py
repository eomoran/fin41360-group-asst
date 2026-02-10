"""
Analysis helpers for comparing multiple mean–variance frontiers.

This module focuses on **question 2(c)** for the 30-industry portfolios:
- Given several frontiers (sample, Bayes–Stein mean only, Bayes–Stein mean+cov),
  compute GMV and tangency portfolio statistics for each.
- Produce a comparison table and simple, rule-based commentary about changes
  in Sharpe ratios and risk/return trade-offs.

EXPLAIN: Keeping this logic in a small module makes it easy to re-use the same
analysis for stocks or factors later without duplicating notebook code.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .mv_frontier import gmv_weights, tangency_weights, portfolio_stats


def summarise_gmv_tan(
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float,
    label: str,
) -> pd.DataFrame:
    """
    Compute GMV and tangency statistics for a single (mu, Sigma) pair.

    Returns a 2-row DataFrame indexed by ['GMV', 'TAN'] with columns:
    ['label', 'mean', 'vol', 'excess_mean', 'sharpe'].
    """
    mu = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)

    w_gmv = gmv_weights(Sigma)
    mu_gmv, vol_gmv = portfolio_stats(w_gmv, mu, Sigma)

    w_tan = tangency_weights(mu, Sigma, rf)
    mu_tan, vol_tan = portfolio_stats(w_tan, mu, Sigma)

    res = pd.DataFrame(
        index=pd.Index(["GMV", "TAN"], name="portfolio"),
        columns=["label", "mean", "vol", "excess_mean", "sharpe"],
        dtype=float,
    )
    res["label"] = label
    res.loc["GMV", ["mean", "vol"]] = [mu_gmv, vol_gmv]
    res.loc["TAN", ["mean", "vol"]] = [mu_tan, vol_tan]

    # Excess means and Sharpe ratios (monthly)
    res["excess_mean"] = res["mean"] - rf
    res["sharpe"] = res["excess_mean"] / res["vol"]
    return res


def compare_frontiers(
    frontiers: Dict[str, Tuple[np.ndarray, np.ndarray]],
    rf: float,
) -> pd.DataFrame:
    """
    Summarise GMV and tangency statistics for multiple frontiers.

    Parameters
    ----------
    frontiers : dict
        Mapping from label -> (mu, Sigma).
        Example keys: 'sample', 'bs_mean', 'bs_mean_cov'.
    rf : float
        Risk-free rate (net, monthly).

    Returns
    -------
    DataFrame
        Multi-row summary for all frontiers and both GMV/TAN.
    """
    frames = []
    for label, (mu, Sigma) in frontiers.items():
        frames.append(summarise_gmv_tan(mu, Sigma, rf, label))
    summary = pd.concat(frames).reset_index()
    return summary


def print_sharpe_comparison(summary: pd.DataFrame, base_label: str = "sample") -> None:
    """
    Print simple rule-based commentary comparing Sharpe ratios across frontiers.

    Parameters
    ----------
    summary : DataFrame
        Output of compare_frontiers().
    base_label : str, default 'sample'
        Which frontier is treated as the baseline for comparisons.
    """
    # Focus on tangency portfolios for Sharpe comparisons.
    tan = summary[summary["portfolio"] == "TAN"].set_index("label")
    if base_label not in tan.index:
        print(f"Baseline label '{base_label}' not found in summary; skipping comparison.")
        return

    base_sharpe = tan.loc[base_label, "sharpe"]
    for label, row in tan.iterrows():
        if label == base_label:
            continue
        diff = row["sharpe"] - base_sharpe
        if abs(diff) < 0.02:
            comment = "Sharpe ratio is very similar to the sample-based frontier."
        elif diff > 0:
            comment = "Sharpe ratio is noticeably higher than the sample-based frontier (beneficial shrinkage)."
        else:
            comment = "Sharpe ratio is noticeably lower than the sample-based frontier (shrinkage may be too aggressive or mis-specified)."
        print(f"[{label}] Tangency Sharpe change vs {base_label}: {diff:.3f} → {comment}")

