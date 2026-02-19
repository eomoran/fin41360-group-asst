"""
Section-level workflow entry points for FIN41360 notebooks.

Design goal: keep final notebooks mostly to descriptive function calls, with
data loading and plotting delegated to reusable module code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .bayes_stein import bayes_stein_means, shrink_covariance_identity
from .frontier_workflow import build_frontier_curve, build_mu_range, gmv_tan_stats
from .mv_frontier import compute_moments_from_gross


def run_scope2_industries_sample_vs_bs(
    ind_gross: pd.DataFrame,
    rf_gross: pd.Series,
    cov_shrink: float = 0.1,
    n_points: int = 200,
) -> dict[str, Any]:
    """
    Scope 2: 30-industry frontiers using sample and Bayes-Stein counterparts.

    Returns a dict with:
    - inputs: aligned sample metadata
    - models: parameter sets (mu, Sigma) for sample / BS-mean / BS-mean+cov
    - summary_tables: GMV/TAN table
    - plot_data: frontier curves and GMV/TAN points for plotting
    - diagnostics: shrinkage intensity and covariance eigenvalue diagnostics
    """
    if ind_gross.empty:
        raise ValueError("ind_gross is empty")
    if rf_gross.empty:
        raise ValueError("rf_gross is empty")

    common_idx = ind_gross.index.intersection(rf_gross.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between industry and risk-free series.")

    ind_aligned = ind_gross.loc[common_idx]
    rf_net = (rf_gross.loc[common_idx] - 1.0).astype(float)
    rf_mean = float(rf_net.mean())

    mu_sample, Sigma_sample = compute_moments_from_gross(ind_aligned.values)
    T = len(ind_aligned)
    bs = bayes_stein_means(mu_sample, Sigma_sample, T=T)
    mu_bs = bs.mu_bs
    Sigma_bs = shrink_covariance_identity(Sigma_sample, shrinkage=cov_shrink)

    models = {
        "sample": {"mu": mu_sample, "Sigma": Sigma_sample},
        "bs_mean": {"mu": mu_bs, "Sigma": Sigma_sample},
        "bs_mean_cov": {"mu": mu_bs, "Sigma": Sigma_bs},
    }

    stats = {label: gmv_tan_stats(m["mu"], m["Sigma"], rf=rf_mean) for label, m in models.items()}

    summary_rows = []
    for label, st in stats.items():
        summary_rows.append(
            {
                "label": label,
                "portfolio": "GMV",
                "mean": st["stats"]["gmv"]["mean"],
                "vol": st["stats"]["gmv"]["vol"],
                "excess_mean": st["stats"]["gmv"]["excess_mean"],
                "sharpe": st["stats"]["gmv"]["excess_mean"] / st["stats"]["gmv"]["vol"]
                if st["stats"]["gmv"]["vol"] > 0
                else np.nan,
            }
        )
        summary_rows.append(
            {
                "label": label,
                "portfolio": "TAN",
                "mean": st["stats"]["tan"]["mean"],
                "vol": st["stats"]["tan"]["vol"],
                "excess_mean": st["stats"]["tan"]["excess_mean"],
                "sharpe": st["stats"]["tan"]["sharpe"],
            }
        )
    summary_gmv_tan = pd.DataFrame(summary_rows)

    mu_targets = [
        mu_sample,
        mu_bs,
        np.array([stats["sample"]["stats"]["gmv"]["mean"], stats["sample"]["stats"]["tan"]["mean"]]),
        np.array([stats["bs_mean"]["stats"]["gmv"]["mean"], stats["bs_mean"]["stats"]["tan"]["mean"]]),
        np.array([stats["bs_mean_cov"]["stats"]["gmv"]["mean"], stats["bs_mean_cov"]["stats"]["tan"]["mean"]]),
    ]
    mu_min, mu_max = build_mu_range(mu_targets, pad=0.001)
    curves = {
        label: build_frontier_curve(m["mu"], m["Sigma"], n_points=n_points, mu_min=mu_min, mu_max=mu_max)
        for label, m in models.items()
    }

    points = {
        label: {
            "gmv": stats[label]["stats"]["gmv"],
            "tan": stats[label]["stats"]["tan"],
        }
        for label in models.keys()
    }

    eig_sample = np.linalg.eigvalsh(Sigma_sample)
    eig_bs = np.linalg.eigvalsh(Sigma_bs)

    return {
        "inputs": {
            "start": common_idx.min().strftime("%Y-%m"),
            "end": common_idx.max().strftime("%Y-%m"),
            "n_obs": int(len(common_idx)),
            "n_assets": int(ind_aligned.shape[1]),
            "rf_mean": rf_mean,
        },
        "models": models,
        "summary_tables": {"gmv_tan": summary_gmv_tan},
        "plot_data": {"curves": curves, "points": points},
        "diagnostics": {
            "bs_mean_shrinkage_intensity": bs.shrinkage_intensity,
            "bs_target_mean": bs.target_mean,
            "sample_cov_eig_min": float(np.min(eig_sample)),
            "sample_cov_eig_max": float(np.max(eig_sample)),
            "bs_cov_eig_min": float(np.min(eig_bs)),
            "bs_cov_eig_max": float(np.max(eig_bs)),
        },
    }


def run_scope3_industries_vs_stocks(
    ind_gross: pd.DataFrame,
    stocks_gross: pd.DataFrame,
    rf_gross: pd.Series,
    cov_shrink: float = 0.1,
    n_points: int = 200,
) -> dict[str, Any]:
    """
    Placeholder for Scope 3 workflow.
    """
    raise NotImplementedError("Scope 3 workflow scaffolded but not implemented yet.")


def run_scope4_industries_with_rf(
    ind_gross: pd.DataFrame,
    rf_gross: pd.Series,
    n_points: int = 200,
) -> dict[str, Any]:
    """
    Placeholder for Scope 4 workflow.
    """
    raise NotImplementedError("Scope 4 workflow scaffolded but not implemented yet.")


def run_scope5_industries_vs_ff(
    ind_gross: pd.DataFrame,
    ff3_excess: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    rf_gross: pd.Series,
    n_points: int = 200,
) -> dict[str, Any]:
    """
    Placeholder for Scope 5 workflow.
    """
    raise NotImplementedError("Scope 5 workflow scaffolded but not implemented yet.")


def run_scope6_ff_vs_proxies(
    ff3_excess: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    proxy_returns: pd.DataFrame,
    rf_gross: pd.Series,
    n_points: int = 200,
) -> dict[str, Any]:
    """
    Placeholder for Scope 6 workflow.
    """
    raise NotImplementedError("Scope 6 workflow scaffolded but not implemented yet.")


def run_scope7_is_oos_tests(
    ind_gross: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    rf_gross: pd.Series,
    end_is: str = "2002-12",
    start_oos: str = "2003-01",
    end_oos: str = "2025-12",
) -> dict[str, Any]:
    """
    Placeholder for Scope 7 workflow.
    """
    raise NotImplementedError("Scope 7 workflow scaffolded but not implemented yet.")
