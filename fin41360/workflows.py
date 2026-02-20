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
from .frontier_workflow import (
    build_frontier_curve,
    build_mu_range,
    gmv_tan_stats,
    summarize_sharpe_is_oos,
)
from .mv_frontier import (
    compute_moments_from_gross,
    compute_moments_from_net,
    gmv_weights,
    portfolio_stats,
    tangency_weights,
)
from .sharpe_tests import (
    frontier_replication_alpha,
    jobson_korkie_test,
    ledoit_wolf_test,
)


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
    Scope 7: in-sample / out-of-sample frontier stability and Sharpe tests.

    This function is self-contained and order-independent: all intermediate
    series needed for JK/LW tests and OOS frontier checks are computed within
    the function, avoiding notebook cell-order fragility.
    """
    if ind_gross.empty:
        raise ValueError("ind_gross is empty")
    if ff5_excess.empty:
        raise ValueError("ff5_excess is empty")
    if rf_gross.empty:
        raise ValueError("rf_gross is empty")

    # ---------------------------
    # 30 industries + risk-free
    # ---------------------------
    ind_idx = ind_gross.index.intersection(rf_gross.index).sort_values()
    if len(ind_idx) == 0:
        raise ValueError("No overlapping dates between ind_gross and rf_gross.")

    ind_net = (ind_gross.loc[ind_idx] - 1.0).astype(float).values
    rf_net = (rf_gross.loc[ind_idx] - 1.0).astype(float).values

    ind_is_mask = ind_idx <= pd.Timestamp(end_is).to_period("M").to_timestamp("M")
    ind_oos_mask = (ind_idx >= pd.Timestamp(start_oos).to_period("M").to_timestamp("M")) & (
        ind_idx <= pd.Timestamp(end_oos).to_period("M").to_timestamp("M")
    )

    ind_idx_is = ind_idx[ind_is_mask]
    ind_idx_oos = ind_idx[ind_oos_mask]
    if len(ind_idx_is) == 0 or len(ind_idx_oos) == 0:
        raise ValueError("Industry IS/OOS split produced an empty period.")

    ind_net_is = ind_net[ind_is_mask]
    ind_net_oos = ind_net[ind_oos_mask]
    rf_is_net = rf_net[ind_is_mask]
    rf_oos_net = rf_net[ind_oos_mask]

    mu_ind_is, Sigma_ind_is = compute_moments_from_net(ind_net_is)
    mu_ind_oos, Sigma_ind_oos = compute_moments_from_net(ind_net_oos)
    rf_ind_is = float(np.mean(rf_is_net))
    rf_ind_oos = float(np.mean(rf_oos_net))

    w_tan_ind = tangency_weights(mu_ind_is, Sigma_ind_is, rf_ind_is)
    w_gmv_ind = gmv_weights(Sigma_ind_is)

    mu_tan_ind_is, vol_tan_ind_is = portfolio_stats(w_tan_ind, mu_ind_is, Sigma_ind_is)
    mu_gmv_ind_is, vol_gmv_ind_is = portfolio_stats(w_gmv_ind, mu_ind_is, Sigma_ind_is)

    r_tan_ind_is = ind_net_is @ w_tan_ind
    r_gmv_ind_is = ind_net_is @ w_gmv_ind
    r_tan_ind_oos = ind_net_oos @ w_tan_ind
    r_gmv_ind_oos = ind_net_oos @ w_gmv_ind

    # ---------------------------
    # FF5 excess returns
    # ---------------------------
    ff5_idx = ff5_excess.index.sort_values()
    ff5_arr = ff5_excess.loc[ff5_idx].astype(float).values

    ff5_is_mask = ff5_idx <= pd.Timestamp(end_is).to_period("M").to_timestamp("M")
    ff5_oos_mask = (ff5_idx >= pd.Timestamp(start_oos).to_period("M").to_timestamp("M")) & (
        ff5_idx <= pd.Timestamp(end_oos).to_period("M").to_timestamp("M")
    )

    ff5_idx_is = ff5_idx[ff5_is_mask]
    ff5_idx_oos = ff5_idx[ff5_oos_mask]
    if len(ff5_idx_is) == 0 or len(ff5_idx_oos) == 0:
        raise ValueError("FF5 IS/OOS split produced an empty period.")

    ff5_is_arr = ff5_arr[ff5_is_mask]
    ff5_oos_arr = ff5_arr[ff5_oos_mask]

    mu_ff5_is, Sigma_ff5_is = compute_moments_from_net(ff5_is_arr)
    mu_ff5_oos, Sigma_ff5_oos = compute_moments_from_net(ff5_oos_arr)

    w_tan_ff5 = tangency_weights(mu_ff5_is, Sigma_ff5_is, 0.0)
    w_gmv_ff5 = gmv_weights(Sigma_ff5_is)

    mu_tan_ff5_is, vol_tan_ff5_is = portfolio_stats(w_tan_ff5, mu_ff5_is, Sigma_ff5_is)
    mu_gmv_ff5_is, vol_gmv_ff5_is = portfolio_stats(w_gmv_ff5, mu_ff5_is, Sigma_ff5_is)

    r_tan_ff5_is = ff5_is_arr @ w_tan_ff5
    r_gmv_ff5_is = ff5_is_arr @ w_gmv_ff5
    r_tan_ff5_oos = ff5_oos_arr @ w_tan_ff5
    r_gmv_ff5_oos = ff5_oos_arr @ w_gmv_ff5

    # IS/OOS Sharpe summaries
    sr_ind_tan = summarize_sharpe_is_oos(r_tan_ind_is, r_tan_ind_oos, rf_is=rf_is_net, rf_oos=rf_oos_net)
    sr_ind_gmv = summarize_sharpe_is_oos(r_gmv_ind_is, r_gmv_ind_oos, rf_is=rf_is_net, rf_oos=rf_oos_net)
    sr_ff5_tan = summarize_sharpe_is_oos(r_tan_ff5_is, r_tan_ff5_oos, rf_is=0.0, rf_oos=0.0)
    sr_ff5_gmv = summarize_sharpe_is_oos(r_gmv_ff5_is, r_gmv_ff5_oos, rf_is=0.0, rf_oos=0.0)

    sharpe_rows = [
        {"portfolio": "30 ind TAN", **sr_ind_tan},
        {"portfolio": "30 ind GMV", **sr_ind_gmv},
        {"portfolio": "FF5 TAN", **sr_ff5_tan},
        {"portfolio": "FF5 GMV", **sr_ff5_gmv},
    ]
    sharpe_df = pd.DataFrame(sharpe_rows)

    # JK and LW tests
    portfolio_tests = [
        ("30 ind TAN", r_tan_ind_is, r_tan_ind_oos, rf_ind_is, rf_ind_oos),
        ("30 ind GMV", r_gmv_ind_is, r_gmv_ind_oos, rf_ind_is, rf_ind_oos),
        ("FF5 TAN", r_tan_ff5_is, r_tan_ff5_oos, 0.0, 0.0),
        ("FF5 GMV", r_gmv_ff5_is, r_gmv_ff5_oos, 0.0, 0.0),
    ]

    jk_rows = []
    lw_rows = []
    for name, r_is, r_oos, rf_is_val, rf_oos_val in portfolio_tests:
        jk = jobson_korkie_test(r_is, r_oos, rf1=rf_is_val, rf2=rf_oos_val)
        lw = ledoit_wolf_test(r_is, r_oos, rf1=rf_is_val, rf2=rf_oos_val, n_boot=2000)
        jk_rows.append(
            {
                "portfolio": name,
                "sr_is": jk.sharpe1,
                "sr_oos": jk.sharpe2,
                "z_stat": jk.statistic,
                "pvalue": jk.pvalue_two_sided,
            }
        )
        lw_rows.append(
            {
                "portfolio": name,
                "sr_is": lw.sharpe1,
                "sr_oos": lw.sharpe2,
                "diff": lw.difference,
                "ci_low": lw.ci_low,
                "ci_high": lw.ci_high,
                "pvalue": lw.pvalue_two_sided,
            }
        )

    jk_df = pd.DataFrame(jk_rows)
    lw_df = pd.DataFrame(lw_rows)

    # OOS frontier checks
    w_tan_ind_oos = tangency_weights(mu_ind_oos, Sigma_ind_oos, rf_ind_oos)
    w_gmv_ind_oos = gmv_weights(Sigma_ind_oos)
    w_tan_ff5_oos = tangency_weights(mu_ff5_oos, Sigma_ff5_oos, 0.0)
    w_gmv_ff5_oos = gmv_weights(Sigma_ff5_oos)

    rep_rows = []
    for name, w_is, w_tan_oos, w_gmv_oos in [
        ("30 ind TAN", w_tan_ind, w_tan_ind_oos, w_gmv_ind_oos),
        ("30 ind GMV", w_gmv_ind, w_tan_ind_oos, w_gmv_ind_oos),
        ("FF5 TAN", w_tan_ff5, w_tan_ff5_oos, w_gmv_ff5_oos),
        ("FF5 GMV", w_gmv_ff5, w_tan_ff5_oos, w_gmv_ff5_oos),
    ]:
        rep = frontier_replication_alpha(w_is, w_tan_oos, w_gmv_oos)
        rep_rows.append(
            {
                "portfolio": name,
                "alpha": rep.alpha,
                "r_squared": rep.r_squared,
                "residual_norm": rep.residual_norm,
            }
        )
    frontier_rep_df = pd.DataFrame(rep_rows)

    # In-sample point table for report convenience
    is_rows = [
        {
            "portfolio": "30 ind TAN",
            "asset_set": "industries",
            "mean_is": mu_tan_ind_is,
            "vol_is": vol_tan_ind_is,
            "sharpe_is": sr_ind_tan["sr_is"],
        },
        {
            "portfolio": "30 ind GMV",
            "asset_set": "industries",
            "mean_is": mu_gmv_ind_is,
            "vol_is": vol_gmv_ind_is,
            "sharpe_is": sr_ind_gmv["sr_is"],
        },
        {
            "portfolio": "FF5 TAN",
            "asset_set": "ff5_excess",
            "mean_is": mu_tan_ff5_is,
            "vol_is": vol_tan_ff5_is,
            "sharpe_is": sr_ff5_tan["sr_is"],
        },
        {
            "portfolio": "FF5 GMV",
            "asset_set": "ff5_excess",
            "mean_is": mu_gmv_ff5_is,
            "vol_is": vol_gmv_ff5_is,
            "sharpe_is": sr_ff5_gmv["sr_is"],
        },
    ]
    is_points_df = pd.DataFrame(is_rows)

    return {
        "inputs": {
            "end_is": end_is,
            "start_oos": start_oos,
            "end_oos": end_oos,
            "industry_is_start": ind_idx_is.min().strftime("%Y-%m"),
            "industry_is_end": ind_idx_is.max().strftime("%Y-%m"),
            "industry_oos_start": ind_idx_oos.min().strftime("%Y-%m"),
            "industry_oos_end": ind_idx_oos.max().strftime("%Y-%m"),
            "ff5_is_start": ff5_idx_is.min().strftime("%Y-%m"),
            "ff5_is_end": ff5_idx_is.max().strftime("%Y-%m"),
            "ff5_oos_start": ff5_idx_oos.min().strftime("%Y-%m"),
            "ff5_oos_end": ff5_idx_oos.max().strftime("%Y-%m"),
            "n_ind_is": int(len(ind_idx_is)),
            "n_ind_oos": int(len(ind_idx_oos)),
            "n_ff5_is": int(len(ff5_idx_is)),
            "n_ff5_oos": int(len(ff5_idx_oos)),
            "rf_ind_is_mean": rf_ind_is,
            "rf_ind_oos_mean": rf_ind_oos,
        },
        "summary_tables": {
            "is_points": is_points_df,
            "sharpe_is_oos": sharpe_df,
            "jk_tests": jk_df,
            "lw_tests": lw_df,
            "frontier_replication": frontier_rep_df,
        },
        "plot_data": {
            "returns": {
                "r_tan_ind_is": r_tan_ind_is,
                "r_tan_ind_oos": r_tan_ind_oos,
                "r_gmv_ind_is": r_gmv_ind_is,
                "r_gmv_ind_oos": r_gmv_ind_oos,
                "r_tan_ff5_is": r_tan_ff5_is,
                "r_tan_ff5_oos": r_tan_ff5_oos,
                "r_gmv_ff5_is": r_gmv_ff5_is,
                "r_gmv_ff5_oos": r_gmv_ff5_oos,
            },
            "weights": {
                "w_tan_ind_is": w_tan_ind,
                "w_gmv_ind_is": w_gmv_ind,
                "w_tan_ff5_is": w_tan_ff5,
                "w_gmv_ff5_is": w_gmv_ff5,
                "w_tan_ind_oos": w_tan_ind_oos,
                "w_gmv_ind_oos": w_gmv_ind_oos,
                "w_tan_ff5_oos": w_tan_ff5_oos,
                "w_gmv_ff5_oos": w_gmv_ff5_oos,
            },
        },
        "diagnostics": {
            "note": "Scope 7 workflow computes all dependent variables internally to avoid notebook order fragility.",
        },
    }
