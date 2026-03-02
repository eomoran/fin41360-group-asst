"""
Section-level workflow entry points for FIN41360 notebooks.

Design goal: keep final notebooks mostly to descriptive function calls, with
data loading and plotting delegated to reusable module code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bayes_stein import bayes_stein_means, shrink_covariance_identity
from .frontier_workflow import (
    build_cml,
    build_frontier_curve,
    build_mu_range,
    gmv_tan_stats,
    summarize_sharpe_is_oos,
)
from .stock_data import get_common_sample_period
from .mv_frontier import (
    compute_moments_from_gross,
    compute_moments_from_net,
    gmv_weights,
    portfolio_stats,
    tangency_weights_constrained,
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
    tan_return_mult: float = 1.2,
    n_points: int = 1200,
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
                "universe": "industries",
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
                "universe": "industries",
                "portfolio": "TAN",
                "mean": st["stats"]["tan"]["mean"],
                "vol": st["stats"]["tan"]["vol"],
                "excess_mean": st["stats"]["tan"]["excess_mean"],
                "sharpe": st["stats"]["tan"]["sharpe"],
            }
        )
    summary_gmv_tan = pd.DataFrame(summary_rows)

    max_tan_mean = max(float(stats[k]["stats"]["tan"]["mean"]) for k in ("sample", "bs_mean", "bs_mean_cov"))
    mu_min = 0.0
    mu_max = max(0.001, tan_return_mult * max_tan_mean)
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
        "plot_data": {"curves": curves, "points": points, "range": {"mu_min": mu_min, "mu_max": mu_max}},
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
    tan_return_mult: float = 1.2,
    n_points: int = 200,
) -> dict[str, Any]:
    """
    Scope 3: compare 30 industries vs 30 stocks on a common sample period.

    Repeats sample / BS-mean / BS-mean+cov frontier construction for both
    universes and returns structured tables + plot data.
    """
    if ind_gross.empty:
        raise ValueError("ind_gross is empty")
    if stocks_gross.empty:
        raise ValueError("stocks_gross is empty")
    if rf_gross.empty:
        raise ValueError("rf_gross is empty")

    common_start, common_end = get_common_sample_period(ind_gross, stocks_gross)
    ind_common = ind_gross.loc[common_start:common_end].dropna(how="any")
    stocks_common = stocks_gross.loc[common_start:common_end].dropna(how="any")

    common_idx = ind_common.index.intersection(stocks_common.index).intersection(rf_gross.index).sort_values()
    if len(common_idx) == 0:
        raise ValueError("No common period across industries, stocks, and risk-free series.")

    ind_common = ind_common.loc[common_idx]
    stocks_common = stocks_common.loc[common_idx]
    rf_common = (rf_gross.loc[common_idx] - 1.0).astype(float)
    rf_mean = float(rf_common.mean())

    mu_ind, Sigma_ind = compute_moments_from_gross(ind_common.values)
    mu_stk, Sigma_stk = compute_moments_from_gross(stocks_common.values)

    T_common = len(common_idx)
    bs_ind = bayes_stein_means(mu_ind, Sigma_ind, T=T_common)
    bs_stk = bayes_stein_means(mu_stk, Sigma_stk, T=T_common)
    Sigma_ind_bs = shrink_covariance_identity(Sigma_ind, shrinkage=cov_shrink)
    Sigma_stk_bs = shrink_covariance_identity(Sigma_stk, shrinkage=cov_shrink)

    models = {
        "sample": {
            "industry": {"mu": mu_ind, "Sigma": Sigma_ind},
            "stock": {"mu": mu_stk, "Sigma": Sigma_stk},
        },
        "bs_mean": {
            "industry": {"mu": bs_ind.mu_bs, "Sigma": Sigma_ind},
            "stock": {"mu": bs_stk.mu_bs, "Sigma": Sigma_stk},
        },
        "bs_mean_cov": {
            "industry": {"mu": bs_ind.mu_bs, "Sigma": Sigma_ind_bs},
            "stock": {"mu": bs_stk.mu_bs, "Sigma": Sigma_stk_bs},
        },
    }

    stats = {}
    for est, universes in models.items():
        stats[est] = {}
        for universe, payload in universes.items():
            stats[est][universe] = gmv_tan_stats(payload["mu"], payload["Sigma"], rf=rf_mean)

    summary_rows = []
    for est in ("sample", "bs_mean", "bs_mean_cov"):
        for universe in ("industry", "stock"):
            st = stats[est][universe]["stats"]
            summary_rows.append(
                {
                    "estimator": est,
                    "universe": universe,
                    "portfolio": "GMV",
                    "mean": st["gmv"]["mean"],
                    "vol": st["gmv"]["vol"],
                    "excess_mean": st["gmv"]["excess_mean"],
                    "sharpe": st["gmv"]["excess_mean"] / st["gmv"]["vol"] if st["gmv"]["vol"] > 0 else np.nan,
                }
            )
            summary_rows.append(
                {
                    "estimator": est,
                    "universe": universe,
                    "portfolio": "TAN",
                    "mean": st["tan"]["mean"],
                    "vol": st["tan"]["vol"],
                    "excess_mean": st["tan"]["excess_mean"],
                    "sharpe": st["tan"]["sharpe"],
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    max_tan_mean = max(
        float(stats[est][universe]["stats"]["tan"]["mean"])
        for est in ("sample", "bs_mean", "bs_mean_cov")
        for universe in ("industry", "stock")
    )
    mu_min = 0.0
    mu_max = max(0.001, tan_return_mult * max_tan_mean)

    curves = {}
    points = {}
    for est in ("sample", "bs_mean", "bs_mean_cov"):
        curves[est] = {}
        points[est] = {}
        for universe in ("industry", "stock"):
            payload = models[est][universe]
            curves[est][universe] = build_frontier_curve(
                payload["mu"],
                payload["Sigma"],
                n_points=n_points,
                mu_min=mu_min,
                mu_max=mu_max,
            )
            points[est][universe] = {
                "gmv": stats[est][universe]["stats"]["gmv"],
                "tan": stats[est][universe]["stats"]["tan"],
            }

    return {
        "inputs": {
            "common_start": common_idx.min().strftime("%Y-%m"),
            "common_end": common_idx.max().strftime("%Y-%m"),
            "n_obs": int(T_common),
            "n_assets_industry": int(ind_common.shape[1]),
            "n_assets_stock": int(stocks_common.shape[1]),
            "rf_mean": rf_mean,
            "stock_source": stocks_gross.attrs.get("stock_source"),
            "stock_source_file": stocks_gross.attrs.get("stock_source_file"),
        },
        "models": models,
        "summary_tables": {"gmv_tan": summary_df},
        "plot_data": {"curves": curves, "points": points, "range": {"mu_min": mu_min, "mu_max": mu_max}},
        "diagnostics": {
            "bs_industry_shrinkage_intensity": bs_ind.shrinkage_intensity,
            "bs_stock_shrinkage_intensity": bs_stk.shrinkage_intensity,
            "note": "Scope 3 frontiers are estimated on overlapping dates with complete data.",
            "note_data_source_and_mapping": (
                "Stock data source is loaded dynamically from load_stock_returns_monthly(...) "
                f"using source={stocks_gross.attrs.get('stock_source')!r}, "
                f"file={stocks_gross.attrs.get('stock_source_file')!r}. "
                "If the balanced clean stock file is selected, returns are converted from "
                "monthly net returns to gross returns before estimation. The stock-to-FF30 "
                "mapping is a project crosswalk (not an official Ken French/CRSP one-stock mapping); "
                "some assignments (especially residual-style categories such as 'Other') "
                "remain TODO for validation."
            ),
            "stock_source": stocks_gross.attrs.get("stock_source"),
            "stock_source_file": stocks_gross.attrs.get("stock_source_file"),
            "stock_column_labels": list(stocks_common.columns),
        },
    }


def _infer_coal_stock_column(
    stocks_gross: pd.DataFrame,
    mapping_csv: str | Path | None = None,
) -> str | None:
    """
    Infer which stock column corresponds to FF30 Coal.

    Priority:
    1) direct "Coal" column (legacy industry-labeled stock file),
    2) explicit mapping CSV if provided / available,
    3) fallback to known project crosswalks in stock_data.py.
    """
    cols = set(stocks_gross.columns.astype(str))
    if "Coal" in cols:
        return "Coal"

    candidate_paths: list[Path] = []
    if mapping_csv is not None:
        candidate_paths.append(Path(mapping_csv))
    candidate_paths.append(Path("fin41360_data/processed/scope3_selected_30_stocks_ff30.csv"))

    for p in candidate_paths:
        try:
            if p.exists():
                m = pd.read_csv(p)
                if {"ff30_industry", "ticker"}.issubset(m.columns):
                    row = m[m["ff30_industry"] == "Coal"]
                    if not row.empty:
                        ticker = str(row.iloc[0]["ticker"]).upper().strip()
                        if ticker in cols:
                            return ticker
        except Exception:
            continue

    # Fallback: import known crosswalk constants lazily to avoid hard coupling.
    try:
        from .stock_data import BALANCED_STOCK_FF30_CROSSWALK

        coal_tickers = [t for t, ind in BALANCED_STOCK_FF30_CROSSWALK.items() if ind == "Coal"]
        for t in coal_tickers:
            if t in cols:
                return t
    except Exception:
        pass

    return None


def run_scope3_sensitivity_with_and_without_coal(
    ind_gross: pd.DataFrame,
    stocks_gross: pd.DataFrame,
    rf_gross: pd.Series,
    cov_shrink: float = 0.1,
    tan_return_mult: float = 1.2,
    n_points: int = 200,
    coal_mapping_csv: str | Path | None = None,
) -> dict[str, Any]:
    """
    Scope 3 sensitivity package:
    - Base comparison with Coal included (30 vs 30 when available),
    - Alternative comparison dropping Coal from both universes (29 vs 29),
    - Industry-only stability comparison:
      full-sample vs short common window implied by the with-Coal setup.

    Returns
    -------
    dict with keys:
    - with_coal_30: output of run_scope3_industries_vs_stocks
    - drop_coal_29: output of run_scope3_industries_vs_stocks
    - industry_stability: {full_sample, short_common_with_coal, summary_window_table}
    """
    if "Coal" not in ind_gross.columns:
        raise ValueError("Industry dataset does not contain a 'Coal' column.")

    # 1) Base case (includes Coal)
    with_coal = run_scope3_industries_vs_stocks(
        ind_gross=ind_gross,
        stocks_gross=stocks_gross,
        rf_gross=rf_gross,
        cov_shrink=cov_shrink,
        tan_return_mult=tan_return_mult,
        n_points=n_points,
    )

    short_start = with_coal["inputs"]["common_start"]
    short_end = with_coal["inputs"]["common_end"]

    # 2) Alternative: drop Coal from both universes
    coal_stock_col = _infer_coal_stock_column(stocks_gross, mapping_csv=coal_mapping_csv)
    if coal_stock_col is None:
        raise ValueError(
            "Could not infer Coal stock column in stocks_gross. "
            "Pass coal_mapping_csv or ensure mapping file exists."
        )

    ind_no_coal = ind_gross.drop(columns=["Coal"]).copy()
    stocks_no_coal = stocks_gross.drop(columns=[coal_stock_col]).copy()
    stocks_no_coal.attrs.update(stocks_gross.attrs)
    stocks_no_coal.attrs["coal_column_dropped"] = coal_stock_col

    drop_coal = run_scope3_industries_vs_stocks(
        ind_gross=ind_no_coal,
        stocks_gross=stocks_no_coal,
        rf_gross=rf_gross,
        cov_shrink=cov_shrink,
        tan_return_mult=tan_return_mult,
        n_points=n_points,
    )

    # 3) Industry-only stability diagnostic: full vs short window
    ind_full_scope2 = run_scope2_industries_sample_vs_bs(
        ind_gross=ind_gross,
        rf_gross=rf_gross,
        cov_shrink=cov_shrink,
        tan_return_mult=tan_return_mult,
        n_points=n_points,
    )
    ind_short_scope2 = run_scope2_industries_sample_vs_bs(
        ind_gross=ind_gross.loc[short_start:short_end],
        rf_gross=rf_gross.loc[short_start:short_end],
        cov_shrink=cov_shrink,
        tan_return_mult=tan_return_mult,
        n_points=n_points,
    )

    window_summary = pd.DataFrame(
        [
            {
                "scenario": "with_coal_30",
                "common_start": with_coal["inputs"]["common_start"],
                "common_end": with_coal["inputs"]["common_end"],
                "n_obs": with_coal["inputs"]["n_obs"],
                "n_assets_industry": with_coal["inputs"]["n_assets_industry"],
                "n_assets_stock": with_coal["inputs"]["n_assets_stock"],
                "bs_industry_shrinkage_intensity": with_coal["diagnostics"]["bs_industry_shrinkage_intensity"],
                "bs_stock_shrinkage_intensity": with_coal["diagnostics"]["bs_stock_shrinkage_intensity"],
            },
            {
                "scenario": "drop_coal_29",
                "common_start": drop_coal["inputs"]["common_start"],
                "common_end": drop_coal["inputs"]["common_end"],
                "n_obs": drop_coal["inputs"]["n_obs"],
                "n_assets_industry": drop_coal["inputs"]["n_assets_industry"],
                "n_assets_stock": drop_coal["inputs"]["n_assets_stock"],
                "bs_industry_shrinkage_intensity": drop_coal["diagnostics"]["bs_industry_shrinkage_intensity"],
                "bs_stock_shrinkage_intensity": drop_coal["diagnostics"]["bs_stock_shrinkage_intensity"],
            },
        ]
    )

    return {
        "with_coal_30": with_coal,
        "drop_coal_29": drop_coal,
        "industry_stability": {
            "full_sample": ind_full_scope2,
            "short_common_with_coal": ind_short_scope2,
            "summary_window_table": window_summary,
        },
        "diagnostics": {
            "coal_stock_column_dropped": coal_stock_col,
            "rationale": (
                "with_coal_30 preserves full FF30 breadth but can force a shorter common window; "
                "drop_coal_29 relaxes breadth by one industry to recover a longer common sample."
            ),
        },
    }


def run_scope4_industries_with_rf(
    ind_gross: pd.DataFrame,
    rf_gross: pd.Series,
    tan_return_mult: float = 1.2,
    n_points: int = 200,
) -> dict[str, Any]:
    """
    Scope 4: industries + risk-free asset using sample estimates.

    Returns risky frontier, CML, GMV/TAN stats, and aligned sample metadata.
    """
    if ind_gross.empty:
        raise ValueError("ind_gross is empty")
    if rf_gross.empty:
        raise ValueError("rf_gross is empty")

    common_idx = ind_gross.index.intersection(rf_gross.index).sort_values()
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between industry and risk-free series.")

    ind_aligned = ind_gross.loc[common_idx]
    rf_net = (rf_gross.loc[common_idx] - 1.0).astype(float)
    rf_mean = float(rf_net.mean())

    mu, Sigma = compute_moments_from_gross(ind_aligned.values)
    stats = gmv_tan_stats(mu, Sigma, rf=rf_mean)

    mu_min = 0.0
    mu_max = max(0.001, tan_return_mult * float(stats["stats"]["tan"]["mean"]))
    curve = build_frontier_curve(mu, Sigma, n_points=n_points, mu_min=mu_min, mu_max=mu_max)

    # Use same x-range for CML and risky frontier.
    x_max = float(curve["vols"].max())
    cml_vol, cml_mean = build_cml(
        sharpe_tan=float(stats["stats"]["tan"]["sharpe"]),
        vol_max=x_max,
        rf=rf_mean,
        n_points=150,
    )

    summary = pd.DataFrame(
        [
            {
                "universe": "industries",
                "portfolio": "GMV",
                "mean": stats["stats"]["gmv"]["mean"],
                "vol": stats["stats"]["gmv"]["vol"],
                "excess_mean": stats["stats"]["gmv"]["excess_mean"],
                "sharpe": stats["stats"]["gmv"]["excess_mean"] / stats["stats"]["gmv"]["vol"]
                if stats["stats"]["gmv"]["vol"] > 0
                else np.nan,
            },
            {
                "universe": "industries",
                "portfolio": "TAN",
                "mean": stats["stats"]["tan"]["mean"],
                "vol": stats["stats"]["tan"]["vol"],
                "excess_mean": stats["stats"]["tan"]["excess_mean"],
                "sharpe": stats["stats"]["tan"]["sharpe"],
            },
        ]
    )

    return {
        "inputs": {
            "start": common_idx.min().strftime("%Y-%m"),
            "end": common_idx.max().strftime("%Y-%m"),
            "n_obs": int(len(common_idx)),
            "n_assets": int(ind_aligned.shape[1]),
            "rf_mean": rf_mean,
        },
        "summary_tables": {"gmv_tan": summary},
        "plot_data": {
            "curve": curve,
            "cml": {"vols": cml_vol, "means": cml_mean},
            "range": {"mu_min": mu_min, "mu_max": mu_max, "x_max": x_max},
            "points": {
                "gmv": stats["stats"]["gmv"],
                "tan": stats["stats"]["tan"],
            },
        },
        "diagnostics": {
            "note": "Scope 4 efficient set combines risky frontier and CML through tangency."
        },
    }


def run_scope5_industries_vs_ff(
    ind_gross: pd.DataFrame,
    ff3_excess: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    rf_gross: pd.Series,
    tan_return_mult: float = 1.2,
    n_points: int = 200,
) -> dict[str, Any]:
    """
    Scope 5: compare industries (excess) vs FF3 vs FF5 in excess-return space.
    """
    if ind_gross.empty:
        raise ValueError("ind_gross is empty")
    if ff3_excess.empty:
        raise ValueError("ff3_excess is empty")
    if ff5_excess.empty:
        raise ValueError("ff5_excess is empty")
    if rf_gross.empty:
        raise ValueError("rf_gross is empty")

    common_idx = (
        ind_gross.index.intersection(ff3_excess.index).intersection(ff5_excess.index).intersection(rf_gross.index)
    ).sort_values()
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates across industries, FF3, FF5, and RF.")

    ind_net = (ind_gross.loc[common_idx] - 1.0).astype(float)
    rf_net = (rf_gross.loc[common_idx] - 1.0).astype(float)
    ind_excess = ind_net.sub(rf_net, axis=0)
    ff3 = ff3_excess.loc[common_idx].astype(float)
    ff5 = ff5_excess.loc[common_idx].astype(float)

    mu_ind, Sigma_ind = compute_moments_from_net(ind_excess.values)
    mu_ff3, Sigma_ff3 = compute_moments_from_net(ff3.values)
    mu_ff5, Sigma_ff5 = compute_moments_from_net(ff5.values)

    models = {
        "industries": {"mu": mu_ind, "Sigma": Sigma_ind},
        "ff3": {"mu": mu_ff3, "Sigma": Sigma_ff3},
        "ff5": {"mu": mu_ff5, "Sigma": Sigma_ff5},
    }
    stats = {k: gmv_tan_stats(v["mu"], v["Sigma"], rf=0.0) for k, v in models.items()}

    summary_rows = []
    for key in ("industries", "ff3", "ff5"):
        st = stats[key]["stats"]
        summary_rows.append(
            {
                "series": key,
                "universe": key,
                "portfolio": "GMV",
                "mean": st["gmv"]["mean"],
                "vol": st["gmv"]["vol"],
                "excess_mean": st["gmv"]["excess_mean"],
                "sharpe": st["gmv"]["excess_mean"] / st["gmv"]["vol"] if st["gmv"]["vol"] > 0 else np.nan,
            }
        )
        summary_rows.append(
            {
                "series": key,
                "universe": key,
                "portfolio": "TAN",
                "mean": st["tan"]["mean"],
                "vol": st["tan"]["vol"],
                "excess_mean": st["tan"]["excess_mean"],
                "sharpe": st["tan"]["sharpe"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    max_tan_mean = max(float(points["tan"]["mean"]) for points in [stats["industries"]["stats"], stats["ff3"]["stats"], stats["ff5"]["stats"]])
    mu_min = 0.0
    mu_max = max(0.001, tan_return_mult * max_tan_mean)
    curves = {
        key: build_frontier_curve(models[key]["mu"], models[key]["Sigma"], n_points=n_points, mu_min=mu_min, mu_max=mu_max)
        for key in ("industries", "ff3", "ff5")
    }
    points = {
        key: {
            "gmv": stats[key]["stats"]["gmv"],
            "tan": stats[key]["stats"]["tan"],
        }
        for key in ("industries", "ff3", "ff5")
    }

    # Set x_max from the CML with greatest slope:
    # target return = tan_return_mult * max tangency mean
    # x_max = target_return / max_slope
    max_slope = max(float(points[k]["tan"]["sharpe"]) for k in points.keys())
    x_max = float(mu_max / max(max_slope, 1e-10))
    cml = {}
    for key in ("industries", "ff3", "ff5"):
        cml_vol, cml_mean = build_cml(
            sharpe_tan=float(points[key]["tan"]["sharpe"]),
            vol_max=x_max,
            rf=0.0,
            n_points=150,
        )
        cml[key] = {"vols": cml_vol, "means": cml_mean}

    return {
        "inputs": {
            "start": common_idx.min().strftime("%Y-%m"),
            "end": common_idx.max().strftime("%Y-%m"),
            "n_obs": int(len(common_idx)),
            "space": "excess returns (rf=0)",
        },
        "models": models,
        "summary_tables": {"gmv_tan": summary_df},
        "plot_data": {
            "curves": curves,
            "points": points,
            "cml": cml,
            "range": {"mu_min": mu_min, "mu_max": mu_max, "x_max": x_max},
        },
        "diagnostics": {
            "note": "Industries are converted to excess returns before comparison to FF3/FF5."
        },
    }


def run_scope6_ff_vs_proxies(
    ff3_excess: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    proxy_returns: pd.DataFrame,
    rf_gross: pd.Series,
    tan_return_mult: float = 1.2,
    n_points: int = 200,
) -> dict[str, Any]:
    """
    Scope 6: compare FF3/FF5 frontiers to practical proxy frontiers (excess space).
    """
    if ff3_excess.empty:
        raise ValueError("ff3_excess is empty")
    if ff5_excess.empty:
        raise ValueError("ff5_excess is empty")
    if proxy_returns.empty:
        raise ValueError("proxy_returns is empty")
    if rf_gross.empty:
        raise ValueError("rf_gross is empty")

    required_proxy_cols = {"Mkt", "SMB", "HML", "RMW", "CMA"}
    missing = required_proxy_cols - set(proxy_returns.columns)
    if missing:
        raise ValueError(f"proxy_returns missing required columns: {sorted(missing)}")

    common_idx = (
        proxy_returns.index.intersection(ff3_excess.index).intersection(ff5_excess.index).intersection(rf_gross.index)
    ).sort_values()
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates across proxies, FF3, FF5, and RF.")

    proxy = proxy_returns.loc[common_idx].astype(float)
    ff3 = ff3_excess.loc[common_idx].astype(float)
    ff5 = ff5_excess.loc[common_idx].astype(float)
    rf_net = (rf_gross.loc[common_idx] - 1.0).astype(float)
    proxy_excess = proxy.sub(rf_net, axis=0)

    mu_ff3, Sigma_ff3 = compute_moments_from_net(ff3.values)
    mu_ff5, Sigma_ff5 = compute_moments_from_net(ff5.values)
    mu_proxy3, Sigma_proxy3 = compute_moments_from_net(proxy_excess[["Mkt", "SMB", "HML"]].values)
    mu_proxy5, Sigma_proxy5 = compute_moments_from_net(proxy_excess[["Mkt", "SMB", "HML", "RMW", "CMA"]].values)

    models = {
        "ff3": {"mu": mu_ff3, "Sigma": Sigma_ff3},
        "proxy3": {"mu": mu_proxy3, "Sigma": Sigma_proxy3},
        "ff5": {"mu": mu_ff5, "Sigma": Sigma_ff5},
        "proxy5": {"mu": mu_proxy5, "Sigma": Sigma_proxy5},
    }
    stats = {k: gmv_tan_stats(v["mu"], v["Sigma"], rf=0.0) for k, v in models.items()}

    # Optional constrained tangency diagnostics for factors (especially FF3)
    w_tan_ff3_c = tangency_weights_constrained(mu_ff3, Sigma_ff3, rf=0.0, w_min=-1.0)
    mu_t_ff3_c, vol_t_ff3_c = portfolio_stats(w_tan_ff3_c, mu_ff3, Sigma_ff3)
    sr_t_ff3_c = mu_t_ff3_c / vol_t_ff3_c if vol_t_ff3_c > 0 else np.nan
    stats["ff3_constrained"] = {
        "weights": {"tan": w_tan_ff3_c},
        "stats": {"tan": {"mean": mu_t_ff3_c, "vol": vol_t_ff3_c, "sharpe": sr_t_ff3_c}},
    }

    summary_rows = []
    for key, universe in [
        ("ff3", "ff3"),
        ("proxy3", "proxy3"),
        ("ff5", "ff5"),
        ("proxy5", "proxy5"),
    ]:
        st = stats[key]["stats"]
        summary_rows.append(
            {
                "series": key,
                "universe": universe,
                "portfolio": "GMV",
                "mean": st["gmv"]["mean"],
                "vol": st["gmv"]["vol"],
                "excess_mean": st["gmv"]["excess_mean"],
                "sharpe": st["gmv"]["excess_mean"] / st["gmv"]["vol"] if st["gmv"]["vol"] > 0 else np.nan,
            }
        )
        summary_rows.append(
            {
                "series": key,
                "universe": universe,
                "portfolio": "TAN",
                "mean": st["tan"]["mean"],
                "vol": st["tan"]["vol"],
                "excess_mean": st["tan"]["excess_mean"],
                "sharpe": st["tan"]["sharpe"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    max_tan_mean = max(float(stats[k]["stats"]["tan"]["mean"]) for k in ("ff3", "proxy3", "ff5", "proxy5"))
    mu_min = 0.0
    mu_max = max(0.001, tan_return_mult * max_tan_mean)

    curves = {
        key: build_frontier_curve(models[key]["mu"], models[key]["Sigma"], n_points=n_points, mu_min=mu_min, mu_max=mu_max)
        for key in ("ff3", "proxy3", "ff5", "proxy5")
    }
    points = {
        key: {
            "gmv": stats[key]["stats"]["gmv"],
            "tan": stats[key]["stats"]["tan"],
        }
        for key in ("ff3", "proxy3", "ff5", "proxy5")
    }

    max_slope = max(float(points[k]["tan"]["sharpe"]) for k in points.keys())
    x_max = float(mu_max / max(max_slope, 1e-10))
    cml = {}
    for key in ("ff3", "proxy3", "ff5", "proxy5"):
        cml_vol, cml_mean = build_cml(
            sharpe_tan=float(points[key]["tan"]["sharpe"]),
            vol_max=x_max,
            rf=0.0,
            n_points=150,
        )
        cml[key] = {"vols": cml_vol, "means": cml_mean}

    return {
        "inputs": {
            "start": common_idx.min().strftime("%Y-%m"),
            "end": common_idx.max().strftime("%Y-%m"),
            "n_obs": int(len(common_idx)),
            "space": "excess returns (rf=0)",
        },
        "models": models,
        "summary_tables": {"gmv_tan": summary_df},
        "plot_data": {
            "curves": curves,
            "points": points,
            "cml": cml,
            "range": {"mu_min": mu_min, "mu_max": mu_max, "x_max": x_max},
        },
        "diagnostics": {
            "ff3_tan_unconstrained_mean": float(stats["ff3"]["stats"]["tan"]["mean"]),
            "ff3_tan_unconstrained_vol": float(stats["ff3"]["stats"]["tan"]["vol"]),
            "ff3_tan_unconstrained_sharpe": float(stats["ff3"]["stats"]["tan"]["sharpe"]),
            "ff3_tan_unconstrained_weights": stats["ff3"]["weights"]["tan"],
            "ff3_tan_constrained_wmin": -1.0,
            "ff3_tan_constrained_mean": mu_t_ff3_c,
            "ff3_tan_constrained_vol": vol_t_ff3_c,
            "ff3_tan_constrained_sharpe": sr_t_ff3_c,
            "note": "Proxies are converted to excess returns by subtracting RF.",
        },
    }


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
