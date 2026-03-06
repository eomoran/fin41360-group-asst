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

from .bayes_stein import bayes_stein_means, shrink_covariance_identity, shrink_covariance_ledoit_wolf
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
    efficient_frontier_constrained,
    gmv_weights,
    gmv_weights_constrained,
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
    cov_shrink: float | str = "ledoit_wolf",
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
    if isinstance(cov_shrink, str):
        mode = cov_shrink.strip().lower()
        if mode in {"ledoit_wolf", "lw"}:
            Sigma_bs, cov_shrink_eff = shrink_covariance_ledoit_wolf(ind_aligned.values - 1.0)
            cov_shrink_method = "ledoit_wolf"
        else:
            raise ValueError("cov_shrink string mode must be one of {'ledoit_wolf','lw'}")
    else:
        cov_shrink_eff = float(cov_shrink)
        Sigma_bs = shrink_covariance_identity(Sigma_sample, shrinkage=cov_shrink_eff)
        cov_shrink_method = "identity_fixed_lambda"

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
            "cov_shrink_method": cov_shrink_method,
            "cov_shrink_effective_lambda": float(cov_shrink_eff),
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
    cov_shrink: float | str = "ledoit_wolf",
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
    if isinstance(cov_shrink, str):
        mode = cov_shrink.strip().lower()
        if mode in {"ledoit_wolf", "lw"}:
            Sigma_ind_bs, cov_shrink_eff_ind = shrink_covariance_ledoit_wolf(ind_common.values - 1.0)
            Sigma_stk_bs, cov_shrink_eff_stk = shrink_covariance_ledoit_wolf(stocks_common.values - 1.0)
            cov_shrink_method = "ledoit_wolf"
        else:
            raise ValueError("cov_shrink string mode must be one of {'ledoit_wolf','lw'}")
    else:
        cov_shrink_eff_ind = float(cov_shrink)
        cov_shrink_eff_stk = float(cov_shrink)
        Sigma_ind_bs = shrink_covariance_identity(Sigma_ind, shrinkage=cov_shrink_eff_ind)
        Sigma_stk_bs = shrink_covariance_identity(Sigma_stk, shrinkage=cov_shrink_eff_stk)
        cov_shrink_method = "identity_fixed_lambda"

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
            "cov_shrink_method": cov_shrink_method,
            "cov_shrink_effective_lambda_industry": float(cov_shrink_eff_ind),
            "cov_shrink_effective_lambda_stock": float(cov_shrink_eff_stk),
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
    cov_shrink: float | str = "ledoit_wolf",
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
                "mean_shrinkage_intensity_industry": with_coal["diagnostics"]["bs_industry_shrinkage_intensity"],
                "mean_shrinkage_intensity_stock": with_coal["diagnostics"]["bs_stock_shrinkage_intensity"],
                "cov_shrink_method": with_coal["diagnostics"]["cov_shrink_method"],
                "cov_shrinkage_intensity_industry": with_coal["diagnostics"]["cov_shrink_effective_lambda_industry"],
                "cov_shrinkage_intensity_stock": with_coal["diagnostics"]["cov_shrink_effective_lambda_stock"],
            },
            {
                "scenario": "drop_coal_29",
                "common_start": drop_coal["inputs"]["common_start"],
                "common_end": drop_coal["inputs"]["common_end"],
                "n_obs": drop_coal["inputs"]["n_obs"],
                "n_assets_industry": drop_coal["inputs"]["n_assets_industry"],
                "n_assets_stock": drop_coal["inputs"]["n_assets_stock"],
                "mean_shrinkage_intensity_industry": drop_coal["diagnostics"]["bs_industry_shrinkage_intensity"],
                "mean_shrinkage_intensity_stock": drop_coal["diagnostics"]["bs_stock_shrinkage_intensity"],
                "cov_shrink_method": drop_coal["diagnostics"]["cov_shrink_method"],
                "cov_shrinkage_intensity_industry": drop_coal["diagnostics"]["cov_shrink_effective_lambda_industry"],
                "cov_shrinkage_intensity_stock": drop_coal["diagnostics"]["cov_shrink_effective_lambda_stock"],
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


def run_scope8_constraints(
    ind_gross: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    rf_gross: pd.Series,
    constraint_levels: tuple[float, ...] = (0.0, -0.25),
    end_is: str = "2002-12",
    start_oos: str = "2003-01",
    end_oos: str = "2025-12",
) -> dict[str, Any]:
    """
    Scope 8: constrained tangency extension with IS/OOS persistence diagnostics.

    For each asset set (30 industries with RF, FF5 excess) this workflow compares:
    - unconstrained tangency
    - constrained tangency portfolios with lower bounds w_i >= w_min

    It returns IS/OOS Sharpe summaries, JK/LW tests, OOS frontier replication
    diagnostics, and weight diagnostics to support a constraints-first Scope 8.
    """
    if ind_gross.empty:
        raise ValueError("ind_gross is empty")
    if ff5_excess.empty:
        raise ValueError("ff5_excess is empty")
    if rf_gross.empty:
        raise ValueError("rf_gross is empty")

    w_min_levels = tuple(dict.fromkeys(float(x) for x in constraint_levels))
    if len(w_min_levels) == 0:
        raise ValueError("constraint_levels cannot be empty.")
    if any(x > 0.999999 for x in w_min_levels):
        raise ValueError("Each w_min must be < 1.0.")

    # ---------------------------
    # Split industries and RF into IS/OOS
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
    split_policy = "requested_dates"
    split_warning = None
    if len(ind_idx_is) == 0 or len(ind_idx_oos) == 0:
        n = len(ind_idx)
        if n < 24:
            raise ValueError(
                "Industry IS/OOS split produced an empty period and overlap is too short "
                f"for fallback split (n={n})."
            )
        mid = n // 2
        ind_idx_is = ind_idx[:mid]
        ind_idx_oos = ind_idx[mid:]
        ind_is_mask = ind_idx.isin(ind_idx_is)
        ind_oos_mask = ind_idx.isin(ind_idx_oos)
        split_policy = "fallback_half_split"
        split_warning = (
            f"Requested split ({end_is} / {start_oos}) yielded empty IS or OOS for industries; "
            "used contiguous half split instead."
        )

    ind_is_arr = ind_net[ind_is_mask]
    ind_oos_arr = ind_net[ind_oos_mask]
    rf_is_arr = rf_net[ind_is_mask]
    rf_oos_arr = rf_net[ind_oos_mask]

    mu_ind_is, Sigma_ind_is = compute_moments_from_net(ind_is_arr)
    mu_ind_oos, Sigma_ind_oos = compute_moments_from_net(ind_oos_arr)
    rf_ind_is = float(np.mean(rf_is_arr))
    rf_ind_oos = float(np.mean(rf_oos_arr))

    # ---------------------------
    # Split FF5 excess into IS/OOS
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
        n = len(ff5_idx)
        if n < 24:
            raise ValueError(
                "FF5 IS/OOS split produced an empty period and overlap is too short "
                f"for fallback split (n={n})."
            )
        mid = n // 2
        ff5_idx_is = ff5_idx[:mid]
        ff5_idx_oos = ff5_idx[mid:]
        ff5_is_mask = ff5_idx.isin(ff5_idx_is)
        ff5_oos_mask = ff5_idx.isin(ff5_idx_oos)
        split_policy = "fallback_half_split"
        split_warning = (
            f"Requested split ({end_is} / {start_oos}) yielded empty IS or OOS for FF5; "
            "used contiguous half split instead."
        )

    ff5_is_arr = ff5_arr[ff5_is_mask]
    ff5_oos_arr = ff5_arr[ff5_oos_mask]

    mu_ff5_is, Sigma_ff5_is = compute_moments_from_net(ff5_is_arr)
    mu_ff5_oos, Sigma_ff5_oos = compute_moments_from_net(ff5_oos_arr)

    # OOS frontier anchors (unconstrained TAN + GMV)
    w_tan_ind_oos = tangency_weights(mu_ind_oos, Sigma_ind_oos, rf_ind_oos)
    w_gmv_ind_oos = gmv_weights(Sigma_ind_oos)
    w_tan_ff5_oos = tangency_weights(mu_ff5_oos, Sigma_ff5_oos, 0.0)
    w_gmv_ff5_oos = gmv_weights(Sigma_ff5_oos)

    sharpe_rows = []
    jk_rows = []
    lw_rows = []
    rep_rows = []
    weight_rows = []

    asset_specs = [
        {
            "asset_set": "industries",
            "is_arr": ind_is_arr,
            "oos_arr": ind_oos_arr,
            "mu_is": mu_ind_is,
            "Sigma_is": Sigma_ind_is,
            "rf_is_scalar": rf_ind_is,
            "rf_oos_scalar": rf_ind_oos,
            "rf_is_series": rf_is_arr,
            "rf_oos_series": rf_oos_arr,
            "w_tan_oos": w_tan_ind_oos,
            "w_gmv_oos": w_gmv_ind_oos,
        },
        {
            "asset_set": "ff5_excess",
            "is_arr": ff5_is_arr,
            "oos_arr": ff5_oos_arr,
            "mu_is": mu_ff5_is,
            "Sigma_is": Sigma_ff5_is,
            "rf_is_scalar": 0.0,
            "rf_oos_scalar": 0.0,
            "rf_is_series": 0.0,
            "rf_oos_series": 0.0,
            "w_tan_oos": w_tan_ff5_oos,
            "w_gmv_oos": w_gmv_ff5_oos,
        },
    ]

    for spec in asset_specs:
        asset_set = str(spec["asset_set"])
        is_arr = np.asarray(spec["is_arr"])
        oos_arr = np.asarray(spec["oos_arr"])
        mu_is = np.asarray(spec["mu_is"])
        Sigma_is = np.asarray(spec["Sigma_is"])
        rf_is_scalar = float(spec["rf_is_scalar"])
        rf_oos_scalar = float(spec["rf_oos_scalar"])
        rf_is_series = spec["rf_is_series"]
        rf_oos_series = spec["rf_oos_series"]
        w_tan_oos = np.asarray(spec["w_tan_oos"])
        w_gmv_oos = np.asarray(spec["w_gmv_oos"])

        model_defs: list[tuple[str, float | None, np.ndarray]] = [
            ("unconstrained", None, tangency_weights(mu_is, Sigma_is, rf_is_scalar))
        ]
        for w_min in w_min_levels:
            model_defs.append(
                (
                    "constrained",
                    float(w_min),
                    tangency_weights_constrained(mu_is, Sigma_is, rf=rf_is_scalar, w_min=float(w_min)),
                )
            )

        for model_name, w_min_val, w in model_defs:
            r_is = is_arr @ w
            r_oos = oos_arr @ w
            sr = summarize_sharpe_is_oos(r_is, r_oos, rf_is=rf_is_series, rf_oos=rf_oos_series)
            jk = jobson_korkie_test(r_is, r_oos, rf1=rf_is_scalar, rf2=rf_oos_scalar)
            lw = ledoit_wolf_test(r_is, r_oos, rf1=rf_is_scalar, rf2=rf_oos_scalar, n_boot=2000)
            rep = frontier_replication_alpha(w, w_tan_oos, w_gmv_oos)

            label = model_name if w_min_val is None else f"w_i>={w_min_val:.2f}"

            sharpe_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    **sr,
                }
            )
            jk_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "sr_is": jk.sharpe1,
                    "sr_oos": jk.sharpe2,
                    "z_stat": jk.statistic,
                    "pvalue": jk.pvalue_two_sided,
                }
            )
            lw_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "sr_is": lw.sharpe1,
                    "sr_oos": lw.sharpe2,
                    "diff": lw.difference,
                    "ci_low": lw.ci_low,
                    "ci_high": lw.ci_high,
                    "pvalue": lw.pvalue_two_sided,
                }
            )
            rep_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "alpha": rep.alpha,
                    "r_squared": rep.r_squared,
                    "residual_norm": rep.residual_norm,
                }
            )
            weight_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "min_w": float(np.min(w)),
                    "max_w": float(np.max(w)),
                    "gross_leverage": float(np.sum(np.abs(w))),
                    "n_assets": int(len(w)),
                    "effective_n": float((np.sum(w**2) ** -1) if np.sum(w**2) > 0 else np.nan),
                }
            )

    sharpe_df = pd.DataFrame(sharpe_rows)
    jk_df = pd.DataFrame(jk_rows)
    lw_df = pd.DataFrame(lw_rows)
    rep_df = pd.DataFrame(rep_rows)
    weights_df = pd.DataFrame(weight_rows)

    # Keep both model-level summaries and full weight vectors for diagnostics.
    weights_payload: dict[str, dict[str, np.ndarray]] = {"industries": {}, "ff5_excess": {}}
    for spec in asset_specs:
        asset_set = str(spec["asset_set"])
        mu_is = np.asarray(spec["mu_is"])
        Sigma_is = np.asarray(spec["Sigma_is"])
        rf_is_scalar = float(spec["rf_is_scalar"])
        weights_payload[asset_set]["unconstrained"] = tangency_weights(mu_is, Sigma_is, rf_is_scalar)
        for w_min in w_min_levels:
            weights_payload[asset_set][f"w_i>={w_min:.2f}"] = tangency_weights_constrained(
                mu_is, Sigma_is, rf=rf_is_scalar, w_min=float(w_min)
            )

    return {
        "inputs": {
            "end_is": end_is,
            "start_oos": start_oos,
            "end_oos": end_oos,
            "split_policy": split_policy,
            "constraint_levels": list(w_min_levels),
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
            "sharpe_is_oos": sharpe_df,
            "jk_tests": jk_df,
            "lw_tests": lw_df,
            "frontier_replication": rep_df,
            "weight_diagnostics": weights_df,
        },
        "plot_data": {
            "weights": weights_payload,
            "oos_frontier_anchors": {
                "w_tan_ind_oos": w_tan_ind_oos,
                "w_gmv_ind_oos": w_gmv_ind_oos,
                "w_tan_ff5_oos": w_tan_ff5_oos,
                "w_gmv_ff5_oos": w_gmv_ff5_oos,
            },
        },
        "diagnostics": {
            "split_warning": split_warning,
            "note": (
                "Scope 8 constraints workflow compares unconstrained and constrained tangency "
                "portfolios selected in-sample, then evaluates OOS persistence."
            )
        },
    }


def run_scope8_constraints_with_proxies(
    ff3_excess: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    proxy_returns: pd.DataFrame,
    rf_gross: pd.Series,
    constraint_levels: tuple[float, ...] = (0.0, -0.25),
    end_is: str = "2002-12",
    start_oos: str = "2003-01",
    end_oos: str = "2025-12",
    n_points: int = 1200,
) -> dict[str, Any]:
    """
    Scope 8 extension: constrained tangency persistence for FF/proxy asset sets.

    Asset sets evaluated (excess-return space, rf=0):
    - ff3
    - proxy3 (Mkt, SMB, HML) from practical proxies net of RF
    - ff5
    - proxy5 (Mkt, SMB, HML, RMW, CMA) from practical proxies net of RF
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

    w_min_levels = tuple(dict.fromkeys(float(x) for x in constraint_levels))
    if len(w_min_levels) == 0:
        raise ValueError("constraint_levels cannot be empty.")
    if any(x > 0.999999 for x in w_min_levels):
        raise ValueError("Each w_min must be < 1.0.")

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

    split_is = common_idx <= pd.Timestamp(end_is).to_period("M").to_timestamp("M")
    split_oos = (common_idx >= pd.Timestamp(start_oos).to_period("M").to_timestamp("M")) & (
        common_idx <= pd.Timestamp(end_oos).to_period("M").to_timestamp("M")
    )
    idx_is = common_idx[split_is]
    idx_oos = common_idx[split_oos]

    split_policy = "requested_dates"
    split_warning = None
    if len(idx_is) == 0 or len(idx_oos) == 0:
        # Proxy overlap can be much shorter (e.g., ETF-era starts after 2003),
        # so the assignment split can leave one side empty. Fall back to a
        # valid contiguous half split on the available common window.
        n = len(common_idx)
        if n < 24:
            raise ValueError(
                "IS/OOS split produced an empty period and common overlap is too short "
                f"for fallback split (n={n})."
            )
        mid = n // 2
        idx_is = common_idx[:mid]
        idx_oos = common_idx[mid:]
        split_policy = "fallback_half_split"
        split_warning = (
            f"Requested split ({end_is} / {start_oos}) yielded empty IS or OOS for "
            "the proxy common sample; used contiguous half split instead."
        )

    ff3_is_arr = ff3.loc[idx_is].values
    ff3_oos_arr = ff3.loc[idx_oos].values
    ff5_is_arr = ff5.loc[idx_is].values
    ff5_oos_arr = ff5.loc[idx_oos].values
    proxy3_is_arr = proxy_excess.loc[idx_is, ["Mkt", "SMB", "HML"]].values
    proxy3_oos_arr = proxy_excess.loc[idx_oos, ["Mkt", "SMB", "HML"]].values
    proxy5_is_arr = proxy_excess.loc[idx_is, ["Mkt", "SMB", "HML", "RMW", "CMA"]].values
    proxy5_oos_arr = proxy_excess.loc[idx_oos, ["Mkt", "SMB", "HML", "RMW", "CMA"]].values

    ff3_mu_is, ff3_sigma_is = compute_moments_from_net(ff3_is_arr)
    ff3_mu_oos, ff3_sigma_oos = compute_moments_from_net(ff3_oos_arr)
    ff5_mu_is, ff5_sigma_is = compute_moments_from_net(ff5_is_arr)
    ff5_mu_oos, ff5_sigma_oos = compute_moments_from_net(ff5_oos_arr)
    proxy3_mu_is, proxy3_sigma_is = compute_moments_from_net(proxy3_is_arr)
    proxy3_mu_oos, proxy3_sigma_oos = compute_moments_from_net(proxy3_oos_arr)
    proxy5_mu_is, proxy5_sigma_is = compute_moments_from_net(proxy5_is_arr)
    proxy5_mu_oos, proxy5_sigma_oos = compute_moments_from_net(proxy5_oos_arr)

    specs = [
        {
            "asset_set": "ff3",
            "is_arr": ff3_is_arr,
            "oos_arr": ff3_oos_arr,
            "mu_is": ff3_mu_is,
            "sigma_is": ff3_sigma_is,
            "w_tan_oos": tangency_weights(ff3_mu_oos, ff3_sigma_oos, 0.0),
            "w_gmv_oos": gmv_weights(ff3_sigma_oos),
        },
        {
            "asset_set": "proxy3",
            "is_arr": proxy3_is_arr,
            "oos_arr": proxy3_oos_arr,
            "mu_is": proxy3_mu_is,
            "sigma_is": proxy3_sigma_is,
            "w_tan_oos": tangency_weights(proxy3_mu_oos, proxy3_sigma_oos, 0.0),
            "w_gmv_oos": gmv_weights(proxy3_sigma_oos),
        },
        {
            "asset_set": "ff5",
            "is_arr": ff5_is_arr,
            "oos_arr": ff5_oos_arr,
            "mu_is": ff5_mu_is,
            "sigma_is": ff5_sigma_is,
            "w_tan_oos": tangency_weights(ff5_mu_oos, ff5_sigma_oos, 0.0),
            "w_gmv_oos": gmv_weights(ff5_sigma_oos),
        },
        {
            "asset_set": "proxy5",
            "is_arr": proxy5_is_arr,
            "oos_arr": proxy5_oos_arr,
            "mu_is": proxy5_mu_is,
            "sigma_is": proxy5_sigma_is,
            "w_tan_oos": tangency_weights(proxy5_mu_oos, proxy5_sigma_oos, 0.0),
            "w_gmv_oos": gmv_weights(proxy5_sigma_oos),
        },
    ]

    sharpe_rows = []
    jk_rows = []
    lw_rows = []
    rep_rows = []
    weight_rows = []

    plot_curves: dict[str, dict[str, np.ndarray]] = {}
    plot_points: dict[str, dict[str, dict[str, float]]] = {}
    constrained_points: dict[str, dict[str, dict[str, float]]] = {}
    constrained_curves: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    max_tan_mean = 0.0
    for s in specs:
        w_tan_u = tangency_weights(s["mu_is"], s["sigma_is"], 0.0)
        mu_tan_u, _ = portfolio_stats(w_tan_u, s["mu_is"], s["sigma_is"])
        max_tan_mean = max(max_tan_mean, float(mu_tan_u))

    mu_min = 0.0
    mu_max = max(0.001, 1.2 * max_tan_mean)

    for s in specs:
        asset_set = str(s["asset_set"])
        is_arr = np.asarray(s["is_arr"])
        oos_arr = np.asarray(s["oos_arr"])
        mu_is = np.asarray(s["mu_is"])
        sigma_is = np.asarray(s["sigma_is"])
        w_tan_oos = np.asarray(s["w_tan_oos"])
        w_gmv_oos = np.asarray(s["w_gmv_oos"])

        w_tan_u = tangency_weights(mu_is, sigma_is, 0.0)
        w_gmv_u = gmv_weights(sigma_is)
        mu_tan_u, vol_tan_u = portfolio_stats(w_tan_u, mu_is, sigma_is)
        mu_gmv_u, vol_gmv_u = portfolio_stats(w_gmv_u, mu_is, sigma_is)

        plot_curves[asset_set] = build_frontier_curve(mu_is, sigma_is, n_points=n_points, mu_min=mu_min, mu_max=mu_max)
        plot_points[asset_set] = {
            "gmv": {"mean": float(mu_gmv_u), "vol": float(vol_gmv_u)},
            "tan": {"mean": float(mu_tan_u), "vol": float(vol_tan_u)},
        }
        constrained_points[asset_set] = {}
        constrained_curves[asset_set] = {}

        model_defs: list[tuple[str, float | None, np.ndarray]] = [("unconstrained", None, w_tan_u)]
        for w_min in w_min_levels:
            model_defs.append(
                (
                    "constrained",
                    float(w_min),
                    tangency_weights_constrained(mu_is, sigma_is, rf=0.0, w_min=float(w_min)),
                )
            )

        for model_name, w_min_val, w in model_defs:
            r_is = is_arr @ w
            r_oos = oos_arr @ w
            sr = summarize_sharpe_is_oos(r_is, r_oos, rf_is=0.0, rf_oos=0.0)
            jk = jobson_korkie_test(r_is, r_oos, rf1=0.0, rf2=0.0)
            lw = ledoit_wolf_test(r_is, r_oos, rf1=0.0, rf2=0.0, n_boot=2000)
            rep = frontier_replication_alpha(w, w_tan_oos, w_gmv_oos)
            label = model_name if w_min_val is None else f"w_i>={w_min_val:.2f}"

            mu_w, vol_w = portfolio_stats(w, mu_is, sigma_is)
            if model_name == "constrained" and w_min_val is not None:
                constrained_points[asset_set][label] = {"mean": float(mu_w), "vol": float(vol_w)}
                c_means, c_vols, _ = efficient_frontier_constrained(
                    mu_is,
                    sigma_is,
                    n_points=n_points,
                    mu_min=None,
                    mu_max=None,
                    w_min=float(w_min_val),
                )
                constrained_curves[asset_set][label] = {"means": c_means, "vols": c_vols}

            sharpe_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    **sr,
                }
            )
            jk_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "sr_is": jk.sharpe1,
                    "sr_oos": jk.sharpe2,
                    "z_stat": jk.statistic,
                    "pvalue": jk.pvalue_two_sided,
                }
            )
            lw_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "sr_is": lw.sharpe1,
                    "sr_oos": lw.sharpe2,
                    "diff": lw.difference,
                    "ci_low": lw.ci_low,
                    "ci_high": lw.ci_high,
                    "pvalue": lw.pvalue_two_sided,
                }
            )
            rep_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "alpha": rep.alpha,
                    "r_squared": rep.r_squared,
                    "residual_norm": rep.residual_norm,
                }
            )
            weight_rows.append(
                {
                    "asset_set": asset_set,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "min_w": float(np.min(w)),
                    "max_w": float(np.max(w)),
                    "gross_leverage": float(np.sum(np.abs(w))),
                    "n_assets": int(len(w)),
                    "effective_n": float((np.sum(w**2) ** -1) if np.sum(w**2) > 0 else np.nan),
                }
            )

    return {
        "inputs": {
            "end_is": end_is,
            "start_oos": start_oos,
            "end_oos": end_oos,
            "split_policy": split_policy,
            "constraint_levels": list(w_min_levels),
            "is_start": idx_is.min().strftime("%Y-%m"),
            "is_end": idx_is.max().strftime("%Y-%m"),
            "oos_start": idx_oos.min().strftime("%Y-%m"),
            "oos_end": idx_oos.max().strftime("%Y-%m"),
            "n_is": int(len(idx_is)),
            "n_oos": int(len(idx_oos)),
            "space": "excess returns (rf=0)",
        },
        "summary_tables": {
            "sharpe_is_oos": pd.DataFrame(sharpe_rows),
            "jk_tests": pd.DataFrame(jk_rows),
            "lw_tests": pd.DataFrame(lw_rows),
            "frontier_replication": pd.DataFrame(rep_rows),
            "weight_diagnostics": pd.DataFrame(weight_rows),
        },
        "plot_data": {
            "curves": plot_curves,
            "points": plot_points,
            "constrained_curves": constrained_curves,
            "constrained_tangency_points": constrained_points,
            "range": {"mu_min": mu_min, "mu_max": mu_max},
        },
        "diagnostics": {
            "split_warning": split_warning,
            "note": (
                "Scope 8 proxy extension compares unconstrained and constrained tangency "
                "portfolios for FF3/Proxy3/FF5/Proxy5, selected in-sample and evaluated OOS."
            )
        },
    }


def _split_index_with_fallback(
    idx: pd.DatetimeIndex,
    end_is: str,
    start_oos: str,
    end_oos: str,
    min_obs: int = 24,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, str, str | None]:
    """Split index into IS/OOS with fallback half-split when requested window is empty."""
    is_mask = idx <= pd.Timestamp(end_is).to_period("M").to_timestamp("M")
    oos_mask = (idx >= pd.Timestamp(start_oos).to_period("M").to_timestamp("M")) & (
        idx <= pd.Timestamp(end_oos).to_period("M").to_timestamp("M")
    )
    idx_is = idx[is_mask]
    idx_oos = idx[oos_mask]
    if len(idx_is) > 0 and len(idx_oos) > 0:
        return idx_is, idx_oos, "requested_dates", None

    if len(idx) < min_obs:
        raise ValueError(
            "IS/OOS split produced an empty period and overlap is too short "
            f"for fallback split (n={len(idx)})."
        )
    mid = len(idx) // 2
    idx_is = idx[:mid]
    idx_oos = idx[mid:]
    warn = (
        f"Requested split ({end_is} / {start_oos}) yielded empty IS or OOS; "
        "used contiguous half split instead."
    )
    return idx_is, idx_oos, "fallback_half_split", warn


def _asset_window_payload(
    asset_set: str,
    returns: pd.DataFrame,
    idx_is: pd.DatetimeIndex,
    idx_oos: pd.DatetimeIndex,
    rf_is: float | np.ndarray,
    rf_oos: float | np.ndarray,
) -> dict[str, Any]:
    """Build common payload for IS/OOS analysis in either net-return or excess-return space."""
    is_arr = returns.loc[idx_is].values
    oos_arr = returns.loc[idx_oos].values
    mu_is, sigma_is = compute_moments_from_net(is_arr)
    mu_oos, sigma_oos = compute_moments_from_net(oos_arr)

    rf_is_scalar = float(np.mean(rf_is)) if not np.isscalar(rf_is) else float(rf_is)
    rf_oos_scalar = float(np.mean(rf_oos)) if not np.isscalar(rf_oos) else float(rf_oos)
    return {
        "asset_set": asset_set,
        "idx_is": idx_is,
        "idx_oos": idx_oos,
        "is_arr": is_arr,
        "oos_arr": oos_arr,
        "mu_is": mu_is,
        "sigma_is": sigma_is,
        "mu_oos": mu_oos,
        "sigma_oos": sigma_oos,
        "rf_is_series": rf_is,
        "rf_oos_series": rf_oos,
        "rf_is_scalar": rf_is_scalar,
        "rf_oos_scalar": rf_oos_scalar,
    }


def run_scope8_1_constraints_full_sample(
    ff3_excess: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    proxy_returns: pd.DataFrame,
    rf_gross: pd.Series,
    constraint_levels: tuple[float, ...] = (0.0, -0.25),
    n_points: int = 1200,
) -> dict[str, Any]:
    """
    Scope 8.1: full common-sample constrained frontier comparison (Scope 6 style).
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

    w_min_levels = tuple(dict.fromkeys(float(x) for x in constraint_levels))
    if len(w_min_levels) == 0:
        raise ValueError("constraint_levels cannot be empty.")
    if any(x > 0.999999 for x in w_min_levels):
        raise ValueError("Each w_min must be < 1.0.")

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

    specs = {
        "ff3": ff3.values,
        "proxy3": proxy_excess[["Mkt", "SMB", "HML"]].values,
        "ff5": ff5.values,
        "proxy5": proxy_excess[["Mkt", "SMB", "HML", "RMW", "CMA"]].values,
    }

    curves: dict[str, dict[str, np.ndarray]] = {}
    points: dict[str, dict[str, dict[str, float]]] = {}
    constrained_curves: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    constrained_tan_points: dict[str, dict[str, dict[str, float]]] = {}
    rows = []

    max_tan_mean = 0.0
    parsed = {}
    for name, arr in specs.items():
        mu, sigma = compute_moments_from_net(arr)
        parsed[name] = (mu, sigma)
        w_tan = tangency_weights(mu, sigma, 0.0)
        mu_t, _ = portfolio_stats(w_tan, mu, sigma)
        max_tan_mean = max(max_tan_mean, float(mu_t))

    mu_min = 0.0
    mu_max = max(0.001, 1.2 * max_tan_mean)

    for name, (mu, sigma) in parsed.items():
        w_tan = tangency_weights(mu, sigma, 0.0)
        w_gmv = gmv_weights(sigma)
        mu_t, vol_t = portfolio_stats(w_tan, mu, sigma)
        mu_g, vol_g = portfolio_stats(w_gmv, mu, sigma)
        curves[name] = build_frontier_curve(mu, sigma, n_points=n_points, mu_min=mu_min, mu_max=mu_max)
        points[name] = {"gmv": {"mean": float(mu_g), "vol": float(vol_g)}, "tan": {"mean": float(mu_t), "vol": float(vol_t)}}
        constrained_curves[name] = {}
        constrained_tan_points[name] = {}

        rows.append(
            {
                "asset_set": name,
                "model": "unconstrained",
                "constraint_label": "unconstrained",
                "w_min": np.nan,
                "mean": float(mu_t),
                "vol": float(vol_t),
                "sharpe": float(mu_t / vol_t) if vol_t > 0 else np.nan,
                "min_w": float(np.min(w_tan)),
                "max_w": float(np.max(w_tan)),
                "gross_leverage": float(np.sum(np.abs(w_tan))),
                "effective_n": float((np.sum(w_tan**2) ** -1) if np.sum(w_tan**2) > 0 else np.nan),
            }
        )

        for w_min in w_min_levels:
            label = f"w_i>={w_min:.2f}"
            w_c = tangency_weights_constrained(mu, sigma, rf=0.0, w_min=float(w_min))
            mu_c, vol_c = portfolio_stats(w_c, mu, sigma)
            c_means, c_vols, _ = efficient_frontier_constrained(mu, sigma, n_points=n_points, mu_min=None, mu_max=None, w_min=float(w_min))
            constrained_curves[name][label] = {"means": c_means, "vols": c_vols}
            constrained_tan_points[name][label] = {"mean": float(mu_c), "vol": float(vol_c)}

            rows.append(
                {
                    "asset_set": name,
                    "model": "constrained",
                    "constraint_label": label,
                    "w_min": float(w_min),
                    "mean": float(mu_c),
                    "vol": float(vol_c),
                    "sharpe": float(mu_c / vol_c) if vol_c > 0 else np.nan,
                    "min_w": float(np.min(w_c)),
                    "max_w": float(np.max(w_c)),
                    "gross_leverage": float(np.sum(np.abs(w_c))),
                    "effective_n": float((np.sum(w_c**2) ** -1) if np.sum(w_c**2) > 0 else np.nan),
                }
            )

    return {
        "inputs": {
            "window_policy": "full_common_sample",
            "start": common_idx.min().strftime("%Y-%m"),
            "end": common_idx.max().strftime("%Y-%m"),
            "n_obs": int(len(common_idx)),
            "constraint_levels": list(w_min_levels),
            "space": "excess returns (rf=0)",
        },
        "summary_tables": {
            "tangency_summary": pd.DataFrame(rows),
        },
        "plot_data": {
            "curves": curves,
            "points": points,
            "constrained_curves": constrained_curves,
            "constrained_tangency_points": constrained_tan_points,
            "range": {"mu_min": mu_min, "mu_max": mu_max},
        },
        "diagnostics": {
            "note": "Scope 8.1 compares constrained vs unconstrained tangency/frontiers on full common sample."
        },
    }


def run_scope8_2_constraints_is_oos(
    ind_gross: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    proxy_returns: pd.DataFrame,
    rf_gross: pd.Series,
    constraint_levels: tuple[float, ...] = (0.0, -0.25),
    end_is: str = "2002-12",
    start_oos: str = "2003-01",
    end_oos: str = "2025-12",
    n_points: int = 600,
) -> dict[str, Any]:
    """
    Scope 8.2: constraints + IS/OOS persistence with full IS/OOS frontiers and points.
    """
    if ind_gross.empty:
        raise ValueError("ind_gross is empty")
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

    w_min_levels = tuple(dict.fromkeys(float(x) for x in constraint_levels))
    if len(w_min_levels) == 0:
        raise ValueError("constraint_levels cannot be empty.")

    warnings = []
    split_policy = "requested_dates"

    # Industries (net space + rf)
    ind_idx = ind_gross.index.intersection(rf_gross.index).sort_values()
    if len(ind_idx) == 0:
        raise ValueError("No overlap between industries and RF.")
    ind_idx_is, ind_idx_oos, pol_ind, warn_ind = _split_index_with_fallback(ind_idx, end_is, start_oos, end_oos)
    split_policy = "fallback_half_split" if pol_ind == "fallback_half_split" else split_policy
    if warn_ind:
        warnings.append(f"industries: {warn_ind}")
    ind_net = (ind_gross.loc[ind_idx] - 1.0).astype(float)
    rf_net = (rf_gross.loc[ind_idx] - 1.0).astype(float)
    ind_payload = _asset_window_payload(
        "industries",
        ind_net,
        ind_idx_is,
        ind_idx_oos,
        rf_net.loc[ind_idx_is].values,
        rf_net.loc[ind_idx_oos].values,
    )

    # FF5 / Proxy5 (excess space)
    fp_idx = ff5_excess.index.intersection(proxy_returns.index).intersection(rf_gross.index).sort_values()
    if len(fp_idx) == 0:
        raise ValueError("No overlap between FF5, proxy returns, and RF.")
    fp_idx_is, fp_idx_oos, pol_fp, warn_fp = _split_index_with_fallback(fp_idx, end_is, start_oos, end_oos)
    split_policy = "fallback_half_split" if pol_fp == "fallback_half_split" else split_policy
    if warn_fp:
        warnings.append(f"ff5/proxy5: {warn_fp}")
    ff5 = ff5_excess.loc[fp_idx].astype(float)
    proxy = proxy_returns.loc[fp_idx].astype(float)
    rf_fp = (rf_gross.loc[fp_idx] - 1.0).astype(float)
    proxy5_excess = proxy[["Mkt", "SMB", "HML", "RMW", "CMA"]].sub(rf_fp, axis=0)

    ff5_payload = _asset_window_payload("ff5", ff5, fp_idx_is, fp_idx_oos, 0.0, 0.0)
    proxy5_payload = _asset_window_payload("proxy5", proxy5_excess, fp_idx_is, fp_idx_oos, 0.0, 0.0)

    payloads = [ind_payload, ff5_payload, proxy5_payload]
    sharpe_rows = []
    jk_rows = []
    lw_rows = []
    rep_rows = []
    weight_rows = []
    recomb_rows = []
    plot_universes: dict[str, dict[str, Any]] = {}

    for p in payloads:
        u = p["asset_set"]
        mu_is = p["mu_is"]
        sigma_is = p["sigma_is"]
        mu_oos = p["mu_oos"]
        sigma_oos = p["sigma_oos"]
        rf_is_scalar = p["rf_is_scalar"]
        rf_oos_scalar = p["rf_oos_scalar"]

        w_tan_is = tangency_weights(mu_is, sigma_is, rf_is_scalar)
        w_gmv_is = gmv_weights(sigma_is)
        w_tan_oos = tangency_weights(mu_oos, sigma_oos, rf_oos_scalar)
        w_gmv_oos = gmv_weights(sigma_oos)

        # Full frontiers in IS/OOS
        max_mu = max(
            float(portfolio_stats(w_tan_is, mu_is, sigma_is)[0]),
            float(portfolio_stats(w_tan_oos, mu_oos, sigma_oos)[0]),
        )
        mu_max = max(0.001, 1.2 * max_mu)
        curve_is = build_frontier_curve(mu_is, sigma_is, n_points=n_points, mu_min=0.0, mu_max=mu_max)
        curve_oos = build_frontier_curve(mu_oos, sigma_oos, n_points=n_points, mu_min=0.0, mu_max=mu_max)

        c_curves_is = {}
        c_curves_oos = {}
        model_defs: list[tuple[str, float | None, np.ndarray]] = [("unconstrained", None, w_tan_is)]
        for w_min in w_min_levels:
            w_c_is = tangency_weights_constrained(mu_is, sigma_is, rf=rf_is_scalar, w_min=float(w_min))
            model_defs.append(("constrained", float(w_min), w_c_is))
            c_means_is, c_vols_is, _ = efficient_frontier_constrained(
                mu_is, sigma_is, n_points=n_points, mu_min=None, mu_max=None, w_min=float(w_min)
            )
            c_means_oos, c_vols_oos, _ = efficient_frontier_constrained(
                mu_oos, sigma_oos, n_points=n_points, mu_min=None, mu_max=None, w_min=float(w_min)
            )
            c_curves_is[f"w_i>={w_min:.2f}"] = {"means": c_means_is, "vols": c_vols_is}
            c_curves_oos[f"w_i>={w_min:.2f}"] = {"means": c_means_oos, "vols": c_vols_oos}

            # Constrained recombination diagnostic:
            # best convex combination of IS constrained GMV/TAN in OOS versus OOS constrained TAN.
            w_gmv_is_c = gmv_weights_constrained(sigma_is, w_min=float(w_min))
            w_tan_oos_c = tangency_weights_constrained(mu_oos, sigma_oos, rf=rf_oos_scalar, w_min=float(w_min))
            alphas = np.linspace(0.0, 1.0, 1001)
            best_alpha = 0.0
            best_sr = -np.inf
            best_w = w_gmv_is_c.copy()
            for a in alphas:
                w_mix = a * w_c_is + (1.0 - a) * w_gmv_is_c
                sr_mix = portfolio_stats(w_mix, mu_oos, sigma_oos)[0] - rf_oos_scalar
                vol_mix = portfolio_stats(w_mix, mu_oos, sigma_oos)[1]
                sr_mix = float(sr_mix / vol_mix) if vol_mix > 0 else -np.inf
                if sr_mix > best_sr:
                    best_sr = sr_mix
                    best_alpha = float(a)
                    best_w = w_mix

            m_mix, v_mix = portfolio_stats(best_w, mu_oos, sigma_oos)
            m_tan_oos_c, v_tan_oos_c = portfolio_stats(w_tan_oos_c, mu_oos, sigma_oos)
            sr_tan_oos_c = float((m_tan_oos_c - rf_oos_scalar) / v_tan_oos_c) if v_tan_oos_c > 0 else np.nan
            recomb_rows.append(
                {
                    "asset_set": u,
                    "constraint_label": f"w_i>={w_min:.2f}",
                    "w_min": float(w_min),
                    "alpha_mix_is_tan": best_alpha,
                    "sr_mix_oos": float(best_sr),
                    "sr_oos_tan_constrained": sr_tan_oos_c,
                    "sr_gap_vs_oos_tan": float(best_sr - sr_tan_oos_c) if np.isfinite(sr_tan_oos_c) else np.nan,
                    "vol_mix_oos": float(v_mix),
                    "mean_mix_oos": float(m_mix),
                    "vol_oos_tan_constrained": float(v_tan_oos_c),
                    "mean_oos_tan_constrained": float(m_tan_oos_c),
                    "weight_l2_distance": float(np.linalg.norm(best_w - w_tan_oos_c)),
                }
            )

        point_book = {
            "is_opt": {},
            "oos_opt": {},
            "is_selected_on_is": {},
            "is_selected_on_oos": {},
        }
        for tag, w, mu_ref, sigma_ref in [
            ("gmv", w_gmv_is, mu_is, sigma_is),
            ("tan", w_tan_is, mu_is, sigma_is),
        ]:
            m, v = portfolio_stats(w, mu_ref, sigma_ref)
            point_book["is_opt"][tag] = {"mean": float(m), "vol": float(v)}
        for tag, w, mu_ref, sigma_ref in [
            ("gmv", w_gmv_oos, mu_oos, sigma_oos),
            ("tan", w_tan_oos, mu_oos, sigma_oos),
        ]:
            m, v = portfolio_stats(w, mu_ref, sigma_ref)
            point_book["oos_opt"][tag] = {"mean": float(m), "vol": float(v)}

        for model_name, w_min_val, w in model_defs:
            label = model_name if w_min_val is None else f"w_i>={w_min_val:.2f}"
            m_is, v_is = portfolio_stats(w, mu_is, sigma_is)
            m_oos, v_oos = portfolio_stats(w, mu_oos, sigma_oos)
            point_book["is_selected_on_is"][label] = {"mean": float(m_is), "vol": float(v_is)}
            point_book["is_selected_on_oos"][label] = {"mean": float(m_oos), "vol": float(v_oos)}

            r_is = p["is_arr"] @ w
            r_oos = p["oos_arr"] @ w
            sr = summarize_sharpe_is_oos(r_is, r_oos, rf_is=p["rf_is_series"], rf_oos=p["rf_oos_series"])
            jk = jobson_korkie_test(r_is, r_oos, rf1=rf_is_scalar, rf2=rf_oos_scalar)
            lw = ledoit_wolf_test(r_is, r_oos, rf1=rf_is_scalar, rf2=rf_oos_scalar, n_boot=2000)
            rep = frontier_replication_alpha(w, w_tan_oos, w_gmv_oos)

            sharpe_rows.append(
                {
                    "asset_set": u,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    **sr,
                }
            )
            jk_rows.append(
                {
                    "asset_set": u,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "sr_is": jk.sharpe1,
                    "sr_oos": jk.sharpe2,
                    "z_stat": jk.statistic,
                    "pvalue": jk.pvalue_two_sided,
                }
            )
            lw_rows.append(
                {
                    "asset_set": u,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "sr_is": lw.sharpe1,
                    "sr_oos": lw.sharpe2,
                    "diff": lw.difference,
                    "ci_low": lw.ci_low,
                    "ci_high": lw.ci_high,
                    "pvalue": lw.pvalue_two_sided,
                }
            )
            rep_rows.append(
                {
                    "asset_set": u,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "alpha": rep.alpha,
                    "r_squared": rep.r_squared,
                    "residual_norm": rep.residual_norm,
                }
            )
            weight_rows.append(
                {
                    "asset_set": u,
                    "model": model_name,
                    "constraint_label": label,
                    "w_min": np.nan if w_min_val is None else float(w_min_val),
                    "min_w": float(np.min(w)),
                    "max_w": float(np.max(w)),
                    "gross_leverage": float(np.sum(np.abs(w))),
                    "effective_n": float((np.sum(w**2) ** -1) if np.sum(w**2) > 0 else np.nan),
                }
            )

        plot_universes[u] = {
            "is": {"curve": curve_is, "constrained_curves": c_curves_is},
            "oos": {"curve": curve_oos, "constrained_curves": c_curves_oos},
            "points": point_book,
        }

    return {
        "inputs": {
            "window_policy": "is_oos_split",
            "end_is": end_is,
            "start_oos": start_oos,
            "end_oos": end_oos,
            "split_policy": split_policy,
            "constraint_levels": list(w_min_levels),
            "universes": ["industries", "ff5", "proxy5"],
            "industry_is_start": ind_idx_is.min().strftime("%Y-%m"),
            "industry_is_end": ind_idx_is.max().strftime("%Y-%m"),
            "industry_oos_start": ind_idx_oos.min().strftime("%Y-%m"),
            "industry_oos_end": ind_idx_oos.max().strftime("%Y-%m"),
            "ff5_proxy5_is_start": fp_idx_is.min().strftime("%Y-%m"),
            "ff5_proxy5_is_end": fp_idx_is.max().strftime("%Y-%m"),
            "ff5_proxy5_oos_start": fp_idx_oos.min().strftime("%Y-%m"),
            "ff5_proxy5_oos_end": fp_idx_oos.max().strftime("%Y-%m"),
        },
        "summary_tables": {
            "sharpe_is_oos": pd.DataFrame(sharpe_rows),
            "jk_tests": pd.DataFrame(jk_rows),
            "lw_tests": pd.DataFrame(lw_rows),
            "frontier_replication": pd.DataFrame(rep_rows),
            "weight_diagnostics": pd.DataFrame(weight_rows),
            "constrained_recombination": pd.DataFrame(recomb_rows),
        },
        "plot_data": {"universes": plot_universes},
        "diagnostics": {
            "split_warning": " | ".join(warnings) if warnings else None,
            "note": "Scope 8.2 evaluates IS-selected constrained portfolios in OOS with full IS/OOS frontier plots.",
        },
    }


def run_scope8_3_shrinkage_persistence(
    ind_gross: pd.DataFrame,
    ff5_excess: pd.DataFrame,
    proxy_returns: pd.DataFrame,
    rf_gross: pd.Series,
    constraint_levels: tuple[float, ...] = (0.0, -0.25),
    end_is: str = "2002-12",
    start_oos: str = "2003-01",
    end_oos: str = "2025-12",
    cov_shrink: float | str = "ledoit_wolf",
) -> dict[str, Any]:
    """
    Scope 8.3: IS-only BS shrinkage persistence for TAN (industries, FF5, Proxy5).
    """
    if ind_gross.empty:
        raise ValueError("ind_gross is empty")
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

    w_min_levels = tuple(dict.fromkeys(float(x) for x in constraint_levels))
    if len(w_min_levels) == 0:
        raise ValueError("constraint_levels cannot be empty.")

    warnings = []
    split_policy = "requested_dates"

    # Industries split
    ind_idx = ind_gross.index.intersection(rf_gross.index).sort_values()
    ind_idx_is, ind_idx_oos, pol_ind, warn_ind = _split_index_with_fallback(ind_idx, end_is, start_oos, end_oos)
    split_policy = "fallback_half_split" if pol_ind == "fallback_half_split" else split_policy
    if warn_ind:
        warnings.append(f"industries: {warn_ind}")
    ind_net = (ind_gross.loc[ind_idx] - 1.0).astype(float)
    rf_ind = (rf_gross.loc[ind_idx] - 1.0).astype(float)
    ind_is = ind_net.loc[ind_idx_is]
    ind_oos = ind_net.loc[ind_idx_oos]
    rf_ind_is = rf_ind.loc[ind_idx_is].values
    rf_ind_oos = rf_ind.loc[ind_idx_oos].values

    # FF5/proxy5 split
    fp_idx = ff5_excess.index.intersection(proxy_returns.index).intersection(rf_gross.index).sort_values()
    fp_idx_is, fp_idx_oos, pol_fp, warn_fp = _split_index_with_fallback(fp_idx, end_is, start_oos, end_oos)
    split_policy = "fallback_half_split" if pol_fp == "fallback_half_split" else split_policy
    if warn_fp:
        warnings.append(f"ff5/proxy5: {warn_fp}")
    ff5 = ff5_excess.loc[fp_idx].astype(float)
    rf_fp = (rf_gross.loc[fp_idx] - 1.0).astype(float)
    proxy5_excess = proxy_returns.loc[fp_idx, ["Mkt", "SMB", "HML", "RMW", "CMA"]].astype(float).sub(rf_fp, axis=0)
    ff5_is, ff5_oos = ff5.loc[fp_idx_is], ff5.loc[fp_idx_oos]
    proxy5_is, proxy5_oos = proxy5_excess.loc[fp_idx_is], proxy5_excess.loc[fp_idx_oos]

    # Build universes with rf policies
    universes = {
        "industries": {
            "is_arr": ind_is.values,
            "oos_arr": ind_oos.values,
            "rf_is_series": rf_ind_is,
            "rf_oos_series": rf_ind_oos,
            "rf_is_scalar": float(np.mean(rf_ind_is)),
            "rf_oos_scalar": float(np.mean(rf_ind_oos)),
        },
        "ff5": {
            "is_arr": ff5_is.values,
            "oos_arr": ff5_oos.values,
            "rf_is_series": 0.0,
            "rf_oos_series": 0.0,
            "rf_is_scalar": 0.0,
            "rf_oos_scalar": 0.0,
        },
        "proxy5": {
            "is_arr": proxy5_is.values,
            "oos_arr": proxy5_oos.values,
            "rf_is_series": 0.0,
            "rf_oos_series": 0.0,
            "rf_is_scalar": 0.0,
            "rf_oos_scalar": 0.0,
        },
    }

    sharpe_rows = []
    jk_rows = []
    lw_rows = []
    rep_rows = []
    weight_rows = []
    shrink_rows = []
    split_note = "IS-only shrinkage estimation."

    for u_name, u in universes.items():
        is_arr = u["is_arr"]
        oos_arr = u["oos_arr"]
        rf_is_scalar = float(u["rf_is_scalar"])
        rf_oos_scalar = float(u["rf_oos_scalar"])
        rf_is_series = u["rf_is_series"]
        rf_oos_series = u["rf_oos_series"]

        mu_is_s, sigma_is_s = compute_moments_from_net(is_arr)
        mu_oos_s, sigma_oos_s = compute_moments_from_net(oos_arr)
        t_is = is_arr.shape[0]

        bs = bayes_stein_means(mu_is_s, sigma_is_s, T=t_is)
        mu_is_bs = bs.mu_bs
        if isinstance(cov_shrink, str):
            mode = cov_shrink.strip().lower()
            if mode in {"ledoit_wolf", "lw"}:
                sigma_is_bs, cov_lambda = shrink_covariance_ledoit_wolf(is_arr)
                cov_method = "ledoit_wolf"
            else:
                raise ValueError("cov_shrink string mode must be one of {'ledoit_wolf','lw'}")
        else:
            cov_lambda = float(cov_shrink)
            sigma_is_bs = shrink_covariance_identity(sigma_is_s, shrinkage=cov_lambda)
            cov_method = "identity_fixed_lambda"

        model_params = {
            "sample": (mu_is_s, sigma_is_s),
            "bs_mu": (mu_is_bs, sigma_is_s),
            "bs_mu_cov": (mu_is_bs, sigma_is_bs),
        }

        # OOS anchors from sample OOS moments (realized opportunity set)
        w_tan_oos = tangency_weights(mu_oos_s, sigma_oos_s, rf_oos_scalar)
        w_gmv_oos = gmv_weights(sigma_oos_s)

        shrink_rows.append(
            {
                "asset_set": u_name,
                "t_is": int(t_is),
                "mu_shrinkage_intensity": float(bs.shrinkage_intensity),
                "cov_shrink_method": cov_method,
                "cov_shrink_effective_lambda": float(cov_lambda),
            }
        )

        for est_name, (mu_is, sigma_is) in model_params.items():
            model_defs: list[tuple[str, float | None, np.ndarray]] = [
                ("unconstrained", None, tangency_weights(mu_is, sigma_is, rf_is_scalar))
            ]
            for w_min in w_min_levels:
                model_defs.append(
                    (
                        "constrained",
                        float(w_min),
                        tangency_weights_constrained(mu_is, sigma_is, rf=rf_is_scalar, w_min=float(w_min)),
                    )
                )

            for model_name, w_min_val, w in model_defs:
                label = model_name if w_min_val is None else f"w_i>={w_min_val:.2f}"
                r_is = is_arr @ w
                r_oos = oos_arr @ w
                sr = summarize_sharpe_is_oos(r_is, r_oos, rf_is=rf_is_series, rf_oos=rf_oos_series)
                jk = jobson_korkie_test(r_is, r_oos, rf1=rf_is_scalar, rf2=rf_oos_scalar)
                lw = ledoit_wolf_test(r_is, r_oos, rf1=rf_is_scalar, rf2=rf_oos_scalar, n_boot=2000)
                rep = frontier_replication_alpha(w, w_tan_oos, w_gmv_oos)

                sharpe_rows.append(
                    {
                        "asset_set": u_name,
                        "estimator": est_name,
                        "model": model_name,
                        "constraint_label": label,
                        "w_min": np.nan if w_min_val is None else float(w_min_val),
                        **sr,
                    }
                )
                jk_rows.append(
                    {
                        "asset_set": u_name,
                        "estimator": est_name,
                        "model": model_name,
                        "constraint_label": label,
                        "w_min": np.nan if w_min_val is None else float(w_min_val),
                        "sr_is": jk.sharpe1,
                        "sr_oos": jk.sharpe2,
                        "z_stat": jk.statistic,
                        "pvalue": jk.pvalue_two_sided,
                    }
                )
                lw_rows.append(
                    {
                        "asset_set": u_name,
                        "estimator": est_name,
                        "model": model_name,
                        "constraint_label": label,
                        "w_min": np.nan if w_min_val is None else float(w_min_val),
                        "sr_is": lw.sharpe1,
                        "sr_oos": lw.sharpe2,
                        "diff": lw.difference,
                        "ci_low": lw.ci_low,
                        "ci_high": lw.ci_high,
                        "pvalue": lw.pvalue_two_sided,
                    }
                )
                rep_rows.append(
                    {
                        "asset_set": u_name,
                        "estimator": est_name,
                        "model": model_name,
                        "constraint_label": label,
                        "w_min": np.nan if w_min_val is None else float(w_min_val),
                        "alpha": rep.alpha,
                        "r_squared": rep.r_squared,
                        "residual_norm": rep.residual_norm,
                    }
                )
                weight_rows.append(
                    {
                        "asset_set": u_name,
                        "estimator": est_name,
                        "model": model_name,
                        "constraint_label": label,
                        "w_min": np.nan if w_min_val is None else float(w_min_val),
                        "min_w": float(np.min(w)),
                        "max_w": float(np.max(w)),
                        "gross_leverage": float(np.sum(np.abs(w))),
                        "effective_n": float((np.sum(w**2) ** -1) if np.sum(w**2) > 0 else np.nan),
                    }
                )

    return {
        "inputs": {
            "window_policy": "is_oos_split",
            "end_is": end_is,
            "start_oos": start_oos,
            "end_oos": end_oos,
            "split_policy": split_policy,
            "constraint_levels": list(w_min_levels),
            "universes": ["industries", "ff5", "proxy5"],
            "shrinkage_policy": "estimate_on_is_only",
            "is_method_note": split_note,
        },
        "summary_tables": {
            "sharpe_is_oos": pd.DataFrame(sharpe_rows),
            "jk_tests": pd.DataFrame(jk_rows),
            "lw_tests": pd.DataFrame(lw_rows),
            "frontier_replication": pd.DataFrame(rep_rows),
            "weight_diagnostics": pd.DataFrame(weight_rows),
            "shrinkage_diagnostics": pd.DataFrame(shrink_rows),
        },
        "plot_data": {},
        "diagnostics": {
            "split_warning": " | ".join(warnings) if warnings else None,
            "note": "Scope 8.3 evaluates IS-only BS shrinkage impact on OOS persistence of tangency.",
        },
    }
