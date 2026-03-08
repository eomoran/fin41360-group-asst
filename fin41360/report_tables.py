"""
Presentation-table builders for report-facing Scope 2-7 outputs.

These helpers reshape workflow summary tables into comparison-first layouts with:
- stable sorting/grouping,
- signed delta columns,
- optional absolute Sharpe deltas for closeness ranking,
- presentation scaling for mean/vol columns (default: x1e3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ESTIMATOR_ORDER = {"sample": 0, "bs_mean": 1, "bs_mean_cov": 2}
PORTFOLIO_ORDER = {"GMV": 0, "TAN": 1}
UNIVERSE_ORDER_S3 = {"industry": 0, "stock": 1}
UNIVERSE_ORDER_S5 = {"industries": 0, "ff3": 1, "ff5": 2}
SCENARIO_ORDER = {"with_coal_30": 0, "drop_coal_29": 1}


def _with_signed_deltas(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    baseline_col: str,
    baseline_value: str,
) -> pd.DataFrame:
    out = df.copy()
    for col in ("mean", "vol", "sharpe"):
        base = (
            out[out[baseline_col] == baseline_value]
            .groupby(group_cols, dropna=False)[col]
            .first()
            .rename(f"_base_{col}")
        )
        out = out.merge(base, on=group_cols, how="left")
        out[f"delta_{col}"] = out[col] - out[f"_base_{col}"]
    out["abs_delta_sharpe"] = out["delta_sharpe"].abs()
    return out.drop(columns=[c for c in out.columns if c.startswith("_base_")])


def _apply_scale(df: pd.DataFrame, *, scale: float) -> pd.DataFrame:
    out = df.copy()
    for col in ("mean", "vol", "delta_mean", "delta_vol"):
        if col in out.columns:
            out[f"{col}_x1e3"] = out[col] * scale
    return out


def build_scope2_table(scope2_result: dict[str, Any], *, scale: float = 1_000.0) -> pd.DataFrame:
    src = scope2_result["summary_tables"]["gmv_tan"].copy()
    src = _with_signed_deltas(
        src,
        group_cols=["portfolio"],
        baseline_col="label",
        baseline_value="sample",
    )
    src["portfolio_order"] = src["portfolio"].map(PORTFOLIO_ORDER)
    src["estimator_order"] = src["label"].map(ESTIMATOR_ORDER)
    src = src.sort_values(["portfolio_order", "estimator_order"]).reset_index(drop=True)
    src = _apply_scale(src, scale=scale)
    return src[
        [
            "portfolio",
            "label",
            "mean_x1e3",
            "vol_x1e3",
            "sharpe",
            "delta_mean_x1e3",
            "delta_vol_x1e3",
            "delta_sharpe",
            "abs_delta_sharpe",
        ]
    ].rename(columns={"label": "estimator"})


def build_scope3_tables(scope3_sensitivity_result: dict[str, Any], *, scale: float = 1_000.0) -> dict[str, pd.DataFrame]:
    with_coal = scope3_sensitivity_result["with_coal_30"]["summary_tables"]["gmv_tan"].copy()
    with_coal["scenario"] = "with_coal_30"
    drop_coal = scope3_sensitivity_result["drop_coal_29"]["summary_tables"]["gmv_tan"].copy()
    drop_coal["scenario"] = "drop_coal_29"
    src = pd.concat([with_coal, drop_coal], ignore_index=True)

    src = _with_signed_deltas(
        src,
        group_cols=["scenario", "portfolio", "universe"],
        baseline_col="estimator",
        baseline_value="sample",
    )
    src["scenario_order"] = src["scenario"].map(SCENARIO_ORDER)
    src["portfolio_order"] = src["portfolio"].map(PORTFOLIO_ORDER)
    src["universe_order"] = src["universe"].map(UNIVERSE_ORDER_S3)
    src["estimator_order"] = src["estimator"].map(ESTIMATOR_ORDER)
    src = src.sort_values(
        ["scenario_order", "portfolio_order", "universe_order", "estimator_order"]
    ).reset_index(drop=True)
    src = _apply_scale(src, scale=scale)

    summary = src[
        [
            "scenario",
            "portfolio",
            "universe",
            "estimator",
            "mean_x1e3",
            "vol_x1e3",
            "sharpe",
            "delta_mean_x1e3",
            "delta_vol_x1e3",
            "delta_sharpe",
            "abs_delta_sharpe",
        ]
    ]

    window = scope3_sensitivity_result["industry_stability"]["summary_window_table"].copy()
    return {"comparison": summary, "window_summary": window}


def build_scope5_table(scope5_result: dict[str, Any], *, scale: float = 1_000.0) -> pd.DataFrame:
    src = scope5_result["summary_tables"]["gmv_tan"].copy()
    src = _with_signed_deltas(
        src,
        group_cols=["portfolio"],
        baseline_col="series",
        baseline_value="industries",
    )
    src["portfolio_order"] = src["portfolio"].map(PORTFOLIO_ORDER)
    src["universe_order"] = src["series"].map(UNIVERSE_ORDER_S5)
    src = src.sort_values(["portfolio_order", "universe_order"]).reset_index(drop=True)
    src = _apply_scale(src, scale=scale)
    return src[
        [
            "portfolio",
            "series",
            "mean_x1e3",
            "vol_x1e3",
            "sharpe",
            "delta_mean_x1e3",
            "delta_vol_x1e3",
            "delta_sharpe",
            "abs_delta_sharpe",
        ]
    ].rename(columns={"series": "universe", "delta_mean_x1e3": "delta_mean_vs_ind_x1e3", "delta_vol_x1e3": "delta_vol_vs_ind_x1e3", "delta_sharpe": "delta_sharpe_vs_ind", "abs_delta_sharpe": "abs_delta_sharpe_vs_ind"})


def build_scope6_pairwise_table(scope6_result: dict[str, Any], *, scale: float = 1_000.0) -> pd.DataFrame:
    src = scope6_result["summary_tables"]["gmv_tan"].copy()
    rows: list[dict[str, Any]] = []

    def _row_for_pair(portfolio: str, ff_key: str, proxy_key: str, pair_name: str) -> dict[str, Any]:
        ff = src[(src["series"] == ff_key) & (src["portfolio"] == portfolio)].iloc[0]
        px = src[(src["series"] == proxy_key) & (src["portfolio"] == portfolio)].iloc[0]
        d_sharpe = float(px["sharpe"] - ff["sharpe"])
        return {
            "portfolio": portfolio,
            "pair": pair_name,
            "ff_mean_x1e3": float(ff["mean"]) * scale,
            "proxy_mean_x1e3": float(px["mean"]) * scale,
            "delta_mean_x1e3": (float(px["mean"]) - float(ff["mean"])) * scale,
            "ff_vol_x1e3": float(ff["vol"]) * scale,
            "proxy_vol_x1e3": float(px["vol"]) * scale,
            "delta_vol_x1e3": (float(px["vol"]) - float(ff["vol"])) * scale,
            "ff_sharpe": float(ff["sharpe"]),
            "proxy_sharpe": float(px["sharpe"]),
            "delta_sharpe": d_sharpe,
            "abs_delta_sharpe": abs(d_sharpe),
        }

    for p in ("GMV", "TAN"):
        rows.append(_row_for_pair(p, "ff3", "proxy3", "FF3 vs Proxy3"))
        rows.append(_row_for_pair(p, "ff5", "proxy5", "FF5 vs Proxy5"))

    out = pd.DataFrame(rows)
    out["portfolio_order"] = out["portfolio"].map(PORTFOLIO_ORDER)
    out["pair_order"] = out["pair"].map({"FF3 vs Proxy3": 0, "FF5 vs Proxy5": 1})
    out = out.sort_values(["portfolio_order", "pair_order"]).reset_index(drop=True)
    return out.drop(columns=["portfolio_order", "pair_order"])


def build_scope7_decision_table(scope7_result: dict[str, Any]) -> pd.DataFrame:
    sr = scope7_result["summary_tables"]["sharpe_is_oos"].copy()
    jk = scope7_result["summary_tables"]["jk_tests"][["portfolio", "pvalue"]].rename(columns={"pvalue": "jk_pvalue"})
    lw = scope7_result["summary_tables"]["lw_tests"][
        ["portfolio", "pvalue", "ci_low", "ci_high"]
    ].rename(columns={"pvalue": "lw_pvalue"})

    out = sr.merge(jk, on="portfolio", how="left").merge(lw, on="portfolio", how="left")

    def _inference(row: pd.Series) -> str:
        if float(row["lw_pvalue"]) < 0.05 and float(row["delta"]) < 0:
            return "IS Sharpe > OOS Sharpe (significant)"
        if float(row["lw_pvalue"]) < 0.05 and float(row["delta"]) > 0:
            return "OOS Sharpe > IS Sharpe (significant)"
        return "No significant Sharpe change"

    out["inference_flag"] = out.apply(_inference, axis=1)
    out["portfolio_order"] = out["portfolio"].map(
        {"30 ind GMV": 0, "30 ind TAN": 1, "FF5 GMV": 2, "FF5 TAN": 3}
    )
    out = out.sort_values("portfolio_order").reset_index(drop=True)
    return out[
        [
            "portfolio",
            "sr_is",
            "sr_oos",
            "delta",
            "jk_pvalue",
            "lw_pvalue",
            "ci_low",
            "ci_high",
            "inference_flag",
        ]
    ].rename(columns={"delta": "delta_sr", "ci_low": "lw_ci_low", "ci_high": "lw_ci_high"})
