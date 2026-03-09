"""
Generate report and appendix assets for Scopes 2-7.

Policy implemented:
- Report figures use plot-specific readability limits aligned to the notebook.
- Appendix figures use anchored-origin common limits for comparability.
- Scope 6 report figures use tangency-vol-rank framing (rank 2).
- Exported figures include chart titles for easier review outside LaTeX.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fin41360.process_french import load_industry_30_monthly, load_ff3_monthly, load_ff5_monthly
from fin41360.stock_data import load_stock_returns_monthly, load_scope6_proxy_returns_monthly
from fin41360.workflows import (
    run_scope2_industries_sample_vs_bs,
    run_scope3_sensitivity_with_and_without_coal,
    run_scope4_industries_with_rf,
    run_scope5_industries_vs_ff,
    run_scope6_ff_vs_proxies,
    run_scope7_is_oos_tests,
)
from fin41360.plot_frontiers import (
    plot_scope2_overlay,
    plot_scope3_overlay,
    plot_scope4_with_rf,
    plot_scope5_overlay,
    plot_scope6_overlay,
    plot_scope6_panels,
    set_plot_defaults,
)
from fin41360.report_assets import (
    save_figures,
    save_scope_summary_tables,
    save_scope_presentation_tables,
)


def _configure_local_cache_env() -> None:
    """Route Matplotlib/fontconfig caches into the workspace when unset."""
    cache_root = PROJECT_ROOT / ".cache"
    mpl_cache = cache_root / "matplotlib"
    fontconfig_cache = cache_root / "fontconfig"
    cache_root.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    fontconfig_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("FONTCONFIG_PATH", "/opt/homebrew/etc/fonts")
    os.environ.setdefault("FONTCONFIG_FILE", "/opt/homebrew/etc/fonts/fonts.conf")


_configure_local_cache_env()

import numpy as np
import seaborn as sns


def _scope3_common_limits(scope3_with: dict, scope3_drop: dict, *, efficient_frontier_only: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    x_vals: list[float] = []
    y_vals: list[float] = []
    for result in (scope3_with, scope3_drop):
        curves = result["plot_data"]["curves"]
        points = result["plot_data"]["points"]
        for est in ("sample", "bs_mean", "bs_mean_cov"):
            for univ in ("industry", "stock"):
                curve = curves[est][univ]
                vols = np.asarray(curve["vols"], dtype=float)
                means = np.asarray(curve["means"], dtype=float)
                if efficient_frontier_only:
                    gmv_mean = float(points[est][univ]["gmv"]["mean"])
                    mask = means >= gmv_mean
                    if np.any(mask):
                        vols = vols[mask]
                        means = means[mask]
                x_vals.extend(vols.tolist())
                y_vals.extend(means.tolist())
                x_vals.extend([float(points[est][univ]["gmv"]["vol"]), float(points[est][univ]["tan"]["vol"])])
                y_vals.extend([float(points[est][univ]["gmv"]["mean"]), float(points[est][univ]["tan"]["mean"])])
    return (0.0, float(np.nanmax(x_vals))), (0.0, float(np.nanmax(y_vals)))


def _scope2_common_limits(scope2: dict, *, efficient_frontier_only: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    curves = scope2["plot_data"]["curves"]
    points = scope2["plot_data"]["points"]
    x_vals: list[float] = []
    y_vals: list[float] = []
    for label in ("sample", "bs_mean", "bs_mean_cov"):
        vols = np.asarray(curves[label]["vols"], dtype=float)
        means = np.asarray(curves[label]["means"], dtype=float)
        if efficient_frontier_only:
            mask = means >= float(points[label]["gmv"]["mean"])
            if np.any(mask):
                vols = vols[mask]
                means = means[mask]
        x_vals.extend(vols.tolist())
        y_vals.extend(means.tolist())
        x_vals.extend([float(points[label]["gmv"]["vol"]), float(points[label]["tan"]["vol"])])
        y_vals.extend([float(points[label]["gmv"]["mean"]), float(points[label]["tan"]["mean"])])
    return (0.0, float(np.nanmax(x_vals))), (0.0, float(np.nanmax(y_vals)))


def _scope2_report_limits(scope2: dict, *, efficient_frontier_only: bool, tan_mult: float = 1.2) -> tuple[tuple[float, float], tuple[float, float]]:
    curves = scope2["plot_data"]["curves"]
    points = scope2["plot_data"]["points"]
    x_front: list[float] = []
    y_front: list[float] = []
    x_tan: list[float] = []
    y_tan: list[float] = []
    for k in ("sample", "bs_mean", "bs_mean_cov"):
        vols = np.asarray(curves[k]["vols"], dtype=float)
        means = np.asarray(curves[k]["means"], dtype=float)
        if efficient_frontier_only:
            mask = means >= float(points[k]["gmv"]["mean"])
            if np.any(mask):
                vols = vols[mask]
                means = means[mask]
        x_front.extend(vols.tolist())
        y_front.extend(means.tolist())
        x_tan.append(float(points[k]["tan"]["vol"]))
        y_tan.append(float(points[k]["tan"]["mean"]))
    return (float(np.nanmin(x_front)), tan_mult * float(np.nanmax(x_tan))), (float(np.nanmin(y_front)), tan_mult * float(np.nanmax(y_tan)))


def _scope3_report_limits(
    scope3_result: dict,
    *,
    efficient_frontier_only: bool,
    tan_mult: float = 1.2,
    use_tan_cap: bool = True,
    tan_rank: int = 1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    curves = scope3_result["plot_data"]["curves"]
    points = scope3_result["plot_data"]["points"]
    x_front: list[float] = []
    y_front: list[float] = []
    x_tan: list[float] = []
    y_tan: list[float] = []
    for est in ("sample", "bs_mean", "bs_mean_cov"):
        for univ in ("industry", "stock"):
            curve = curves[est][univ]
            vols = np.asarray(curve["vols"], dtype=float)
            means = np.asarray(curve["means"], dtype=float)
            if efficient_frontier_only:
                mask = means >= float(points[est][univ]["gmv"]["mean"])
                if np.any(mask):
                    vols = vols[mask]
                    means = means[mask]
            x_front.extend(vols.tolist())
            y_front.extend(means.tolist())
            x_tan.append(float(points[est][univ]["tan"]["vol"]))
            y_tan.append(float(points[est][univ]["tan"]["mean"]))
    if use_tan_cap:
        rank = max(1, int(tan_rank))
        x_tan_sorted = sorted(x_tan, reverse=True)
        y_tan_sorted = sorted(y_tan, reverse=True)
        x_ref = x_tan_sorted[min(rank - 1, len(x_tan_sorted) - 1)]
        y_ref = y_tan_sorted[min(rank - 1, len(y_tan_sorted) - 1)]
        x_max = tan_mult * float(x_ref)
        y_max = tan_mult * float(y_ref)
    else:
        x_max = float(np.nanmax(x_front))
        y_max = float(np.nanmax(y_front))
    return (float(np.nanmin(x_front)), x_max), (float(np.nanmin(y_front)), y_max)


def _scope3_report_common_limits(
    scope3_with: dict,
    scope3_drop: dict,
    *,
    efficient_frontier_only: bool,
    tan_mult: float = 1.2,
    tan_rank: int = 1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Common Scope 3 report limits anchored at the minimum GMV x/y across both cases."""
    x_lims: list[tuple[float, float]] = []
    y_lims: list[tuple[float, float]] = []
    for result in (scope3_with, scope3_drop):
        x_lim, y_lim = _scope3_report_limits(
            result,
            efficient_frontier_only=efficient_frontier_only,
            tan_mult=tan_mult,
            use_tan_cap=True,
            tan_rank=tan_rank,
        )
        x_lims.append(x_lim)
        y_lims.append(y_lim)
    x_min = min(limit[0] for limit in x_lims)
    x_max = min(limit[1] for limit in x_lims)
    y_min = min(limit[0] for limit in y_lims)
    y_max = min(limit[1] for limit in y_lims)
    return (x_min, x_max), (y_min, y_max)


def _scope4_report_limits(scope4: dict, *, efficient_frontier_only: bool, tan_mult: float = 1.2) -> tuple[tuple[float, float], tuple[float, float]]:
    curve = scope4["plot_data"]["curve"]
    points = scope4["plot_data"]["points"]
    vols = np.asarray(curve["vols"], dtype=float)
    means = np.asarray(curve["means"], dtype=float)
    if efficient_frontier_only:
        mask = means >= float(points["gmv"]["mean"])
        if np.any(mask):
            vols = vols[mask]
            means = means[mask]
    x_min = float(np.nanmin(vols))
    y_min = float(np.nanmin(means))
    x_max = tan_mult * float(points["tan"]["vol"])
    y_max = tan_mult * float(points["tan"]["mean"])
    return (x_min, x_max), (y_min, y_max)


def _scope5_report_limits(scope5: dict, *, efficient_frontier_only: bool, tan_mult: float = 1.2) -> tuple[tuple[float, float], tuple[float, float]]:
    curves = scope5["plot_data"]["curves"]
    points = scope5["plot_data"]["points"]
    x_front: list[float] = []
    y_front: list[float] = []
    x_tan: list[float] = []
    y_tan: list[float] = []
    for k in ("industries", "ff3", "ff5"):
        vols = np.asarray(curves[k]["vols"], dtype=float)
        means = np.asarray(curves[k]["means"], dtype=float)
        if efficient_frontier_only:
            mask = means >= float(points[k]["gmv"]["mean"])
            if np.any(mask):
                vols = vols[mask]
                means = means[mask]
        x_front.extend(vols.tolist())
        y_front.extend(means.tolist())
        x_tan.append(float(points[k]["tan"]["vol"]))
        y_tan.append(float(points[k]["tan"]["mean"]))
    return (float(np.nanmin(x_front)), tan_mult * float(np.nanmax(x_tan))), (float(np.nanmin(y_front)), tan_mult * float(np.nanmax(y_tan)))


def _scope6_ff5_limits(scope6: dict, *, efficient_frontier_only: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    points = scope6["plot_data"]["points"]
    _ = efficient_frontier_only
    limit_mult = 1.2
    x_max = limit_mult * float(points["ff5"]["tan"]["vol"])
    y_max = limit_mult * float(points["ff5"]["tan"]["mean"])
    return (0.0, x_max), (0.0, y_max)


def _scope6_report_limits(scope6: dict, *, efficient_frontier_only: bool, tan_mult: float = 1.2) -> tuple[tuple[float, float], tuple[float, float]]:
    curves = scope6["plot_data"]["curves"]
    points = scope6["plot_data"]["points"]
    x_front: list[float] = []
    y_front: list[float] = []
    for k in ("ff3", "proxy3", "ff5", "proxy5"):
        vols = np.asarray(curves[k]["vols"], dtype=float)
        means = np.asarray(curves[k]["means"], dtype=float)
        if efficient_frontier_only:
            mask = means >= float(points[k]["gmv"]["mean"])
            if np.any(mask):
                vols = vols[mask]
                means = means[mask]
        x_front.extend(vols.tolist())
        y_front.extend(means.tolist())
    # Exception policy: cap report scope6 using FF5 tangency (exclude FF3 outlier behavior).
    x_max = tan_mult * float(points["ff5"]["tan"]["vol"])
    y_max = tan_mult * float(points["ff5"]["tan"]["mean"])
    return (float(np.nanmin(x_front)), x_max), (float(np.nanmin(y_front)), y_max)


def main() -> None:
    _configure_local_cache_env()
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")

    bs_target = "gmv"
    cov_shrink = "ledoit_wolf"
    scope6_limit_basis = "tangency_vol_rank"
    scope6_tan_vol_rank = 2

    report_overlay_figsize = (10.0, 7.0)
    report_panel_figsize = (14.0, 6.8)
    efficient_frontier_only = True

    set_plot_defaults(
        overlay_layout="single_column",
        panel_layout="full_width",
        show_titles=True,
        figsize_scale=2,
    )

    ind = load_industry_30_monthly(start="1980-01", end="2025-12")
    ff3, rf_gross = load_ff3_monthly(start="1980-01", end="2025-12")
    ff5, _ = load_ff5_monthly(start="1980-01", end="2025-12")
    stocks = load_stock_returns_monthly(start="1980-01", end="2025-12", use_cache=True, source="auto")
    proxy_rets = load_scope6_proxy_returns_monthly(start="2000-01", end="2025-12", use_cache=True, source="auto", refresh=False)

    scope2 = run_scope2_industries_sample_vs_bs(
        ind_gross=ind,
        rf_gross=rf_gross,
        bs_target=bs_target,
        cov_shrink=cov_shrink,
    )
    scope3_sens = run_scope3_sensitivity_with_and_without_coal(
        ind_gross=ind,
        stocks_gross=stocks,
        rf_gross=rf_gross,
        bs_target=bs_target,
        cov_shrink=cov_shrink,
    )
    scope3_with = scope3_sens["with_coal_30"]
    scope3_drop = scope3_sens["drop_coal_29"]
    scope4 = run_scope4_industries_with_rf(ind_gross=ind, rf_gross=rf_gross)
    scope5 = run_scope5_industries_vs_ff(ind_gross=ind, ff3_excess=ff3, ff5_excess=ff5, rf_gross=rf_gross)
    scope6 = run_scope6_ff_vs_proxies(ff3_excess=ff3, ff5_excess=ff5, proxy_returns=proxy_rets, rf_gross=rf_gross, n_points=2000)
    scope7 = run_scope7_is_oos_tests(ind_gross=ind, ff5_excess=ff5, rf_gross=rf_gross, end_is="2002-12", start_oos="2003-01", end_oos="2025-12")

    scope2_report_xlim, scope2_report_ylim = _scope2_report_limits(scope2, efficient_frontier_only=efficient_frontier_only, tan_mult=1.2)
    scope3_report_xlim, _scope3_report_ylim = _scope3_report_common_limits(
        scope3_with,
        scope3_drop,
        efficient_frontier_only=efficient_frontier_only,
        tan_mult=1.2,
        tan_rank=1,
    )
    scope3_report_ylim = (0.0, min(0.04, _scope3_report_ylim[1]))
    scope3_with_report_xlim, scope3_with_report_ylim = scope3_report_xlim, scope3_report_ylim
    scope3_drop_report_xlim, scope3_drop_report_ylim = scope3_report_xlim, scope3_report_ylim
    scope4_report_xlim, scope4_report_ylim = _scope4_report_limits(scope4, efficient_frontier_only=efficient_frontier_only, tan_mult=1.2)
    scope5_report_xlim, scope5_report_ylim = _scope5_report_limits(scope5, efficient_frontier_only=efficient_frontier_only, tan_mult=1.2)
    scope6_report_xlim, scope6_report_ylim = _scope6_report_limits(scope6, efficient_frontier_only=efficient_frontier_only, tan_mult=1.2)

    # Report figures (explicit limits; xmin/ymin from frontiers, xmax/ymax from tangency policy).
    fig_report_scope2 = plot_scope2_overlay(
        scope2,
        efficient_frontier_only=True,
        figsize=report_overlay_figsize,
        xlim=scope2_report_xlim,
        ylim=scope2_report_ylim,
        show_title=True,
    )
    fig_report_scope3_with = plot_scope3_overlay(
        scope3_with,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        xlim=scope3_with_report_xlim,
        ylim=scope3_with_report_ylim,
        show_title=True,
    )
    fig_report_scope3_drop = plot_scope3_overlay(
        scope3_drop,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        xlim=scope3_drop_report_xlim,
        ylim=scope3_drop_report_ylim,
        show_title=True,
    )
    fig_report_scope4_ymin0 = plot_scope4_with_rf(
        scope4,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        xlim=scope4_report_xlim,
        ylim=(0.0, scope4_report_ylim[1]),
        show_title=True,
    )
    fig_report_scope4_xymin0 = plot_scope4_with_rf(
        scope4,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        xlim=(0.0, scope4_report_xlim[1]),
        ylim=(0.0, scope4_report_ylim[1]),
        show_title=True,
    )
    fig_report_scope5 = plot_scope5_overlay(
        scope5,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        xlim=(0.0, scope5_report_xlim[1]),
        ylim=(0.0, scope5_report_ylim[1]),
        show_title=True,
    )
    fig_report_scope5_non_anchored_origin = plot_scope5_overlay(
        scope5,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        xlim=scope5_report_xlim,
        ylim=scope5_report_ylim,
        show_title=True,
    )
    fig_report_scope6_overlay = plot_scope6_overlay(
        scope6,
        efficient_frontier_only=efficient_frontier_only,
        limit_basis=scope6_limit_basis,
        tan_vol_rank=scope6_tan_vol_rank,
        limit_mult=1.2,
        figsize=report_overlay_figsize,
        xlim=(0.0, scope6_report_xlim[1]),
        ylim=(0.0, scope6_report_ylim[1]),
        show_title=True,
    )
    fig_report_scope6_overlay_non_anchored_origin = plot_scope6_overlay(
        scope6,
        efficient_frontier_only=efficient_frontier_only,
        limit_basis=scope6_limit_basis,
        tan_vol_rank=scope6_tan_vol_rank,
        limit_mult=1.2,
        figsize=report_overlay_figsize,
        xlim=scope6_report_xlim,
        ylim=scope6_report_ylim,
        show_title=True,
    )
    fig_report_scope6_panels = plot_scope6_panels(
        scope6,
        efficient_frontier_only=efficient_frontier_only,
        limit_basis=scope6_limit_basis,
        tan_vol_rank=scope6_tan_vol_rank,
        limit_mult=1.2,
        figsize=report_panel_figsize,
        xlim=(0.0, scope6_report_xlim[1]),
        ylim=(0.0, scope6_report_ylim[1]),
        show_title=True,
    )
    fig_report_scope6_panels_non_anchored_origin = plot_scope6_panels(
        scope6,
        efficient_frontier_only=efficient_frontier_only,
        limit_basis=scope6_limit_basis,
        tan_vol_rank=scope6_tan_vol_rank,
        limit_mult=1.2,
        figsize=report_panel_figsize,
        xlim=scope6_report_xlim,
        ylim=scope6_report_ylim,
        show_title=True,
    )

    # Appendix comparability figures (anchored-origin common limits).
    scope2_common_xlim, scope2_common_ylim = _scope2_common_limits(scope2, efficient_frontier_only=True)
    _scope3_xlim, _scope3_ylim = _scope3_common_limits(scope3_with, scope3_drop, efficient_frontier_only=efficient_frontier_only)
    _scope6_xlim, _scope6_ylim = _scope6_ff5_limits(scope6, efficient_frontier_only=efficient_frontier_only)

    fig_appendix_scope2 = plot_scope2_overlay(
        scope2,
        efficient_frontier_only=True,
        figsize=report_overlay_figsize,
        anchor_origin=True,
        xlim=scope2_common_xlim,
        ylim=scope2_common_ylim,
        show_title=False,
    )
    fig_appendix_scope3_with = plot_scope3_overlay(
        scope3_with,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        anchor_origin=True,
        xlim=scope2_common_xlim,
        ylim=scope2_common_ylim,
        show_title=False,
    )
    fig_appendix_scope3_drop = plot_scope3_overlay(
        scope3_drop,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        anchor_origin=True,
        xlim=scope2_common_xlim,
        ylim=scope2_common_ylim,
        show_title=False,
    )
    fig_appendix_scope5 = plot_scope5_overlay(
        scope5,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        anchor_origin=True,
        xlim=scope2_common_xlim,
        ylim=scope2_common_ylim,
        show_title=False,
    )
    fig_appendix_scope6_overlay = plot_scope6_overlay(
        scope6,
        efficient_frontier_only=efficient_frontier_only,
        limit_basis="ff5",
        figsize=report_overlay_figsize,
        anchor_origin=True,
        xlim=scope2_common_xlim,
        ylim=scope2_common_ylim,
        show_title=False,
    )
    fig_appendix_scope6_panels = plot_scope6_panels(
        scope6,
        efficient_frontier_only=efficient_frontier_only,
        limit_basis="ff5",
        figsize=report_panel_figsize,
        anchor_origin=True,
        xlim=scope2_common_xlim,
        ylim=scope2_common_ylim,
        show_title=False,
    )
    # Exception appendix plots (non-common limits) for unstable/outlier cases.
    fig_appendix_scope3_with_non_common = plot_scope3_overlay(
        scope3_with,
        efficient_frontier_only=efficient_frontier_only,
        figsize=report_overlay_figsize,
        anchor_origin=True,
        show_title=False,
    )
    fig_appendix_scope6_ff3_non_common = plot_scope6_panels(
        scope6,
        efficient_frontier_only=efficient_frontier_only,
        limit_basis="tangency_vol_rank",
        tan_vol_rank=1,
        figsize=report_panel_figsize,
        anchor_origin=True,
        show_title=False,
    )

    saved = save_figures(
        {
            "report_scope2": fig_report_scope2,
            "report_scope3_with_coal": fig_report_scope3_with,
            "report_scope3_drop_coal": fig_report_scope3_drop,
            "report_scope4_ymin0": fig_report_scope4_ymin0,
            "report_scope4_xymin0": fig_report_scope4_xymin0,
            "report_scope4": fig_report_scope4_xymin0,
            "report_scope5": fig_report_scope5,
            "report_scope5_non_anchored_origin": fig_report_scope5_non_anchored_origin,
            "report_scope6_overlay": fig_report_scope6_overlay,
            "report_scope6_panels": fig_report_scope6_panels,
            "report_scope6_overlay_non_anchored_origin": fig_report_scope6_overlay_non_anchored_origin,
            "report_scope6_panels_non_anchored_origin": fig_report_scope6_panels_non_anchored_origin,
            "appendix_scope2_anchored_origin_common_limits": fig_appendix_scope2,
            "appendix_scope3_with_coal_anchored_origin_common_limits": fig_appendix_scope3_with,
            "appendix_scope3_drop_coal_anchored_origin_common_limits": fig_appendix_scope3_drop,
            "appendix_scope5_anchored_origin_common_limits": fig_appendix_scope5,
            "appendix_scope6_overlay_anchored_origin_common_limits": fig_appendix_scope6_overlay,
            "appendix_scope6_panels_anchored_origin_common_limits": fig_appendix_scope6_panels,
            "appendix_scope3_with_coal_anchored_origin_non_common_limits": fig_appendix_scope3_with_non_common,
            "appendix_scope6_ff3_anchored_origin_non_common_limits": fig_appendix_scope6_ff3_non_common,
        },
        overwrite=True,
        fmt="png",
        dpi=300,
        close=True,
    )

    # Tables (summary + presentation)
    save_scope_summary_tables(scope2, "scope2", overwrite=True, index=False, float_format="%.5f")
    save_scope_summary_tables(scope3_with, "scope3_with_coal", overwrite=True, index=False, float_format="%.5f")
    save_scope_summary_tables(scope3_drop, "scope3_drop_coal", overwrite=True, index=False, float_format="%.5f")
    save_scope_summary_tables(scope4, "scope4", overwrite=True, index=False, float_format="%.5f")
    save_scope_summary_tables(scope5, "scope5", overwrite=True, index=False, float_format="%.5f")
    save_scope_summary_tables(scope6, "scope6", overwrite=True, index=False, float_format="%.5f")
    save_scope_summary_tables(scope7, "scope7", overwrite=True, index=False, float_format="%.5f")
    save_scope_presentation_tables(
        scope2_result=scope2,
        scope3_sensitivity_result=scope3_sens,
        scope5_result=scope5,
        scope6_result=scope6,
        scope7_result=scope7,
        overwrite=True,
        index=False,
        float_format="%.5f",
    )

    for k, p in saved.items():
        print(k, "->", p)


if __name__ == "__main__":
    main()
