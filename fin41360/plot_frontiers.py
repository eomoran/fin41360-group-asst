"""
Plotting helpers for frontier workflows.

Each function accepts workflow output dicts and returns matplotlib Figure
objects so notebooks can stay concise and declarative.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from .plot_styles import SCOPE3_PLOT_STYLE, scope6_legend_handles, scope6_panel_legend_handles, style


def _frontier_arrays(curve: dict, gmv_mean: float, efficient_frontier_only: bool = True):
    """Return frontier x/y arrays, optionally clipped to the efficient branch."""
    vols = np.asarray(curve["vols"], dtype=float)
    means = np.asarray(curve["means"], dtype=float)
    if not efficient_frontier_only:
        return vols, means

    mask = means >= float(gmv_mean)
    if not np.any(mask):
        return vols, means
    return vols[mask], means[mask]


def _compute_xmax(
    curves: dict,
    points: dict,
    x_mode: str = "frontier",
    frontier_mult: float = 1.2,
    tan_mult: float = 2.0,
) -> float:
    """
    Compute x-axis upper bound from a consistent plotting policy.

    x_mode:
    - 'frontier': x_max = frontier_mult * max frontier volatility
    - 'tangency': x_max = tan_mult * max tangency volatility
    - 'max': max of the two rules above

    TODO(LATER): consider adding 'gmv_center' policy so x-limits are framed
    relative to sample GMV / tangency distance for visual consistency across
    report figures.
    TODO(LATER): add cone-consistent bounds based on risky frontier + risk-free
    feasible cone (CML geometry), then treat x_mode as display override.
    """
    max_curve_vol = max(float(curves[k]["vols"].max()) for k in curves.keys())
    max_tan_vol = max(float(points[k]["tan"]["vol"]) for k in points.keys())

    frontier_x = frontier_mult * max_curve_vol
    tan_x = tan_mult * max_tan_vol

    if x_mode == "frontier":
        return frontier_x
    if x_mode == "tangency":
        return tan_x
    if x_mode == "max":
        return max(frontier_x, tan_x)
    raise ValueError("x_mode must be one of: 'frontier', 'tangency', 'max'")


def _compute_ymax(
    curves: dict,
    points: dict,
    y_mode: str = "frontier",
    frontier_mult: float = 1.2,
    tan_mult: float = 1.2,
) -> float:
    """
    Compute y-axis upper bound from a consistent plotting policy.

    y_mode:
    - 'frontier': y_max = frontier_mult * max frontier mean
    - 'tangency': y_max = tan_mult * max tangency mean
    - 'max': max of the two rules above
    TODO(LATER): align y-bounds with the same cone-consistent construction used
    for x-bounds so risky frontier and CML share a coherent plotted domain.
    """
    max_curve_mean = max(float(curves[k]["means"].max()) for k in curves.keys())
    max_tan_mean = max(float(points[k]["tan"]["mean"]) for k in points.keys())

    frontier_y = frontier_mult * max_curve_mean
    tan_y = tan_mult * max_tan_mean

    if y_mode == "frontier":
        return frontier_y
    if y_mode == "tangency":
        return tan_y
    if y_mode == "max":
        return max(frontier_y, tan_y)
    raise ValueError("y_mode must be one of: 'frontier', 'tangency', 'max'")


def plot_scope2_overlay(
    scope2_result: dict,
    x_mode: str = "frontier",
    y_mode: str = "frontier",
    frontier_mult: float = 1.2,
    tan_mult: float = 2.0,
    efficient_frontier_only: bool = True,
):
    """
    Plot sample vs Bayes-Stein frontiers for Scope 2.

    TODO(LATER): evaluate fixed x-axis ranges across related scope figures
    (e.g. all Question 2/3 charts) to improve side-by-side comparability.
    """
    curves = scope2_result["plot_data"]["curves"]
    points = scope2_result["plot_data"]["points"]
    range_data = scope2_result["plot_data"].get("range", {})
    sample_meta = scope2_result.get("inputs", {})

    fig, ax = plt.subplots(figsize=(8, 5))

    line_handles = []
    universe_color = SCOPE3_PLOT_STYLE["universe"]["industry"]["color"]
    plotted_curves = {}
    for label in ("sample", "bs_mean", "bs_mean_cov"):
        est_style = SCOPE3_PLOT_STYLE["estimator"][label]
        line_style = style("frontier", label)
        line_style["color"] = universe_color
        line_style["linestyle"] = est_style["linestyle"]
        line_handle = mlines.Line2D(
            [],
            [],
            color=universe_color,
            linestyle=line_style.get("linestyle", "-"),
            linewidth=line_style.get("linewidth", 1.5),
            label=line_style.get("label", label),
        )
        line_handles.append(line_handle)
        x, y = _frontier_arrays(curves[label], points[label]["gmv"]["mean"], efficient_frontier_only)
        plotted_curves[label] = {"vols": x, "means": y}
        ax.plot(x, y, **line_style)
        gmv_style = style("gmv", label)
        tan_style = style("tan", label)
        gmv_style["color"] = universe_color
        tan_style["color"] = universe_color
        ax.scatter(points[label]["gmv"]["vol"], points[label]["gmv"]["mean"], **gmv_style)
        ax.scatter(points[label]["tan"]["vol"], points[label]["tan"]["mean"], **tan_style)

    # x-axis policy is configurable so we can keep figure geometry consistent across scopes.
    x_max = float(max(np.max(plotted_curves[k]["vols"]) for k in plotted_curves.keys()))
    y_max = float(range_data.get("mu_max", _compute_ymax(
        curves,
        points,
        y_mode=y_mode,
        frontier_mult=frontier_mult,
        tan_mult=tan_mult,
    )))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Volatility (monthly std dev)")
    ax.set_ylabel("Expected return (monthly, net)")
    if sample_meta:
        title = (
            f"Question 2: 30-Industry MV Frontier "
            f"({sample_meta.get('start', '')} to {sample_meta.get('end', '')})"
        )
    else:
        title = "Question 2: 30-Industry MV Frontier"
    ax.set_title(title)
    leg1 = ax.legend(handles=line_handles, loc="upper left", title="Frontier")
    ax.add_artist(leg1)
    marker_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=8, label="GMV"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=8, label="Tangency"),
    ]
    ax.legend(handles=marker_handles, loc="lower right", title="Marker")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_scope3_overlay(
    scope3_result: dict,
    x_mode: str = "frontier",
    y_mode: str = "frontier",
    frontier_mult: float = 1.2,
    tan_mult: float = 2.0,
    efficient_frontier_only: bool = True,
):
    """
    Plot Scope 3 frontiers: 30 industries vs 30 stocks across estimators.
    """
    curves = scope3_result["plot_data"]["curves"]
    points = scope3_result["plot_data"]["points"]
    range_data = scope3_result["plot_data"].get("range", {})
    meta = scope3_result.get("inputs", {})

    fig, ax = plt.subplots(figsize=(9, 6))

    est_order = ("sample", "bs_mean", "bs_mean_cov")
    univ_order = ("industry", "stock")
    line_handles = []
    plotted_curves_nested = {}

    for est in est_order:
        plotted_curves_nested[est] = {}
        for univ in univ_order:
            est_style = SCOPE3_PLOT_STYLE["estimator"][est]
            univ_style = SCOPE3_PLOT_STYLE["universe"][univ]
            curve = curves[est][univ]
            lbl = f"{est_style['label']} — {univ_style['label']}"
            x, y = _frontier_arrays(curve, points[est][univ]["gmv"]["mean"], efficient_frontier_only)
            plotted_curves_nested[est][univ] = {"vols": x, "means": y}
            ax.plot(
                x,
                y,
                color=univ_style["color"],
                linestyle=est_style["linestyle"],
                linewidth=1.5,
                label=lbl,
                zorder=2,
            )
            line_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=univ_style["color"],
                    linestyle=est_style["linestyle"],
                    linewidth=1.5,
                    label=lbl,
                )
            )

    for est in est_order:
        for univ in univ_order:
            color = SCOPE3_PLOT_STYLE["universe"][univ]["color"]
            gmv = points[est][univ]["gmv"]
            tan = points[est][univ]["tan"]
            ax.scatter(
                [gmv["vol"]],
                [gmv["mean"]],
                color=color,
                marker=SCOPE3_PLOT_STYLE["portfolio_marker"]["GMV"],
                s=60,
                zorder=5,
                label="_nolegend_",
            )
            ax.scatter(
                [tan["vol"]],
                [tan["mean"]],
                color=color,
                marker=SCOPE3_PLOT_STYLE["portfolio_marker"]["TAN"],
                s=60,
                zorder=5,
                label="_nolegend_",
            )

    flat_curves = {}
    flat_points = {}
    for est in est_order:
        for univ in univ_order:
            key = f"{est}_{univ}"
            flat_curves[key] = plotted_curves_nested[est][univ]
            flat_points[key] = points[est][univ]
    x_max = float(max(flat_curves[k]["vols"].max() for k in flat_curves.keys()))
    y_max = float(range_data.get("mu_max", _compute_ymax(
        flat_curves,
        flat_points,
        y_mode=y_mode,
        frontier_mult=frontier_mult,
        tan_mult=tan_mult,
    )))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Volatility (monthly std dev)")
    ax.set_ylabel("Expected return (monthly, net)")
    if meta:
        n_ind = meta.get("n_assets_industry")
        n_stk = meta.get("n_assets_stock")
        if n_ind is not None and n_stk is not None:
            ax.set_title(
                f"Question 3: {n_ind} Industries vs {n_stk} Stocks "
                f"({meta.get('common_start', '')} to {meta.get('common_end', '')})"
            )
        else:
            ax.set_title(
                "Question 3: Industries vs Stocks "
                f"({meta.get('common_start', '')} to {meta.get('common_end', '')})"
            )
    else:
        ax.set_title("Question 3: Industries vs Stocks")

    leg1 = ax.legend(handles=line_handles, loc="upper left", fontsize=8, title="Frontier")
    ax.add_artist(leg1)
    marker_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=8, label="GMV"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=8, label="Tangency"),
    ]
    ax.legend(handles=marker_handles, loc="lower right", fontsize=8, title="Marker")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_scope5_overlay(scope5_result: dict, efficient_frontier_only: bool = True):
    """
    Single-panel Scope 5 overlay: industries vs FF3 vs FF5 in excess space.
    """
    curves = scope5_result["plot_data"]["curves"]
    points = scope5_result["plot_data"]["points"]
    cml = scope5_result["plot_data"]["cml"]
    range_data = scope5_result["plot_data"].get("range", {})
    meta = scope5_result.get("inputs", {})

    fig, ax = plt.subplots(figsize=(9, 6))

    for key in ("industries", "ff3", "ff5"):
        x, y = _frontier_arrays(curves[key], points[key]["gmv"]["mean"], efficient_frontier_only)
        ax.plot(x, y, **style("frontier", key))
        ax.plot(cml[key]["vols"], cml[key]["means"], **style("cml", key, label=f"CML ({key}, rf=0)"))
        ax.scatter(points[key]["gmv"]["vol"], points[key]["gmv"]["mean"], **style("gmv", key))
        ax.scatter(points[key]["tan"]["vol"], points[key]["tan"]["mean"], **style("tan", key))

    x_max = float(range_data.get("x_max", max(curves[k]["vols"].max() for k in curves.keys())))
    y_max = float(range_data.get("mu_max", max(curves[k]["means"].max() for k in curves.keys())))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Volatility (monthly, excess)")
    ax.set_ylabel("Expected excess return (monthly)")
    if meta:
        ax.set_title(
            "Question 5: Industries vs FF3 vs FF5 "
            f"({meta.get('start', '')} to {meta.get('end', '')})"
        )
    else:
        ax.set_title("Question 5: Industries vs FF3 vs FF5")
    leg1 = ax.legend(loc="upper left", fontsize=8, title="Frontier/CML")
    ax.add_artist(leg1)
    marker_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=8, label="GMV"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=8, label="Tangency"),
    ]
    ax.legend(handles=marker_handles, loc="lower right", fontsize=8, title="Marker")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_scope4_with_rf(scope4_result: dict, efficient_frontier_only: bool = True):
    """
    Scope 4 plot: industries-only risky frontier plus CML with risk-free asset.
    """
    curve = scope4_result["plot_data"]["curve"]
    cml = scope4_result["plot_data"]["cml"]
    range_data = scope4_result["plot_data"].get("range", {})
    points = scope4_result["plot_data"]["points"]
    meta = scope4_result.get("inputs", {})

    fig, ax = plt.subplots(figsize=(8, 5))
    frontier_style = style("frontier", "industries", label="30 industries only (risky set)")
    frontier_style["alpha"] = 0.75
    frontier_style["linestyle"] = "-"
    x, y = _frontier_arrays(curve, points["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **frontier_style)

    # Keep same colour family as the risky frontier because both come from the
    # same underlying asset universe; differentiate primarily by linewidth/label.
    cml_style = style("cml", "industries", label="CML (efficient set with risk-free)")
    cml_style["alpha"] = 0.9
    cml_style["linewidth"] = 1.5
    cml_style["linestyle"] = "-"
    ax.plot(cml["vols"], cml["means"], **cml_style)
    ax.scatter(points["gmv"]["vol"], points["gmv"]["mean"], **style("gmv", "industries"))
    ax.scatter(points["tan"]["vol"], points["tan"]["mean"], **style("tan", "industries"))

    rf_mean = float(meta.get("rf_mean", 0.0))
    ax.axhline(rf_mean, color="gray", linestyle=":", alpha=0.85, label=f"Risk-free ({rf_mean:.2%})")

    x_max = float(range_data.get("x_max", max(float(curve["vols"].max()), float(cml["vols"].max()))))
    y_max = float(range_data.get("mu_max", max(float(curve["means"].max()), float(cml["means"].max()))))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Volatility (monthly std dev)")
    ax.set_ylabel("Expected return (monthly, net)")
    if meta:
        ax.set_title(
            "Question 4: 30 Industries + Risk-Free "
            f"({meta.get('start', '')} to {meta.get('end', '')})"
        )
    else:
        ax.set_title("Question 4: 30 Industries + Risk-Free")
    leg1 = ax.legend(loc="upper left", fontsize=8, title="Frontier/CML")
    ax.add_artist(leg1)
    marker_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=8, label="GMV"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=8, label="Tangency"),
    ]
    ax.legend(handles=marker_handles, loc="lower right", fontsize=8, title="Marker")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_scope6_panels(
    scope6_result: dict,
    limit_basis: str = "workflow",
    tan_vol_rank: int = 1,
    limit_mult: float = 1.2,
    efficient_frontier_only: bool = True,
):
    """
    Two-panel Scope 6 plot: FF3 vs Proxy-3 and FF5 vs Proxy-5.

    limit_basis:
        - "workflow" (default): use scope6_result plot_data['range'] limits
        - "tangency_vol_rank": set x/y limits from the tangency portfolio with the given
          rank (1=largest vol, 2=second largest vol, etc.) across ff3/proxy3/ff5/proxy5
    limit_mult:
        Scalar applied to ranked tangency vol/mean when limit_basis='tangency_vol_rank'.
    """
    curves = scope6_result["plot_data"]["curves"]
    points = scope6_result["plot_data"]["points"]
    cml = scope6_result["plot_data"]["cml"]
    range_data = scope6_result["plot_data"].get("range", {})
    meta = scope6_result.get("inputs", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: FF3 vs Proxy-3
    ax = axes[0]
    x, y = _frontier_arrays(curves["ff3"], points["ff3"]["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **style("frontier", "ff3"))
    x, y = _frontier_arrays(curves["proxy3"], points["proxy3"]["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **style("frontier", "ff3", proxy=True))
    ax.plot(cml["ff3"]["vols"], cml["ff3"]["means"], **style("cml", "ff3", label="CML FF3 (rf=0)"))
    ax.plot(cml["proxy3"]["vols"], cml["proxy3"]["means"], **style("cml", "ff3", proxy=True, label="CML Proxy-3 (rf=0)"))
    ax.scatter(points["ff3"]["gmv"]["vol"], points["ff3"]["gmv"]["mean"], **style("gmv", "ff3"))
    ax.scatter(points["ff3"]["tan"]["vol"], points["ff3"]["tan"]["mean"], **style("tan", "ff3"))
    ax.scatter(points["proxy3"]["gmv"]["vol"], points["proxy3"]["gmv"]["mean"], **style("gmv", "ff3", proxy=True))
    ax.scatter(points["proxy3"]["tan"]["vol"], points["proxy3"]["tan"]["mean"], **style("tan", "ff3", proxy=True))
    ax.set_xlabel("Volatility (monthly, excess)")
    ax.set_ylabel("Expected excess return (monthly)")
    ax.set_title("Scope 6: FF3 vs Proxy-3")
    ax.grid(True, linestyle="--", alpha=0.4)
    line_handles, marker_handles = scope6_panel_legend_handles("ff3")
    leg1 = ax.legend(handles=line_handles, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="lower right", fontsize=8)

    # Right: FF5 vs Proxy-5
    ax = axes[1]
    x, y = _frontier_arrays(curves["ff5"], points["ff5"]["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **style("frontier", "ff5"))
    x, y = _frontier_arrays(curves["proxy5"], points["proxy5"]["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **style("frontier", "ff5", proxy=True))
    ax.plot(cml["ff5"]["vols"], cml["ff5"]["means"], **style("cml", "ff5", label="CML FF5 (rf=0)"))
    ax.plot(cml["proxy5"]["vols"], cml["proxy5"]["means"], **style("cml", "ff5", proxy=True, label="CML Proxy-5 (rf=0)"))
    ax.scatter(points["ff5"]["gmv"]["vol"], points["ff5"]["gmv"]["mean"], **style("gmv", "ff5"))
    ax.scatter(points["ff5"]["tan"]["vol"], points["ff5"]["tan"]["mean"], **style("tan", "ff5"))
    ax.scatter(points["proxy5"]["gmv"]["vol"], points["proxy5"]["gmv"]["mean"], **style("gmv", "ff5", proxy=True))
    ax.scatter(points["proxy5"]["tan"]["vol"], points["proxy5"]["tan"]["mean"], **style("tan", "ff5", proxy=True))
    ax.set_xlabel("Volatility (monthly, excess)")
    ax.set_ylabel("Expected excess return (monthly)")
    ax.set_title("Scope 6: FF5 vs Proxy-5")
    ax.grid(True, linestyle="--", alpha=0.4)
    line_handles, marker_handles = scope6_panel_legend_handles("ff5")
    leg1 = ax.legend(handles=line_handles, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="lower right", fontsize=8)

    if limit_basis == "workflow":
        x_max = float(range_data.get("x_max", 0.0))
        y_max = float(range_data.get("mu_max", 0.0))
    elif limit_basis == "tangency_vol_rank":
        tan_points = sorted(
            [
                (float(points["ff3"]["tan"]["vol"]), float(points["ff3"]["tan"]["mean"]), "ff3"),
                (float(points["proxy3"]["tan"]["vol"]), float(points["proxy3"]["tan"]["mean"]), "proxy3"),
                (float(points["ff5"]["tan"]["vol"]), float(points["ff5"]["tan"]["mean"]), "ff5"),
                (float(points["proxy5"]["tan"]["vol"]), float(points["proxy5"]["tan"]["mean"]), "proxy5"),
            ],
            key=lambda t: t[0],
            reverse=True,
        )
        rank = max(1, int(tan_vol_rank))
        idx = min(rank - 1, len(tan_points) - 1)
        x_max, y_max, _ = tan_points[idx]
        x_max *= float(limit_mult)
        y_max *= float(limit_mult)
    else:
        raise ValueError("limit_basis must be one of {'workflow', 'tangency_vol_rank'}")

    for ax in axes:
        if x_max > 0:
            ax.set_xlim(0, x_max)
        if y_max > 0:
            ax.set_ylim(0, y_max)

    if meta:
        title = f"Question 6: FF Factors vs Practical Proxies ({meta.get('start', '')} to {meta.get('end', '')})"
        if limit_basis == "tangency_vol_rank":
            title += f" [limits: tan vol rank {max(1, int(tan_vol_rank))}]"
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_scope6_overlay(
    scope6_result: dict,
    limit_basis: str = "ff5",
    tan_vol_rank: int = 1,
    limit_mult: float = 1.2,
    efficient_frontier_only: bool = True,
    x_limit_basis: str | None = None,
):
    """
    Single-panel Scope 6 overlay (FF3/Proxy3 and FF5/Proxy5).

    limit_basis:
        - "ff5" (default): use FF5 + Proxy5 curves/CMLs
        - "tangency_vol_rank": set x/y limits from the tangency portfolio with the given
          rank (1=largest vol, 2=second largest vol, etc.) across ff3/proxy3/ff5/proxy5
    limit_mult:
        Scalar applied to ranked tangency vol/mean when limit_basis='tangency_vol_rank'.
    x_limit_basis:
        Deprecated alias for limit_basis (kept for compatibility).
    """
    curves = scope6_result["plot_data"]["curves"]
    points = scope6_result["plot_data"]["points"]
    cml = scope6_result["plot_data"]["cml"]
    meta = scope6_result.get("inputs", {})

    fig, ax = plt.subplots(figsize=(9, 6))

    for key, base in [("ff3", "ff3"), ("proxy3", "ff3"), ("ff5", "ff5"), ("proxy5", "ff5")]:
        is_proxy = key.startswith("proxy")
        x, y = _frontier_arrays(curves[key], points[key]["gmv"]["mean"], efficient_frontier_only)
        ax.plot(x, y, **style("frontier", base, proxy=is_proxy, label="_nolegend_"))
        ax.plot(cml[key]["vols"], cml[key]["means"], **style("cml", base, proxy=is_proxy, label="_nolegend_"))
        ax.scatter(points[key]["gmv"]["vol"], points[key]["gmv"]["mean"], **style("gmv", base, proxy=is_proxy, label="_nolegend_"))
        ax.scatter(points[key]["tan"]["vol"], points[key]["tan"]["mean"], **style("tan", base, proxy=is_proxy, label="_nolegend_"))

    if x_limit_basis is not None:
        limit_basis = x_limit_basis

    if limit_basis == "ff5":
        # FF5-based axes (FF5 + Proxy5 only) to avoid FF3 instability dominating the frame.
        x_max = max(
            float(curves["ff5"]["vols"].max()),
            float(curves["proxy5"]["vols"].max()),
            float(cml["ff5"]["vols"].max()),
            float(cml["proxy5"]["vols"].max()),
        )
        y_max = max(
            float(curves["ff5"]["means"].max()),
            float(curves["proxy5"]["means"].max()),
            float(cml["ff5"]["means"].max()),
            float(cml["proxy5"]["means"].max()),
        )
    elif limit_basis == "tangency_vol_rank":
        tan_points = sorted(
            [
                (float(points["ff3"]["tan"]["vol"]), float(points["ff3"]["tan"]["mean"]), "ff3"),
                (float(points["proxy3"]["tan"]["vol"]), float(points["proxy3"]["tan"]["mean"]), "proxy3"),
                (float(points["ff5"]["tan"]["vol"]), float(points["ff5"]["tan"]["mean"]), "ff5"),
                (float(points["proxy5"]["tan"]["vol"]), float(points["proxy5"]["tan"]["mean"]), "proxy5"),
            ],
            key=lambda t: t[0],
            reverse=True,
        )
        rank = max(1, int(tan_vol_rank))
        idx = min(rank - 1, len(tan_points) - 1)
        x_max, y_max, _ = tan_points[idx]
        x_max *= float(limit_mult)
        y_max *= float(limit_mult)
    else:
        raise ValueError("limit_basis must be one of {'ff5', 'tangency_vol_rank'}")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Volatility (monthly, excess)")
    ax.set_ylabel("Expected excess return (monthly)")
    if meta:
        title = f"Question 6: FF Factors vs Practical Proxies ({meta.get('start', '')} to {meta.get('end', '')})"
        if limit_basis == "tangency_vol_rank":
            title += f" [limits: tan vol rank {max(1, int(tan_vol_rank))}]"
        ax.set_title(title)
    else:
        ax.set_title("Question 6: FF Factors vs Practical Proxies")
    ax.grid(True, linestyle="--", alpha=0.4)
    line_handles, marker_handles = scope6_legend_handles()
    leg1 = ax.legend(handles=line_handles, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
