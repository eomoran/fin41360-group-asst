"""
Plotting helpers for frontier workflows.

Each function accepts workflow output dicts and returns matplotlib Figure
objects so notebooks can stay concise and declarative.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from .plot_styles import SCOPE3_PLOT_STYLE, style


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
        ax.plot(curves[label]["vols"], curves[label]["means"], **line_style)
        gmv_style = style("gmv", label)
        tan_style = style("tan", label)
        gmv_style["color"] = universe_color
        tan_style["color"] = universe_color
        ax.scatter(points[label]["gmv"]["vol"], points[label]["gmv"]["mean"], **gmv_style)
        ax.scatter(points[label]["tan"]["vol"], points[label]["tan"]["mean"], **tan_style)

    # x-axis policy is configurable so we can keep figure geometry consistent across scopes.
    x_max = float(max(curves[k]["vols"].max() for k in curves.keys()))
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

    for est in est_order:
        for univ in univ_order:
            est_style = SCOPE3_PLOT_STYLE["estimator"][est]
            univ_style = SCOPE3_PLOT_STYLE["universe"][univ]
            curve = curves[est][univ]
            lbl = f"{est_style['label']} — {univ_style['label']}"
            ax.plot(
                curve["vols"],
                curve["means"],
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
            flat_curves[key] = curves[est][univ]
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
        ax.set_title(
            "Question 3: 30 Industries vs 30 Stocks "
            f"({meta.get('common_start', '')} to {meta.get('common_end', '')})"
        )
    else:
        ax.set_title("Question 3: 30 Industries vs 30 Stocks")

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


def plot_scope5_overlay(scope5_result: dict):
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
        ax.plot(curves[key]["vols"], curves[key]["means"], **style("frontier", key))
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


def plot_scope4_with_rf(scope4_result: dict):
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
    ax.plot(curve["vols"], curve["means"], **frontier_style)

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


def plot_scope6_panels(scope6_result: dict):
    """
    Placeholder for Scope 6 plotting.
    """
    raise NotImplementedError("Scope 6 plotting scaffolded but not implemented yet.")
