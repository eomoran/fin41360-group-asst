"""
Plotting helpers for frontier workflows.

Each function accepts workflow output dicts and returns matplotlib Figure
objects so notebooks can stay concise and declarative.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from .plot_styles import style


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


def plot_scope2_overlay(
    scope2_result: dict,
    x_mode: str = "frontier",
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
    sample_meta = scope2_result.get("inputs", {})

    fig, ax = plt.subplots(figsize=(8, 5))

    line_handles = []
    for label in ("sample", "bs_mean", "bs_mean_cov"):
        line_style = style("frontier", label)
        line_handle = mlines.Line2D(
            [],
            [],
            color=line_style.get("color"),
            linestyle=line_style.get("linestyle", "-"),
            linewidth=line_style.get("linewidth", 1.5),
            label=line_style.get("label", label),
        )
        line_handles.append(line_handle)
        ax.plot(curves[label]["vols"], curves[label]["means"], **line_style)
        ax.scatter(points[label]["gmv"]["vol"], points[label]["gmv"]["mean"], **style("gmv", label))
        ax.scatter(points[label]["tan"]["vol"], points[label]["tan"]["mean"], **style("tan", label))

    # x-axis policy is configurable so we can keep figure geometry consistent across scopes.
    x_max = _compute_xmax(
        curves,
        points,
        x_mode=x_mode,
        frontier_mult=frontier_mult,
        tan_mult=tan_mult,
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, None)
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


def plot_scope3_overlay(scope3_result: dict):
    """
    Placeholder for Scope 3 plotting.
    """
    raise NotImplementedError("Scope 3 plotting scaffolded but not implemented yet.")


def plot_scope5_overlay(scope5_result: dict):
    """
    Placeholder for Scope 5 plotting.
    """
    raise NotImplementedError("Scope 5 plotting scaffolded but not implemented yet.")


def plot_scope6_panels(scope6_result: dict):
    """
    Placeholder for Scope 6 plotting.
    """
    raise NotImplementedError("Scope 6 plotting scaffolded but not implemented yet.")
