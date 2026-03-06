"""
Plotting helpers for frontier workflows.

Each function accepts workflow output dicts and returns matplotlib Figure
objects so notebooks can stay concise and declarative.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np

from .plot_styles import SCOPE3_PLOT_STYLE, style

# Report-friendly figure sizes (inches) for A4 with 1in margins:
# text width is ~6.27in, so we keep full-width figures near 6.2in.
REPORT_FIGSIZE = {
    "one_col_full": (6.2, 3.8),
    "one_col_full_two_panel": (6.2, 3.0),
    "two_col_single": (3.0, 2.1),
    "two_col_span": (6.2, 3.0),
}

# Runtime-tunable defaults for report plotting.
# Use `set_plot_defaults(...)` from notebook cells.
PLOT_DEFAULTS = {
    "overlay_layout": "single_column",  # {'single_column', 'full_width'}
    "panel_layout": "full_width",       # panels are intended to be full-width
    "show_titles": True,
    "figsize_scale": 1,                 # integer scale multiplier for export sizing
}


def set_plot_defaults(
    *,
    overlay_layout: str | None = None,
    panel_layout: str | None = None,
    show_titles: bool | None = None,
    figsize_scale: int | None = None,
) -> dict[str, str]:
    """
    Update plotting layout defaults and return the updated defaults.

    overlay_layout:
        - 'single_column'  -> two-column single-width figure (~3.0in)
        - 'full_width'     -> full text width figure (~6.2in)
    panel_layout:
        - 'full_width' (recommended/default)
    """
    if overlay_layout is not None:
        if overlay_layout not in {"single_column", "full_width"}:
            raise ValueError("overlay_layout must be one of {'single_column', 'full_width'}")
        PLOT_DEFAULTS["overlay_layout"] = overlay_layout
    if panel_layout is not None:
        if panel_layout != "full_width":
            raise ValueError("panel_layout must be 'full_width'")
        PLOT_DEFAULTS["panel_layout"] = panel_layout
    if show_titles is not None:
        PLOT_DEFAULTS["show_titles"] = bool(show_titles)
    if figsize_scale is not None:
        s = int(figsize_scale)
        if s < 1 or s > 8:
            raise ValueError("figsize_scale must be an integer between 1 and 8")
        PLOT_DEFAULTS["figsize_scale"] = s
    return dict(PLOT_DEFAULTS)


def _scaled_size(size: tuple[float, float]) -> tuple[float, float]:
    s = int(PLOT_DEFAULTS.get("figsize_scale", 1))
    return (size[0] * s, size[1] * s)


def _resolve_overlay_figsize(layout: str | None = None) -> tuple[float, float]:
    mode = layout if layout is not None else PLOT_DEFAULTS["overlay_layout"]
    if mode == "single_column":
        return _scaled_size(REPORT_FIGSIZE["two_col_single"])
    if mode == "full_width":
        return _scaled_size(REPORT_FIGSIZE["one_col_full"])
    raise ValueError("overlay layout must be one of {'single_column', 'full_width'}")


def _resolve_panel_figsize(layout: str | None = None) -> tuple[float, float]:
    mode = layout if layout is not None else PLOT_DEFAULTS["panel_layout"]
    if mode == "full_width":
        return _scaled_size(REPORT_FIGSIZE["one_col_full_two_panel"])
    raise ValueError("panel layout must be 'full_width'")


def _percent_no_symbol(x: float, _pos: float) -> str:
    """Format decimal returns/volatility as percent units without '%' symbol."""
    v = x * 100.0
    if abs(v) < 1e-9:
        v = 0.0
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _apply_percent_axes(ax, x_label: str, y_label: str) -> None:
    """Display x/y ticks in percent units and append units to labels."""
    ax.xaxis.set_major_formatter(FuncFormatter(_percent_no_symbol))
    ax.yaxis.set_major_formatter(FuncFormatter(_percent_no_symbol))
    ax.xaxis.set_major_locator(MultipleLocator(0.01))    # 1.00% steps
    ax.yaxis.set_major_locator(MultipleLocator(0.0025))  # 0.25% steps
    ax.set_xlabel(f"{x_label} (%)")
    ax.set_ylabel(f"{y_label} (%)")


def _portfolio_marker_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=8, label="GMV"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=8, label="Tangency"),
    ]


def _scope3_estimator_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color="black", linestyle="-", linewidth=1.5, label="Sample"),
        mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.5, label="BS-μ"),
        mlines.Line2D([], [], color="black", linestyle="-.", linewidth=1.5, label="BS-μΣ"),
    ]


def _scope3_universe_handles(n_ind: int | None, n_stk: int | None) -> list[mlines.Line2D]:
    ind_n = n_ind if n_ind is not None else "N"
    stk_n = n_stk if n_stk is not None else "N"
    return [
        mlines.Line2D(
            [], [], color=SCOPE3_PLOT_STYLE["universe"]["industry"]["color"], linestyle="-", linewidth=1.5,
            label=f"{ind_n} industries"
        ),
        mlines.Line2D(
            [], [], color=SCOPE3_PLOT_STYLE["universe"]["stock"]["color"], linestyle="-", linewidth=1.5,
            label=f"{stk_n} stocks"
        ),
    ]


def _scope2_universe_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [], [], color=SCOPE3_PLOT_STYLE["universe"]["industry"]["color"], linestyle="-", linewidth=1.5,
            label="30 industries"
        ),
    ]


def _scope5_universe_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color=style("frontier", "industries")["color"], linestyle="-", linewidth=1.5, label="Industries"),
        mlines.Line2D([], [], color=style("frontier", "ff3")["color"], linestyle="-", linewidth=1.5, label="FF3"),
        mlines.Line2D([], [], color=style("frontier", "ff5")["color"], linestyle="-", linewidth=1.5, label="FF5"),
    ]


def _scope6_universe_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color=style("frontier", "ff3")["color"], linestyle="-", linewidth=1.5, label="FF3"),
        mlines.Line2D([], [], color=style("frontier", "ff5")["color"], linestyle="-", linewidth=1.5, label="FF5"),
    ]


def _scope6_method_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color="black", linestyle="-", linewidth=1.5, label="Sample factors"),
        mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.5, label="Proxy basket"),
    ]


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
    figsize: tuple[float, float] | None = None,
    overlay_layout: str | None = None,
    show_title: bool | None = None,
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

    if figsize is None:
        figsize = _resolve_overlay_figsize(overlay_layout)
    fig, ax = plt.subplots(figsize=figsize)

    universe_color = SCOPE3_PLOT_STYLE["universe"]["industry"]["color"]
    plotted_curves = {}
    for label in ("sample", "bs_mean", "bs_mean_cov"):
        est_style = SCOPE3_PLOT_STYLE["estimator"][label]
        line_style = style("frontier", label)
        line_style["color"] = universe_color
        line_style["linestyle"] = est_style["linestyle"]
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
    _apply_percent_axes(ax, "Volatility (monthly std dev)", "Expected return (monthly, net)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
        if sample_meta:
            title = (
                f"Question 2: 30-Industry MV Frontier "
                f"({sample_meta.get('start', '')} to {sample_meta.get('end', '')})"
            )
        else:
            title = "Question 2: 30-Industry MV Frontier"
        ax.set_title(title)
    leg_colors = ax.legend(handles=_scope2_universe_handles(), loc="upper left", fontsize=8)
    ax.add_artist(leg_colors)
    leg_lines = ax.legend(
        handles=_scope3_estimator_handles(),
        loc="center right",
        bbox_to_anchor=(1.0, 0.36),
        fontsize=8,
    )
    ax.add_artist(leg_lines)
    ax.legend(handles=_portfolio_marker_handles(), loc="lower right", fontsize=8)
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
    figsize: tuple[float, float] | None = None,
    overlay_layout: str | None = None,
    show_title: bool | None = None,
):
    """
    Plot Scope 3 frontiers: 30 industries vs 30 stocks across estimators.
    """
    curves = scope3_result["plot_data"]["curves"]
    points = scope3_result["plot_data"]["points"]
    range_data = scope3_result["plot_data"].get("range", {})
    meta = scope3_result.get("inputs", {})

    if figsize is None:
        figsize = _resolve_overlay_figsize(overlay_layout)
    fig, ax = plt.subplots(figsize=figsize)

    est_order = ("sample", "bs_mean", "bs_mean_cov")
    univ_order = ("industry", "stock")
    plotted_curves_nested = {}

    for est in est_order:
        plotted_curves_nested[est] = {}
        for univ in univ_order:
            est_style = SCOPE3_PLOT_STYLE["estimator"][est]
            univ_style = SCOPE3_PLOT_STYLE["universe"][univ]
            curve = curves[est][univ]
            x, y = _frontier_arrays(curve, points[est][univ]["gmv"]["mean"], efficient_frontier_only)
            plotted_curves_nested[est][univ] = {"vols": x, "means": y}
            ax.plot(
                x,
                y,
                color=univ_style["color"],
                linestyle=est_style["linestyle"],
                linewidth=1.5,
                label="_nolegend_",
                zorder=2,
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
    _apply_percent_axes(ax, "Volatility (monthly std dev)", "Expected return (monthly, net)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
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

    n_ind = meta.get("n_assets_industry") if meta else None
    n_stk = meta.get("n_assets_stock") if meta else None
    leg_colors = ax.legend(handles=_scope3_universe_handles(n_ind, n_stk), loc="upper left", fontsize=8)
    ax.add_artist(leg_colors)
    leg_lines = ax.legend(
        handles=_scope3_estimator_handles(),
        loc="center right",
        bbox_to_anchor=(1.0, 0.36),
        fontsize=8,
    )
    ax.add_artist(leg_lines)
    ax.legend(handles=_portfolio_marker_handles(), loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_scope3_panels(
    scope3_with_coal_result: dict,
    scope3_drop_coal_result: dict,
    *,
    x_mode: str = "frontier",
    y_mode: str = "frontier",
    frontier_mult: float = 1.2,
    tan_mult: float = 2.0,
    efficient_frontier_only: bool = True,
    figsize: tuple[float, float] | None = None,
    panel_layout: str | None = None,
    show_title: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    with_title: str = "Scope 3: With Coal (30)",
    drop_title: str = "Scope 3: No Coal (29)",
):
    """
    Two-panel Scope 3 plot: with_coal_30 vs drop_coal_29.
    """
    if figsize is None:
        figsize = _resolve_panel_figsize(panel_layout)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])

    marker_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=8, label="GMV"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=8, label="Tangency"),
    ]
    est_order = ("sample", "bs_mean", "bs_mean_cov")
    univ_order = ("industry", "stock")

    def _draw_scope3_axis(ax, scope3_result: dict, panel_title: str):
        curves = scope3_result["plot_data"]["curves"]
        points = scope3_result["plot_data"]["points"]
        range_data = scope3_result["plot_data"].get("range", {})
        plotted_curves_nested = {}

        for est in est_order:
            plotted_curves_nested[est] = {}
            for univ in univ_order:
                est_style = SCOPE3_PLOT_STYLE["estimator"][est]
                univ_style = SCOPE3_PLOT_STYLE["universe"][univ]
                curve = curves[est][univ]
                x, y = _frontier_arrays(curve, points[est][univ]["gmv"]["mean"], efficient_frontier_only)
                plotted_curves_nested[est][univ] = {"vols": x, "means": y}
                ax.plot(
                    x,
                    y,
                    color=univ_style["color"],
                    linestyle=est_style["linestyle"],
                    linewidth=1.5,
                    label="_nolegend_",
                    zorder=2,
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

        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        else:
            x_max = float(max(flat_curves[k]["vols"].max() for k in flat_curves.keys()))
            ax.set_xlim(0, x_max)

        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            y_max = float(range_data.get("mu_max", _compute_ymax(
                flat_curves,
                flat_points,
                y_mode=y_mode,
                frontier_mult=frontier_mult,
                tan_mult=tan_mult,
            )))
            ax.set_ylim(0, y_max)

        _apply_percent_axes(ax, "Volatility (monthly std dev)", "Expected return (monthly, net)")
        if show_title:
            ax.set_title(panel_title)

        panel_meta = scope3_result.get("inputs", {})
        n_ind = panel_meta.get("n_assets_industry")
        n_stk = panel_meta.get("n_assets_stock")
        leg_colors = ax.legend(handles=_scope3_universe_handles(n_ind, n_stk), loc="upper left", fontsize=8)
        ax.add_artist(leg_colors)
        leg_lines = ax.legend(
            handles=_scope3_estimator_handles(),
            loc="center right",
            bbox_to_anchor=(1.0, 0.36),
            fontsize=8,
        )
        ax.add_artist(leg_lines)
        ax.legend(handles=marker_handles, loc="lower right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    _draw_scope3_axis(axes[0], scope3_with_coal_result, with_title)
    _draw_scope3_axis(axes[1], scope3_drop_coal_result, drop_title)
    fig.tight_layout()
    return fig


def plot_scope5_overlay(
    scope5_result: dict,
    efficient_frontier_only: bool = True,
    figsize: tuple[float, float] | None = None,
    overlay_layout: str | None = None,
    show_title: bool | None = None,
):
    """
    Single-panel Scope 5 overlay: industries vs FF3 vs FF5 in excess space.
    """
    curves = scope5_result["plot_data"]["curves"]
    points = scope5_result["plot_data"]["points"]
    cml = scope5_result["plot_data"]["cml"]
    range_data = scope5_result["plot_data"].get("range", {})
    meta = scope5_result.get("inputs", {})

    if figsize is None:
        figsize = _resolve_overlay_figsize(overlay_layout)
    fig, ax = plt.subplots(figsize=figsize)

    for key in ("industries", "ff3", "ff5"):
        x, y = _frontier_arrays(curves[key], points[key]["gmv"]["mean"], efficient_frontier_only)
        ax.plot(x, y, **style("frontier", key, label="_nolegend_"))
        ax.plot(cml[key]["vols"], cml[key]["means"], **style("cml", key, label="_nolegend_"))
        ax.scatter(points[key]["gmv"]["vol"], points[key]["gmv"]["mean"], **style("gmv", key))
        ax.scatter(points[key]["tan"]["vol"], points[key]["tan"]["mean"], **style("tan", key))

    x_max = float(range_data.get("x_max", max(curves[k]["vols"].max() for k in curves.keys())))
    y_max = float(range_data.get("mu_max", max(curves[k]["means"].max() for k in curves.keys())))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    _apply_percent_axes(ax, "Volatility (monthly, excess)", "Expected excess return (monthly)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
        if meta:
            ax.set_title(
                "Question 5: Industries vs FF3 vs FF5 "
                f"({meta.get('start', '')} to {meta.get('end', '')})"
            )
        else:
            ax.set_title("Question 5: Industries vs FF3 vs FF5")
    leg_colors = ax.legend(handles=_scope5_universe_handles(), loc="upper left", fontsize=8)
    ax.add_artist(leg_colors)
    ax.legend(handles=_portfolio_marker_handles(), loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_scope4_with_rf(
    scope4_result: dict,
    efficient_frontier_only: bool = True,
    figsize: tuple[float, float] | None = None,
    overlay_layout: str | None = None,
    show_title: bool | None = None,
):
    """
    Scope 4 plot: industries-only risky frontier plus CML with risk-free asset.
    """
    curve = scope4_result["plot_data"]["curve"]
    cml = scope4_result["plot_data"]["cml"]
    range_data = scope4_result["plot_data"].get("range", {})
    points = scope4_result["plot_data"]["points"]
    meta = scope4_result.get("inputs", {})

    if figsize is None:
        figsize = _resolve_overlay_figsize(overlay_layout)
    fig, ax = plt.subplots(figsize=figsize)
    frontier_style = style("frontier", "industries", label="Industries")
    frontier_style["alpha"] = 0.75
    frontier_style["linestyle"] = "-"
    x, y = _frontier_arrays(curve, points["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **frontier_style)

    # Keep same colour family as the risky frontier because both come from the
    # same underlying asset universe; differentiate primarily by linewidth/label.
    cml_style = style("cml", "industries", label="_nolegend_")
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
    _apply_percent_axes(ax, "Volatility (monthly std dev)", "Expected return (monthly, net)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
        if meta:
            ax.set_title(
                "Question 4: 30 Industries + Risk-Free "
                f"({meta.get('start', '')} to {meta.get('end', '')})"
            )
        else:
            ax.set_title("Question 4: 30 Industries + Risk-Free")
    leg1 = ax.legend(loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=_portfolio_marker_handles(), loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_scope6_panels(
    scope6_result: dict,
    limit_basis: str = "workflow",
    tan_vol_rank: int = 1,
    limit_mult: float = 1.2,
    efficient_frontier_only: bool = True,
    figsize: tuple[float, float] | None = None,
    panel_layout: str | None = None,
    show_title: bool | None = None,
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

    if figsize is None:
        figsize = _resolve_panel_figsize(panel_layout)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])

    # Left: FF3 vs Proxy-3
    ax = axes[0]
    x, y = _frontier_arrays(curves["ff3"], points["ff3"]["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **style("frontier", "ff3"))
    x, y = _frontier_arrays(curves["proxy3"], points["proxy3"]["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **style("frontier", "ff3", proxy=True))
    ax.plot(cml["ff3"]["vols"], cml["ff3"]["means"], **style("cml", "ff3", label="_nolegend_"))
    ax.plot(cml["proxy3"]["vols"], cml["proxy3"]["means"], **style("cml", "ff3", proxy=True, label="_nolegend_"))
    ax.scatter(points["ff3"]["gmv"]["vol"], points["ff3"]["gmv"]["mean"], **style("gmv", "ff3"))
    ax.scatter(points["ff3"]["tan"]["vol"], points["ff3"]["tan"]["mean"], **style("tan", "ff3"))
    ax.scatter(points["proxy3"]["gmv"]["vol"], points["proxy3"]["gmv"]["mean"], **style("gmv", "ff3", proxy=True))
    ax.scatter(points["proxy3"]["tan"]["vol"], points["proxy3"]["tan"]["mean"], **style("tan", "ff3", proxy=True))
    _apply_percent_axes(ax, "Volatility (monthly, excess)", "Expected excess return (monthly)")
    if show_title:
        ax.set_title("Scope 6: FF3 vs Proxy-3")
    ax.grid(True, linestyle="--", alpha=0.4)
    leg_lines = ax.legend(handles=_scope6_method_handles(), loc="upper left", fontsize=8)
    ax.add_artist(leg_lines)
    ax.legend(handles=_portfolio_marker_handles(), loc="lower right", fontsize=8)

    # Right: FF5 vs Proxy-5
    ax = axes[1]
    x, y = _frontier_arrays(curves["ff5"], points["ff5"]["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **style("frontier", "ff5"))
    x, y = _frontier_arrays(curves["proxy5"], points["proxy5"]["gmv"]["mean"], efficient_frontier_only)
    ax.plot(x, y, **style("frontier", "ff5", proxy=True))
    ax.plot(cml["ff5"]["vols"], cml["ff5"]["means"], **style("cml", "ff5", label="_nolegend_"))
    ax.plot(cml["proxy5"]["vols"], cml["proxy5"]["means"], **style("cml", "ff5", proxy=True, label="_nolegend_"))
    ax.scatter(points["ff5"]["gmv"]["vol"], points["ff5"]["gmv"]["mean"], **style("gmv", "ff5"))
    ax.scatter(points["ff5"]["tan"]["vol"], points["ff5"]["tan"]["mean"], **style("tan", "ff5"))
    ax.scatter(points["proxy5"]["gmv"]["vol"], points["proxy5"]["gmv"]["mean"], **style("gmv", "ff5", proxy=True))
    ax.scatter(points["proxy5"]["tan"]["vol"], points["proxy5"]["tan"]["mean"], **style("tan", "ff5", proxy=True))
    _apply_percent_axes(ax, "Volatility (monthly, excess)", "Expected excess return (monthly)")
    if show_title:
        ax.set_title("Scope 6: FF5 vs Proxy-5")
    ax.grid(True, linestyle="--", alpha=0.4)
    leg_lines = ax.legend(handles=_scope6_method_handles(), loc="upper left", fontsize=8)
    ax.add_artist(leg_lines)
    ax.legend(handles=_portfolio_marker_handles(), loc="lower right", fontsize=8)

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

    if show_title and meta:
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
    figsize: tuple[float, float] | None = None,
    overlay_layout: str | None = None,
    show_title: bool | None = None,
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

    if figsize is None:
        figsize = _resolve_overlay_figsize(overlay_layout)
    fig, ax = plt.subplots(figsize=figsize)

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
    _apply_percent_axes(ax, "Volatility (monthly, excess)", "Expected excess return (monthly)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
        if meta:
            title = f"Question 6: FF Factors vs Practical Proxies ({meta.get('start', '')} to {meta.get('end', '')})"
            if limit_basis == "tangency_vol_rank":
                title += f" [limits: tan vol rank {max(1, int(tan_vol_rank))}]"
            ax.set_title(title)
        else:
            ax.set_title("Question 6: FF Factors vs Practical Proxies")
    ax.grid(True, linestyle="--", alpha=0.4)
    leg_colors = ax.legend(handles=_scope6_universe_handles(), loc="upper left", fontsize=8)
    ax.add_artist(leg_colors)
    leg_lines = ax.legend(
        handles=_scope6_method_handles(),
        loc="center right",
        bbox_to_anchor=(1.0, 0.36),
        fontsize=8,
    )
    ax.add_artist(leg_lines)
    ax.legend(handles=_portfolio_marker_handles(), loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
