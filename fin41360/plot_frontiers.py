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

from .plot_styles import (
    SCOPE3_PLOT_STYLE,
    scope6_legend_handles,
    scope6_panel_legend_handles,
    style,
)

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
    ax.yaxis.set_major_locator(MultipleLocator(0.0050))  # 0.50% steps (labeled)
    ax.yaxis.set_minor_locator(MultipleLocator(0.0025))  # 0.25% steps (grid only)
    ax.set_xlabel(f"{x_label} (%)")
    ax.set_ylabel(f"{y_label} (%)")


def _apply_report_grid(ax) -> None:
    """Major + minor grid policy for report figures."""
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", linestyle="--", alpha=0.2)


def _portfolio_marker_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=8, label="GMV"),
        mlines.Line2D([], [], color="gray", marker="*", linestyle="None", markersize=11, label="Tangency"),
    ]


def _scope3_estimator_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color=style("frontier", "sample")["color"], linestyle="-", linewidth=1.5, label="Sample"),
        mlines.Line2D([], [], color=style("frontier", "bs_mean")["color"], linestyle="-", linewidth=1.5, label="BS-μ"),
        mlines.Line2D([], [], color=style("frontier", "bs_mean_cov")["color"], linestyle="-", linewidth=1.5, label="BS-μΣ"),
    ]


def _scope3_universe_handles(n_ind: int | None, n_stk: int | None) -> list[mlines.Line2D]:
    ind_n = n_ind if n_ind is not None else "N"
    stk_n = n_stk if n_stk is not None else "N"
    return [
        mlines.Line2D(
            [], [], color="black", linestyle="-", linewidth=1.5,
            label=f"{ind_n} industries"
        ),
        mlines.Line2D(
            [], [], color="black", linestyle="--", linewidth=1.5,
            label=f"{stk_n} stocks"
        ),
    ]


def _scope5_universe_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color=style("frontier", "industries")["color"], linestyle="-", linewidth=1.5, label="Industries"),
        mlines.Line2D([], [], color=style("frontier", "ff3")["color"], linestyle="-", linewidth=1.5, label="FF3"),
        mlines.Line2D([], [], color=style("frontier", "ff5")["color"], linestyle="-", linewidth=1.5, label="FF5"),
    ]


def _set_axis_limits(
    ax,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
    anchor_origin: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    pad_frac: float = 0.08,
) -> None:
    """Apply local readability-oriented limits unless explicit limits are provided."""
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        x_arr = np.asarray(x_ref if x_ref is not None else x_values, dtype=float)
        x_arr = x_arr[np.isfinite(x_arr)]
        if x_arr.size == 0:
            x_arr = np.asarray(x_values, dtype=float)
            x_arr = x_arr[np.isfinite(x_arr)]
        x_lo = float(np.nanmin(x_arr)) if x_arr.size else 0.0
        x_hi = float(np.nanmax(x_arr)) if x_arr.size else 1.0
        x_span = max(x_hi - x_lo, 1e-9)
        x_min = 0.0 if anchor_origin else max(0.0, x_lo - pad_frac * x_span)
        x_max = x_hi + pad_frac * x_span
        if x_max <= x_min:
            x_max = x_min + 1e-6
        ax.set_xlim(x_min, x_max)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        y_arr = np.asarray(y_ref if y_ref is not None else y_values, dtype=float)
        y_arr = y_arr[np.isfinite(y_arr)]
        if y_arr.size == 0:
            y_arr = np.asarray(y_values, dtype=float)
            y_arr = y_arr[np.isfinite(y_arr)]
        y_lo = float(np.nanmin(y_arr)) if y_arr.size else 0.0
        y_hi = float(np.nanmax(y_arr)) if y_arr.size else 1.0
        y_span = max(y_hi - y_lo, 1e-9)
        y_min = 0.0 if anchor_origin else y_lo - pad_frac * y_span
        y_max = y_hi + pad_frac * y_span
        if y_max <= y_min:
            y_max = y_min + 1e-6
        ax.set_ylim(y_min, y_max)


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


def _ranked_tangency_limit(
    tangency_points: list[dict[str, float]],
    *,
    limit_mult: float = 1.2,
    rank: int = 1,
) -> float | None:
    """Return `limit_mult *` the ranked tangency volatility (1=largest)."""
    vols = sorted(
        (
            float(p["vol"])
            for p in tangency_points
            if p is not None and np.isfinite(float(p.get("vol", np.nan)))
        ),
        reverse=True,
    )
    if len(vols) == 0:
        return None
    idx = min(max(1, int(rank)) - 1, len(vols) - 1)
    return float(limit_mult) * vols[idx]


def _ranked_tangency_point(
    tangency_points: list[dict[str, float]],
    *,
    rank: int = 1,
) -> dict[str, float] | None:
    """Return the ranked tangency point by volatility (1=largest)."""
    ordered = sorted(
        (
            {"vol": float(p["vol"]), "mean": float(p["mean"])}
            for p in tangency_points
            if p is not None
            and np.isfinite(float(p.get("vol", np.nan)))
            and np.isfinite(float(p.get("mean", np.nan)))
        ),
        key=lambda p: p["vol"],
        reverse=True,
    )
    if len(ordered) == 0:
        return None
    idx = min(max(1, int(rank)) - 1, len(ordered) - 1)
    return ordered[idx]


def _gmv_lower_limits(gmv_points: list[dict[str, float]]) -> tuple[float | None, float | None]:
    """Return lower bounds from the minimum GMV vol and minimum GMV mean."""
    vols = [float(p["vol"]) for p in gmv_points if p is not None and np.isfinite(float(p.get("vol", np.nan)))]
    means = [float(p["mean"]) for p in gmv_points if p is not None and np.isfinite(float(p.get("mean", np.nan)))]
    x_min = min(vols) if len(vols) > 0 else None
    y_min = min(means) if len(means) > 0 else None
    return x_min, y_min


def _apply_ymin(ax, y_min: float | None) -> None:
    """Keep the current upper limit but enforce a lower y bound when provided."""
    if y_min is None or not np.isfinite(float(y_min)):
        return
    _, current_ymax = ax.get_ylim()
    if current_ymax <= float(y_min):
        current_ymax = float(y_min) + 1e-6
    ax.set_ylim(float(y_min), current_ymax)


def _apply_visible_ymax(ax, *, pad_frac: float = 0.08) -> None:
    """Set y max from data visible inside the current x window."""
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi_current = ax.get_ylim()
    visible_y: list[float] = []

    for line in ax.get_lines():
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys) & (xs >= x_lo) & (xs <= x_hi)
        if np.any(mask):
            visible_y.extend(ys[mask].tolist())

    for coll in ax.collections:
        offsets = getattr(coll, "get_offsets", lambda: np.empty((0, 2)))()
        offsets = np.asarray(offsets, dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] < 2:
            continue
        xs = offsets[:, 0]
        ys = offsets[:, 1]
        mask = np.isfinite(xs) & np.isfinite(ys) & (xs >= x_lo) & (xs <= x_hi)
        if np.any(mask):
            visible_y.extend(ys[mask].tolist())

    if len(visible_y) == 0:
        return

    y_hi = float(max(visible_y))
    span = max(y_hi - y_lo, 1e-9)
    y_max = y_hi + pad_frac * span
    if y_max <= y_lo:
        y_max = y_lo + 1e-6
    if y_max < y_hi_current:
        ax.set_ylim(y_lo, y_max)


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
    anchor_origin: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
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

    x_vals: list[float] = []
    y_vals: list[float] = []
    x_pts: list[float] = []
    y_pts: list[float] = []
    for label in ("sample", "bs_mean", "bs_mean_cov"):
        line_style = style("frontier", label)
        line_style["linestyle"] = "-"
        x, y = _frontier_arrays(curves[label], points[label]["gmv"]["mean"], efficient_frontier_only)
        x_vals.extend(np.asarray(x, dtype=float).tolist())
        y_vals.extend(np.asarray(y, dtype=float).tolist())
        ax.plot(x, y, **line_style)
        gmv_style = style("gmv", label)
        tan_style = style("tan", label)
        ax.scatter(points[label]["gmv"]["vol"], points[label]["gmv"]["mean"], **gmv_style)
        ax.scatter(points[label]["tan"]["vol"], points[label]["tan"]["mean"], **tan_style)
        x_vals.extend([float(points[label]["gmv"]["vol"]), float(points[label]["tan"]["vol"])])
        y_vals.extend([float(points[label]["gmv"]["mean"]), float(points[label]["tan"]["mean"])])
        x_pts.extend([float(points[label]["gmv"]["vol"]), float(points[label]["tan"]["vol"])])
        y_pts.extend([float(points[label]["gmv"]["mean"]), float(points[label]["tan"]["mean"])])

    y_min = 0.0
    if xlim is None or ylim is None:
        gmv_points = [points[label]["gmv"] for label in ("sample", "bs_mean", "bs_mean_cov")]
        tan_points = [points[label]["tan"] for label in ("sample", "bs_mean", "bs_mean_cov")]
        x_min, _ = _gmv_lower_limits(gmv_points)
        tan_ref = _ranked_tangency_point(tan_points, rank=1)
        if xlim is None and x_min is not None and tan_ref is not None:
            xlim = (x_min, 1.2 * float(tan_ref["vol"]))
    _set_axis_limits(
        ax,
        np.asarray(x_vals, dtype=float),
        np.asarray(y_vals, dtype=float),
        anchor_origin=anchor_origin,
        xlim=xlim,
        ylim=ylim,
    )
    _apply_ymin(ax, y_min)
    _apply_visible_ymax(ax)
    _apply_percent_axes(ax, "Volatility (monthly std dev)", "Expected return (monthly, net)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
        if sample_meta:
            title = (
                f"Task 2: 30-Industry Frontier "
                f"({sample_meta.get('start', '')} to {sample_meta.get('end', '')})"
            )
        else:
            title = "Task 2: 30-Industry Frontier"
        ax.set_title(title)
    ax.legend(
        handles=_scope3_estimator_handles() + _portfolio_marker_handles(),
        loc="upper left",
        fontsize=8,
    )
    _apply_report_grid(ax)
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
    anchor_origin: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
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
    x_vals: list[float] = []
    y_vals: list[float] = []
    x_pts: list[float] = []
    y_pts: list[float] = []

    for est in est_order:
        for univ in univ_order:
            curve = curves[est][univ]
            x, y = _frontier_arrays(curve, points[est][univ]["gmv"]["mean"], efficient_frontier_only)
            x_vals.extend(np.asarray(x, dtype=float).tolist())
            y_vals.extend(np.asarray(y, dtype=float).tolist())
            ax.plot(
                x,
                y,
                color=style("frontier", est)["color"],
                linestyle="-" if univ == "industry" else "--",
                linewidth=1.5,
                label="_nolegend_",
                zorder=2,
            )

    for est in est_order:
        for univ in univ_order:
            marker_color = style("frontier", est)["color"]
            gmv = points[est][univ]["gmv"]
            tan = points[est][univ]["tan"]
            x_vals.extend([float(gmv["vol"]), float(tan["vol"])])
            y_vals.extend([float(gmv["mean"]), float(tan["mean"])])
            x_pts.extend([float(gmv["vol"]), float(tan["vol"])])
            y_pts.extend([float(gmv["mean"]), float(tan["mean"])])
            ax.scatter(
                [gmv["vol"]],
                [gmv["mean"]],
                color=marker_color,
                marker=SCOPE3_PLOT_STYLE["portfolio_marker"]["GMV"],
                s=60,
                zorder=5,
                label="_nolegend_",
            )
            ax.scatter(
                [tan["vol"]],
                [tan["mean"]],
                color=marker_color,
                marker=SCOPE3_PLOT_STYLE["portfolio_marker"]["TAN"],
                s=60,
                zorder=6,
                label="_nolegend_",
            )

    y_min = 0.0
    if xlim is None or ylim is None:
        gmv_points = [points[est][univ]["gmv"] for est in est_order for univ in univ_order]
        tan_points = [points[est][univ]["tan"] for est in est_order for univ in univ_order]
        x_min, _ = _gmv_lower_limits(gmv_points)
        tan_ref = _ranked_tangency_point(tan_points, rank=2)
        if xlim is None and x_min is not None and tan_ref is not None:
            xlim = (x_min, 1.2 * float(tan_ref["vol"]))
    _set_axis_limits(
        ax,
        np.asarray(x_vals, dtype=float),
        np.asarray(y_vals, dtype=float),
        anchor_origin=anchor_origin,
        xlim=xlim,
        ylim=ylim,
    )
    _apply_ymin(ax, y_min)
    _apply_visible_ymax(ax)
    _apply_percent_axes(ax, "Volatility (monthly std dev)", "Expected return (monthly, net)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
        if meta:
            n_ind = meta.get("n_assets_industry")
            n_stk = meta.get("n_assets_stock")
            if n_ind is not None and n_stk is not None:
                ax.set_title(
                    f"Task 3: {n_ind} Industries vs {n_stk} Stocks "
                    f"({meta.get('common_start', '')} to {meta.get('common_end', '')})"
                )
            else:
                ax.set_title(
                    "Task 3: Industries vs Stocks "
                    f"({meta.get('common_start', '')} to {meta.get('common_end', '')})"
                )
        else:
            ax.set_title("Task 3: Industries vs Stocks")

    n_ind = meta.get("n_assets_industry") if meta else None
    n_stk = meta.get("n_assets_stock") if meta else None
    ax.legend(
        handles=_scope3_universe_handles(n_ind, n_stk) + _scope3_estimator_handles() + _portfolio_marker_handles(),
        loc="upper left",
        fontsize=8,
    )
    _apply_report_grid(ax)
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
    anchor_origin: bool = False,
    with_title: str = "Task 3 Sensitivity: With Coal (30)",
    drop_title: str = "Task 3 Main: Drop Coal (29)",
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
        mlines.Line2D([], [], color="gray", marker="*", linestyle="None", markersize=11, label="Tangency"),
    ]
    est_order = ("sample", "bs_mean", "bs_mean_cov")
    univ_order = ("industry", "stock")

    def _draw_scope3_axis(ax, scope3_result: dict, panel_title: str):
        curves = scope3_result["plot_data"]["curves"]
        points = scope3_result["plot_data"]["points"]
        range_data = scope3_result["plot_data"].get("range", {})
        x_vals: list[float] = []
        y_vals: list[float] = []
        x_pts: list[float] = []
        y_pts: list[float] = []

        for est in est_order:
            for univ in univ_order:
                curve = curves[est][univ]
                x, y = _frontier_arrays(curve, points[est][univ]["gmv"]["mean"], efficient_frontier_only)
                x_vals.extend(np.asarray(x, dtype=float).tolist())
                y_vals.extend(np.asarray(y, dtype=float).tolist())
                ax.plot(
                    x,
                    y,
                    color=style("frontier", est)["color"],
                    linestyle="-" if univ == "industry" else "--",
                    linewidth=1.5,
                    label="_nolegend_",
                    zorder=2,
                )

        for est in est_order:
            for univ in univ_order:
                marker_color = style("frontier", est)["color"]
                gmv = points[est][univ]["gmv"]
                tan = points[est][univ]["tan"]
                x_vals.extend([float(gmv["vol"]), float(tan["vol"])])
                y_vals.extend([float(gmv["mean"]), float(tan["mean"])])
                x_pts.extend([float(gmv["vol"]), float(tan["vol"])])
                y_pts.extend([float(gmv["mean"]), float(tan["mean"])])
                ax.scatter(
                    [gmv["vol"]],
                    [gmv["mean"]],
                    color=marker_color,
                    marker=SCOPE3_PLOT_STYLE["portfolio_marker"]["GMV"],
                    s=60,
                    zorder=5,
                    label="_nolegend_",
                )
                ax.scatter(
                    [tan["vol"]],
                    [tan["mean"]],
                    color=marker_color,
                    marker=SCOPE3_PLOT_STYLE["portfolio_marker"]["TAN"],
                    s=60,
                    zorder=6,
                    label="_nolegend_",
                )

        local_xlim = xlim
        local_ylim = ylim
        local_ymin = 0.0
        if local_xlim is None or local_ylim is None:
            gmv_points = [points[est][univ]["gmv"] for est in est_order for univ in univ_order]
            tan_points = [points[est][univ]["tan"] for est in est_order for univ in univ_order]
            x_min, _ = _gmv_lower_limits(gmv_points)
            tan_ref = _ranked_tangency_point(tan_points, rank=2)
            if local_xlim is None and x_min is not None and tan_ref is not None:
                local_xlim = (x_min, 1.2 * float(tan_ref["vol"]))

        _set_axis_limits(
            ax,
            np.asarray(x_vals, dtype=float),
            np.asarray(y_vals, dtype=float),
            anchor_origin=anchor_origin,
            xlim=local_xlim,
            ylim=local_ylim,
        )
        _apply_ymin(ax, local_ymin)
        _apply_visible_ymax(ax)

        _apply_percent_axes(ax, "Volatility (monthly std dev)", "Expected return (monthly, net)")
        if show_title:
            ax.set_title(panel_title)

        panel_meta = scope3_result.get("inputs", {})
        n_ind = panel_meta.get("n_assets_industry")
        n_stk = panel_meta.get("n_assets_stock")
        ax.legend(
            handles=_scope3_universe_handles(n_ind, n_stk) + _scope3_estimator_handles() + marker_handles,
            loc="upper left",
            fontsize=8,
        )
        _apply_report_grid(ax)

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
    anchor_origin: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
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

    x_vals: list[float] = []
    y_vals: list[float] = []
    x_pts: list[float] = []
    y_pts: list[float] = []
    for key in ("industries", "ff3", "ff5"):
        alpha_level = 0.6 if key == "industries" else 1.0
        x, y = _frontier_arrays(curves[key], points[key]["gmv"]["mean"], efficient_frontier_only)
        f_style = style("frontier", key, label="_nolegend_")
        f_style["alpha"] = alpha_level
        c_style = style("cml", key, label="_nolegend_")
        c_style["alpha"] = alpha_level
        g_style = style("gmv", key)
        g_style["alpha"] = alpha_level
        t_style = style("tan", key)
        t_style["alpha"] = alpha_level
        ax.plot(x, y, **f_style)
        ax.plot(cml[key]["vols"], cml[key]["means"], **c_style)
        ax.scatter(points[key]["gmv"]["vol"], points[key]["gmv"]["mean"], **g_style)
        ax.scatter(points[key]["tan"]["vol"], points[key]["tan"]["mean"], **t_style)
        x_vals.extend(np.asarray(x, dtype=float).tolist())
        y_vals.extend(np.asarray(y, dtype=float).tolist())
        x_vals.extend(np.asarray(cml[key]["vols"], dtype=float).tolist())
        y_vals.extend(np.asarray(cml[key]["means"], dtype=float).tolist())
        x_vals.extend([float(points[key]["gmv"]["vol"]), float(points[key]["tan"]["vol"])])
        y_vals.extend([float(points[key]["gmv"]["mean"]), float(points[key]["tan"]["mean"])])
        x_pts.extend([float(points[key]["gmv"]["vol"]), float(points[key]["tan"]["vol"])])
        y_pts.extend([float(points[key]["gmv"]["mean"]), float(points[key]["tan"]["mean"])])

    y_min = 0.0 if anchor_origin else None
    if xlim is None or ylim is None:
        tan_points = [points[key]["tan"] for key in ("industries", "ff3", "ff5")]
        tan_ref = _ranked_tangency_point(tan_points, rank=1)
        x_max = 1.2 * float(tan_ref["vol"]) if tan_ref is not None else None
        if xlim is None and x_max is not None:
            xlim = (0.0, x_max)
    _set_axis_limits(
        ax,
        np.asarray(x_vals, dtype=float),
        np.asarray(y_vals, dtype=float),
        anchor_origin=anchor_origin,
        xlim=xlim,
        ylim=ylim,
    )
    _apply_ymin(ax, y_min)
    _apply_visible_ymax(ax)
    _apply_percent_axes(ax, "Volatility (monthly, excess)", "Expected excess return (monthly)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
        if meta:
            ax.set_title(
                "Task 5: Industries vs FF3 vs FF5 "
                f"({meta.get('start', '')} to {meta.get('end', '')})"
            )
        else:
            ax.set_title("Task 5: Industries vs FF3 vs FF5")
    ax.legend(
        handles=_scope5_universe_handles() + _portfolio_marker_handles(),
        loc="upper left",
        fontsize=8,
    )
    _apply_report_grid(ax)
    fig.tight_layout()
    return fig


def plot_scope4_with_rf(
    scope4_result: dict,
    efficient_frontier_only: bool = True,
    figsize: tuple[float, float] | None = None,
    overlay_layout: str | None = None,
    show_title: bool | None = None,
    anchor_origin: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
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
    cml_style["alpha"] = 0.85
    cml_style["linewidth"] = 1.5
    ax.plot(cml["vols"], cml["means"], **cml_style)
    ax.scatter(points["gmv"]["vol"], points["gmv"]["mean"], **style("gmv", "industries"))
    ax.scatter(points["tan"]["vol"], points["tan"]["mean"], **style("tan", "industries"))

    rf_mean = float(meta.get("rf_mean", 0.0))
    ax.axhline(rf_mean, color="gray", linestyle=":", alpha=0.85, label=f"Risk-free ({rf_mean:.2%})")
    ax.scatter([0.0], [rf_mean], color="gray", marker="s", s=45, zorder=6, label="_nolegend_")

    x_vals = np.concatenate(
        [
            np.asarray(x, dtype=float),
            np.asarray(cml["vols"], dtype=float),
            np.asarray([points["gmv"]["vol"], points["tan"]["vol"]], dtype=float),
        ]
    )
    y_vals = np.concatenate(
        [
            np.asarray(y, dtype=float),
            np.asarray(cml["means"], dtype=float),
            np.asarray([points["gmv"]["mean"], points["tan"]["mean"], rf_mean], dtype=float),
        ]
    )
    x_pts = np.asarray([points["gmv"]["vol"], points["tan"]["vol"]], dtype=float)
    y_pts = np.asarray([points["gmv"]["mean"], points["tan"]["mean"], rf_mean], dtype=float)
    y_min = 0.0
    if xlim is None or ylim is None:
        tan_ref = _ranked_tangency_point([points["tan"]], rank=1)
        x_max = 1.2 * float(tan_ref["vol"]) if tan_ref is not None else None
        if xlim is None and x_max is not None:
            xlim = (0.0, x_max)
    if anchor_origin:
        x_cap = float(range_data.get("x_max", max(float(curve["vols"].max()), float(cml["vols"].max()))))
        y_cap = float(range_data.get("mu_max", max(float(curve["means"].max()), float(cml["means"].max()))))
        xlim = xlim if xlim is not None else (0.0, x_cap)
        ylim = ylim if ylim is not None else (0.0, y_cap)
    _set_axis_limits(
        ax,
        x_vals,
        y_vals,
        anchor_origin=anchor_origin,
        xlim=xlim,
        ylim=ylim,
    )
    _apply_ymin(ax, y_min)
    _apply_visible_ymax(ax)
    _apply_percent_axes(ax, "Volatility (monthly std dev)", "Expected return (monthly, net)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
        if meta:
            ax.set_title(
                "Task 4: Industries with Risk-Free Asset "
                f"({meta.get('start', '')} to {meta.get('end', '')})"
            )
        else:
            ax.set_title("Task 4: Industries with Risk-Free Asset")
    line_handles, line_labels = ax.get_legend_handles_labels()
    rf_handles = []
    rf_labels = []
    for handle, label in zip(line_handles, line_labels):
        if label != "_nolegend_":
            rf_handles.append(handle)
            rf_labels.append(label)
    ax.legend(handles=rf_handles + _portfolio_marker_handles(), loc="upper left", fontsize=8)
    _apply_report_grid(ax)
    fig.tight_layout()
    return fig


def plot_scope6_panels(
    scope6_result: dict,
    limit_basis: str = "tangency_vol_rank",
    tan_vol_rank: int = 2,
    limit_mult: float = 1.2,
    efficient_frontier_only: bool = True,
    figsize: tuple[float, float] | None = None,
    panel_layout: str | None = None,
    show_title: bool | None = None,
    anchor_origin: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """
    Two-panel Scope 6 plot: FF3 vs Proxy-3 and FF5 vs Proxy-5.

    limit_basis:
        - "ff5" (default): use FF5 + Proxy5 curves/CMLs for shared panel limits
        - "workflow": use scope6_result plot_data['range'] limits
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
    ff3_front = style("frontier", "ff3", label="_nolegend_")
    ff3_front["alpha"] = 0.6
    ax.plot(x, y, **ff3_front)
    x, y = _frontier_arrays(curves["proxy3"], points["proxy3"]["gmv"]["mean"], efficient_frontier_only)
    proxy3_front = style("frontier", "ff3", proxy=True, label="_nolegend_")
    proxy3_front["alpha"] = 1.0
    ax.plot(x, y, **proxy3_front)
    ff3_cml = style("cml", "ff3", label="_nolegend_")
    ff3_cml["alpha"] = 0.6
    proxy3_cml = style("cml", "ff3", proxy=True, label="_nolegend_")
    proxy3_cml["alpha"] = 1.0
    ax.plot(cml["ff3"]["vols"], cml["ff3"]["means"], **ff3_cml)
    ax.plot(cml["proxy3"]["vols"], cml["proxy3"]["means"], **proxy3_cml)
    ff3_g = style("gmv", "ff3")
    ff3_g["alpha"] = 0.6
    ff3_t = style("tan", "ff3")
    ff3_t["alpha"] = 0.6
    p3_g = style("gmv", "ff3", proxy=True)
    p3_g["alpha"] = 1.0
    p3_t = style("tan", "ff3", proxy=True)
    p3_t["alpha"] = 1.0
    ax.scatter(points["ff3"]["gmv"]["vol"], points["ff3"]["gmv"]["mean"], **ff3_g)
    ax.scatter(points["ff3"]["tan"]["vol"], points["ff3"]["tan"]["mean"], **ff3_t)
    ax.scatter(points["proxy3"]["gmv"]["vol"], points["proxy3"]["gmv"]["mean"], **p3_g)
    ax.scatter(points["proxy3"]["tan"]["vol"], points["proxy3"]["tan"]["mean"], **p3_t)
    _apply_percent_axes(ax, "Volatility (monthly, excess)", "Expected excess return (monthly)")
    if show_title:
        ax.set_title("Task 6: FF3 vs Proxy3")
    _apply_report_grid(ax)
    panel1_lines, panel1_markers = scope6_panel_legend_handles("ff3")
    ax.legend(handles=panel1_lines + panel1_markers, loc="upper left", fontsize=8)

    # Right: FF5 vs Proxy-5
    ax = axes[1]
    x, y = _frontier_arrays(curves["ff5"], points["ff5"]["gmv"]["mean"], efficient_frontier_only)
    ff5_front = style("frontier", "ff5", label="_nolegend_")
    ff5_front["alpha"] = 0.6
    ax.plot(x, y, **ff5_front)
    x, y = _frontier_arrays(curves["proxy5"], points["proxy5"]["gmv"]["mean"], efficient_frontier_only)
    proxy5_front = style("frontier", "ff5", proxy=True, label="_nolegend_")
    proxy5_front["alpha"] = 1.0
    ax.plot(x, y, **proxy5_front)
    ff5_cml = style("cml", "ff5", label="_nolegend_")
    ff5_cml["alpha"] = 0.6
    proxy5_cml = style("cml", "ff5", proxy=True, label="_nolegend_")
    proxy5_cml["alpha"] = 1.0
    ax.plot(cml["ff5"]["vols"], cml["ff5"]["means"], **ff5_cml)
    ax.plot(cml["proxy5"]["vols"], cml["proxy5"]["means"], **proxy5_cml)
    ff5_g = style("gmv", "ff5")
    ff5_g["alpha"] = 0.6
    ff5_t = style("tan", "ff5")
    ff5_t["alpha"] = 0.6
    p5_g = style("gmv", "ff5", proxy=True)
    p5_g["alpha"] = 1.0
    p5_t = style("tan", "ff5", proxy=True)
    p5_t["alpha"] = 1.0
    ax.scatter(points["ff5"]["gmv"]["vol"], points["ff5"]["gmv"]["mean"], **ff5_g)
    ax.scatter(points["ff5"]["tan"]["vol"], points["ff5"]["tan"]["mean"], **ff5_t)
    ax.scatter(points["proxy5"]["gmv"]["vol"], points["proxy5"]["gmv"]["mean"], **p5_g)
    ax.scatter(points["proxy5"]["tan"]["vol"], points["proxy5"]["tan"]["mean"], **p5_t)
    _apply_percent_axes(ax, "Volatility (monthly, excess)", "Expected excess return (monthly)")
    if show_title:
        ax.set_title("Task 6: FF5 vs Proxy5")
    _apply_report_grid(ax)
    panel2_lines, panel2_markers = scope6_panel_legend_handles("ff5")
    ax.legend(handles=panel2_lines + panel2_markers, loc="upper left", fontsize=8)

    if limit_basis == "ff5":
        # Practical FF5 framing for report readability:
        # scale exactly from FF5 tangency point (not proxy, not frontier tails).
        ref_point = points["ff5"]["tan"]
        x_max = float(limit_mult) * float(ref_point["vol"])
    elif limit_basis == "workflow":
        x_max = float(range_data.get("x_max", 0.0))
    elif limit_basis == "tangency_vol_rank":
        ref_point = _ranked_tangency_point(
            [points["ff3"]["tan"], points["proxy3"]["tan"], points["ff5"]["tan"], points["proxy5"]["tan"]],
            rank=tan_vol_rank,
        )
        x_max = float(limit_mult) * float(ref_point["vol"]) if ref_point is not None else 0.0
    else:
        raise ValueError("limit_basis must be one of {'ff5', 'workflow', 'tangency_vol_rank'}")

    # Respect limit_basis-driven limits unless user explicitly overrides with xlim/ylim.
    gmv_points = [points["ff3"]["gmv"], points["proxy3"]["gmv"], points["ff5"]["gmv"], points["proxy5"]["gmv"]]
    x_min = 0.0
    y_min = 0.0
    default_xlim = xlim if xlim is not None else ((x_min, x_max) if x_max > x_min else None)
    default_ylim = ylim
    for i, ax in enumerate(axes):
        x_line_vals = []
        y_line_vals = []
        for line in ax.get_lines():
            x_line_vals.extend(np.asarray(line.get_xdata(), dtype=float).tolist())
            y_line_vals.extend(np.asarray(line.get_ydata(), dtype=float).tolist())
        if i == 0:
            keys = ("ff3", "proxy3")
        else:
            keys = ("ff5", "proxy5")
        x_point_vals = []
        y_point_vals = []
        for key in keys:
            x_point_vals.extend([float(points[key]["gmv"]["vol"]), float(points[key]["tan"]["vol"])])
            y_point_vals.extend([float(points[key]["gmv"]["mean"]), float(points[key]["tan"]["mean"])])
        _set_axis_limits(
            ax,
            np.asarray(x_line_vals, dtype=float),
            np.asarray(y_line_vals, dtype=float),
            anchor_origin=anchor_origin,
            xlim=default_xlim,
            ylim=default_ylim,
        )
        _apply_ymin(ax, y_min)
        _apply_visible_ymax(ax)

    if show_title and meta:
        title = f"Task 6: FF Factors vs Practical Proxies ({meta.get('start', '')} to {meta.get('end', '')})"
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_scope6_overlay(
    scope6_result: dict,
    limit_basis: str = "tangency_vol_rank",
    tan_vol_rank: int = 2,
    limit_mult: float = 1.2,
    efficient_frontier_only: bool = True,
    x_limit_basis: str | None = None,
    figsize: tuple[float, float] | None = None,
    overlay_layout: str | None = None,
    show_title: bool | None = None,
    anchor_origin: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
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
        alpha_level = 1.0 if is_proxy else 0.6
        x, y = _frontier_arrays(curves[key], points[key]["gmv"]["mean"], efficient_frontier_only)
        f_style = style("frontier", base, proxy=is_proxy, label="_nolegend_")
        f_style["alpha"] = alpha_level
        c_style = style("cml", base, proxy=is_proxy, label="_nolegend_")
        c_style["alpha"] = alpha_level
        g_style = style("gmv", base, proxy=is_proxy, label="_nolegend_")
        g_style["alpha"] = alpha_level
        t_style = style("tan", base, proxy=is_proxy, label="_nolegend_")
        t_style["alpha"] = alpha_level
        ax.plot(x, y, **f_style)
        ax.plot(cml[key]["vols"], cml[key]["means"], **c_style)
        ax.scatter(points[key]["gmv"]["vol"], points[key]["gmv"]["mean"], **g_style)
        ax.scatter(points[key]["tan"]["vol"], points[key]["tan"]["mean"], **t_style)

    if x_limit_basis is not None:
        limit_basis = x_limit_basis

    if limit_basis == "ff5":
        # Practical FF5 framing for report readability:
        # scale exactly from FF5 tangency point (not proxy, not frontier tails).
        ref_point = points["ff5"]["tan"]
        x_max = float(limit_mult) * float(ref_point["vol"])
    elif limit_basis == "tangency_vol_rank":
        ref_point = _ranked_tangency_point(
            [points["ff3"]["tan"], points["proxy3"]["tan"], points["ff5"]["tan"], points["proxy5"]["tan"]],
            rank=tan_vol_rank,
        )
        x_max = float(limit_mult) * float(ref_point["vol"]) if ref_point is not None else 0.0
    else:
        raise ValueError("limit_basis must be one of {'ff5', 'tangency_vol_rank'}")
    # Respect limit_basis-driven limits unless user explicitly overrides with xlim/ylim.
    x_min = 0.0
    y_min = 0.0
    if xlim is None and x_min is not None and x_max > x_min:
        xlim = (x_min, x_max)
    line_x = []
    line_y = []
    point_x = []
    point_y = []
    for line in ax.get_lines():
        line_x.extend(np.asarray(line.get_xdata(), dtype=float).tolist())
        line_y.extend(np.asarray(line.get_ydata(), dtype=float).tolist())
    for key in ("ff3", "proxy3", "ff5", "proxy5"):
        point_x.extend([float(points[key]["gmv"]["vol"]), float(points[key]["tan"]["vol"])])
        point_y.extend([float(points[key]["gmv"]["mean"]), float(points[key]["tan"]["mean"])])
    _set_axis_limits(
        ax,
        np.asarray(line_x, dtype=float),
        np.asarray(line_y, dtype=float),
        anchor_origin=anchor_origin,
        xlim=xlim,
        ylim=ylim,
    )
    _apply_ymin(ax, y_min)
    _apply_visible_ymax(ax)
    _apply_percent_axes(ax, "Volatility (monthly, excess)", "Expected excess return (monthly)")
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])
    if show_title:
        if meta:
            title = f"Task 6: FF Factors vs Practical Proxies ({meta.get('start', '')} to {meta.get('end', '')})"
            ax.set_title(title)
        else:
            ax.set_title("Task 6: FF Factors vs Practical Proxies")
    _apply_report_grid(ax)
    scope6_lines, scope6_markers = scope6_legend_handles()
    ax.legend(handles=scope6_lines + scope6_markers, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_scope8_proxy_panels(
    scope8_proxy_result: dict,
    constraint_label: str = "w_i>=0.00",
    include_ff3_unconstrained_tan: bool = False,
    limit_basis: str = "tangency_vol_rank",
    limit_mult: float = 1.1,
    tan_vol_rank: int = 2,
    efficient_frontier_only: bool = True,
    figsize: tuple[float, float] | None = None,
    panel_layout: str | None = None,
    show_title: bool | None = None,
):
    """
    Scope 8 proxy panels with constrained frontier overlays.

    Left panel: FF3 vs Proxy3
    Right panel: FF5 vs Proxy5
    """
    curves = scope8_proxy_result["plot_data"]["curves"]
    points = scope8_proxy_result["plot_data"]["points"]
    c_curves = scope8_proxy_result["plot_data"].get("constrained_curves", {})
    c_points = scope8_proxy_result["plot_data"].get("constrained_tangency_points", {})
    meta = scope8_proxy_result.get("inputs", {})

    if figsize is None:
        figsize = _resolve_panel_figsize(panel_layout)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])

    pairs = [("ff3", "proxy3", "ff3"), ("ff5", "proxy5", "ff5")]
    titles = ["Scope 8: FF3 vs Proxy-3", "Scope 8: FF5 vs Proxy-5"]
    series_colors = {
        "ff3": "C0",
        "proxy3": "C1",
        "ff5": "C2",
        "proxy5": "C3",
    }

    for ax, (lhs, rhs, base), title in zip(axes, pairs, titles):
        # Unconstrained frontiers
        x, y = _frontier_arrays(curves[lhs], points[lhs]["gmv"]["mean"], efficient_frontier_only)
        ax.plot(
            x,
            y,
            color=series_colors[lhs],
            linestyle="-",
            linewidth=1.6,
            label=f"{lhs.upper()} frontier",
        )
        x, y = _frontier_arrays(curves[rhs], points[rhs]["gmv"]["mean"], efficient_frontier_only)
        ax.plot(
            x,
            y,
            color=series_colors[rhs],
            linestyle="--",
            linewidth=1.6,
            label=f"{rhs.upper()} frontier",
        )

        # Constrained frontier overlays (same colors, dotted style)
        if constraint_label in c_curves.get(lhs, {}):
            cc = c_curves[lhs][constraint_label]
            cc_vols = np.asarray(cc["vols"], dtype=float)
            cc_means = np.asarray(cc["means"], dtype=float)
            if efficient_frontier_only and len(cc_means) > 0:
                i_gmv = int(np.argmin(cc_vols))
                cc_mask = cc_means >= float(cc_means[i_gmv])
                cc_vols, cc_means = cc_vols[cc_mask], cc_means[cc_mask]
            ax.plot(
                cc_vols,
                cc_means,
                color=series_colors[lhs],
                linestyle=":",
                linewidth=1.8,
                label=f"{lhs.upper()} constrained",
            )
        if constraint_label in c_curves.get(rhs, {}):
            cc = c_curves[rhs][constraint_label]
            cc_vols = np.asarray(cc["vols"], dtype=float)
            cc_means = np.asarray(cc["means"], dtype=float)
            if efficient_frontier_only and len(cc_means) > 0:
                i_gmv = int(np.argmin(cc_vols))
                cc_mask = cc_means >= float(cc_means[i_gmv])
                cc_vols, cc_means = cc_vols[cc_mask], cc_means[cc_mask]
            c_color = "C4" if rhs == "proxy5" else series_colors[rhs]
            c_style = "-." if rhs == "proxy5" else ":"
            ax.plot(
                cc_vols,
                cc_means,
                color=c_color,
                linestyle=c_style,
                linewidth=2.1,
                zorder=3,
                label=f"{rhs.upper()} constrained",
            )

        # Unconstrained tangency markers
        if include_ff3_unconstrained_tan or lhs != "ff3":
            ax.scatter(
                points[lhs]["tan"]["vol"],
                points[lhs]["tan"]["mean"],
                marker="*",
                s=70,
                color=series_colors[lhs],
                edgecolor="none",
                zorder=6,
                label=f"{lhs.upper()} TAN",
            )
        ax.scatter(
            points[rhs]["tan"]["vol"],
            points[rhs]["tan"]["mean"],
            marker="*",
            s=70,
            facecolors="none",
            edgecolors=series_colors[rhs],
            linewidths=1.2,
            zorder=6,
            label=f"{rhs.upper()} TAN",
        )

        # Constrained tangency markers
        cp_l = c_points.get(lhs, {}).get(constraint_label)
        cp_r = c_points.get(rhs, {}).get(constraint_label)
        if cp_l is not None:
            ax.scatter(
                cp_l["vol"],
                cp_l["mean"],
                marker="X",
                s=70,
                color=series_colors[lhs],
                zorder=7,
                label=f"{lhs.upper()} TAN ({constraint_label})",
            )
        if cp_r is not None:
            cp_color = "C4" if rhs == "proxy5" else series_colors[rhs]
            ax.scatter(
                cp_r["vol"],
                cp_r["mean"],
                marker="X",
                s=70,
                color=cp_color,
                edgecolor="white",
                linewidths=0.6,
                zorder=7,
                label=f"{rhs.upper()} TAN ({constraint_label})",
            )

        _apply_percent_axes(ax, "Volatility (monthly, excess)", "Expected excess return (monthly)")
        if show_title:
            ax.set_title(title)
        _apply_report_grid(ax)
        ax.legend(loc="upper left", fontsize=7)

    if limit_basis == "ff5_proxy5":
        x_candidates = []
        y_candidates = []
        for key in ("ff5", "proxy5"):
            v, m = _frontier_arrays(curves[key], points[key]["gmv"]["mean"], efficient_frontier_only)
            if len(v) > 0:
                x_candidates.append(float(np.nanmax(v)))
            if len(m) > 0:
                y_candidates.append(float(np.nanmax(m)))
            if constraint_label in c_curves.get(key, {}):
                cc = c_curves[key][constraint_label]
                cc_vols = np.asarray(cc["vols"], dtype=float)
                cc_means = np.asarray(cc["means"], dtype=float)
                if efficient_frontier_only and len(cc_means) > 0:
                    i_gmv = int(np.argmin(cc_vols))
                    cc_mask = cc_means >= float(cc_means[i_gmv])
                    cc_vols, cc_means = cc_vols[cc_mask], cc_means[cc_mask]
                if len(cc_vols) > 0:
                    x_candidates.append(float(np.nanmax(cc_vols)))
                if len(cc_means) > 0:
                    y_candidates.append(float(np.nanmax(cc_means)))

        x_max = max(x_candidates) * float(limit_mult) if x_candidates else None
        for ax in axes:
            if x_max is not None and np.isfinite(x_max):
                ax.set_xlim(0, x_max)
            _apply_ymin(ax, 0.0)
            _apply_visible_ymax(ax)
    elif limit_basis == "ff5_tan":
        # Scope 6-like framing: scale from FF5 tangency location.
        ff5_tan = points["ff5"]["tan"]
        x_max = float(ff5_tan["vol"]) * float(limit_mult)
        x_min, y_min = (0.0, 0.0)
        for ax in axes:
            ax.set_xlim(x_min if x_min is not None else 0.0, x_max)
            _apply_ymin(ax, y_min if y_min is not None else 0.0)
            _apply_visible_ymax(ax)
    elif limit_basis == "tangency_vol_rank":
        tan_points = [points[key]["tan"] for key in ("ff3", "proxy3", "ff5", "proxy5")]
        x_min, y_min = (0.0, 0.0)
        ref_point = _ranked_tangency_point(tan_points, rank=tan_vol_rank)
        x_max = float(limit_mult) * float(ref_point["vol"]) if ref_point is not None else 0.0
        for ax in axes:
            ax.set_xlim(x_min if x_min is not None else 0.0, x_max)
            _apply_ymin(ax, y_min if y_min is not None else 0.0)
            _apply_visible_ymax(ax)
    else:
        raise ValueError("limit_basis must be one of {'ff5_proxy5', 'ff5_tan', 'tangency_vol_rank'}")

    if show_title and meta:
        start = meta.get("is_start") or meta.get("start")
        end = meta.get("oos_end") or meta.get("end")
        if start and end:
            fig.suptitle(f"Question 8: Constrained Frontier Extension ({start} to {end})")
        else:
            fig.suptitle("Question 8: Constrained Frontier Extension")
    fig.tight_layout()
    return fig


def plot_scope8_2_is_oos_panels(
    scope8_2_result: dict,
    constraint_label: str = "w_i>=0.00",
    universes: tuple[str, ...] | None = None,
    limit_basis: str = "per_panel",
    limit_mult: float = 1.2,
    figsize: tuple[float, float] | None = None,
    show_title: bool | None = None,
):
    """
    Scope 8.2 plot: full IS/OOS frontiers + points for industries, FF5, Proxy5.
    """
    data = scope8_2_result["plot_data"]["universes"]
    base_order = [u for u in ("industries", "ff5", "proxy5") if u in data]
    if universes is None:
        order = base_order
    else:
        requested = [u for u in universes if u in data]
        if len(requested) == 0:
            raise ValueError("Requested universes are not present in scope8_2_result.")
        order = requested
    if len(order) == 0:
        raise ValueError("No universes found in scope8_2_result plot_data.")

    if figsize is None:
        figsize = (5.0 * len(order), 4.0)
    fig, axes = plt.subplots(1, len(order), figsize=figsize)
    if len(order) == 1:
        axes = [axes]
    if show_title is None:
        show_title = bool(PLOT_DEFAULTS["show_titles"])

    color_is = "C0"
    color_oos = "C1"
    color_c = "C2"

    for ax, u in zip(axes, order):
        block = data[u]
        c_is = block["is"]["curve"]
        c_oos = block["oos"]["curve"]
        pts = block["points"]

        x, y = _frontier_arrays(c_is, pts["is_opt"]["gmv"]["mean"], efficient_frontier_only=True)
        ax.plot(x, y, color=color_is, linestyle="-", linewidth=1.8, label="IS frontier")
        x, y = _frontier_arrays(c_oos, pts["oos_opt"]["gmv"]["mean"], efficient_frontier_only=True)
        ax.plot(x, y, color=color_oos, linestyle="-", linewidth=1.8, label="OOS frontier")

        cc_is = block["is"]["constrained_curves"].get(constraint_label)
        cc_oos = block["oos"]["constrained_curves"].get(constraint_label)
        if cc_is is not None:
            ccv = np.asarray(cc_is["vols"], dtype=float)
            ccm = np.asarray(cc_is["means"], dtype=float)
            if len(ccm) > 0:
                i_gmv = int(np.argmin(ccv))
                m_gmv = float(ccm[i_gmv])
                mask = ccm >= m_gmv
                ccv, ccm = ccv[mask], ccm[mask]
            ax.plot(ccv, ccm, color=color_is, linestyle=":", linewidth=1.8, label=f"IS constrained ({constraint_label})")
        if cc_oos is not None:
            ccv = np.asarray(cc_oos["vols"], dtype=float)
            ccm = np.asarray(cc_oos["means"], dtype=float)
            if len(ccm) > 0:
                i_gmv = int(np.argmin(ccv))
                m_gmv = float(ccm[i_gmv])
                mask = ccm >= m_gmv
                ccv, ccm = ccv[mask], ccm[mask]
            ax.plot(ccv, ccm, color=color_oos, linestyle=":", linewidth=1.8, label=f"OOS constrained ({constraint_label})")

        # IS/OOS optimal points
        ax.scatter(pts["is_opt"]["gmv"]["vol"], pts["is_opt"]["gmv"]["mean"], marker="o", s=55, color=color_is, zorder=5, label="IS GMV(opt)")
        ax.scatter(pts["is_opt"]["tan"]["vol"], pts["is_opt"]["tan"]["mean"], marker="*", s=70, color=color_is, zorder=6, label="IS TAN(opt)")
        ax.scatter(pts["oos_opt"]["gmv"]["vol"], pts["oos_opt"]["gmv"]["mean"], marker="o", s=55, facecolors="none", edgecolors=color_oos, linewidths=1.2, zorder=5, label="OOS GMV(opt)")
        ax.scatter(pts["oos_opt"]["tan"]["vol"], pts["oos_opt"]["tan"]["mean"], marker="*", s=70, facecolors="none", edgecolors=color_oos, linewidths=1.2, zorder=6, label="OOS TAN(opt)")

        # IS-selected portfolios in IS and OOS
        for label, m in [("unconstrained", "x"), (constraint_label, "X")]:
            p_is = pts["is_selected_on_is"].get(label)
            p_oos = pts["is_selected_on_oos"].get(label)
            if p_is is not None:
                ax.scatter(p_is["vol"], p_is["mean"], marker=m, s=65, color=color_c, zorder=7, label=f"IS sel on IS ({label})")
            if p_oos is not None:
                ax.scatter(
                    p_oos["vol"],
                    p_oos["mean"],
                    marker=m,
                    s=65,
                    color=color_c,
                    alpha=0.8,
                    zorder=7,
                    label=f"IS sel on OOS ({label})",
                )

        _apply_percent_axes(ax, "Volatility (monthly)", "Expected return (monthly)")
        title = f"Scope 8.2: {u.upper()}"
        if show_title:
            ax.set_title(title)
        _apply_report_grid(ax)
        ax.legend(loc="upper left", fontsize=7)
        label_text = {"industries": "Industries", "ff5": "FF5", "proxy5": "Proxy5"}.get(u, u)
        ax.text(
            0.98,
            0.03,
            label_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    if limit_basis == "ff5":
        if "ff5" not in data:
            raise ValueError("limit_basis='ff5' requires FF5 universe in plot_data.")
        b = data["ff5"]
        # Use FF5 IS/OOS unconstrained TAN and constrained frontier envelope as scaling anchors.
        x_candidates = [
            float(b["points"]["is_opt"]["tan"]["vol"]),
            float(b["points"]["oos_opt"]["tan"]["vol"]),
        ]
        y_candidates = [
            float(b["points"]["is_opt"]["tan"]["mean"]),
            float(b["points"]["oos_opt"]["tan"]["mean"]),
        ]
        cc_is = b["is"]["constrained_curves"].get(constraint_label)
        cc_oos = b["oos"]["constrained_curves"].get(constraint_label)
        for cc in (cc_is, cc_oos):
            if cc is not None:
                ccv = np.asarray(cc["vols"], dtype=float)
                ccm = np.asarray(cc["means"], dtype=float)
                if len(ccv) > 0:
                    x_candidates.append(float(np.nanmax(ccv)))
                if len(ccm) > 0:
                    y_candidates.append(float(np.nanmax(ccm)))
        x_max = max(x_candidates) * float(limit_mult)
        for ax in axes:
            ax.set_xlim(0, x_max)
            _apply_ymin(ax, 0.0)
            _apply_visible_ymax(ax)
    elif limit_basis != "per_panel":
        raise ValueError("limit_basis must be one of {'per_panel', 'ff5'}")

    fig.tight_layout()
    return fig
