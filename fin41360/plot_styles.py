"""
Central styling for mean–variance frontier and related charts (FIN41360).

Use style(role, series, proxy=False) for all plot/scatter calls. Colours and
markers come from one SERIES palette. proxy=True applies a variant linestyle
(and CML + scatter marker) for factor proxies (ff3 → Proxy-3, ff5 → Proxy-5)
so we don't duplicate series dicts. No hardcoded color= in the notebook.
"""

# --- Base styles (kwargs for ax.plot / ax.scatter) ---
FRONTIER_STYLE = {
    "linewidth": 1.5,
    "zorder": 2,
}

CML_STYLE = {
    "linewidth": 1.5,
    "linestyle": "--",
    "alpha": 0.85,
    "zorder": 2,
}

GMV_STYLE = {
    "marker": "o",
    "s": 60,
    "zorder": 5,
}

TAN_STYLE = {
    "marker": "*",
    "s": 95,
    "zorder": 6,
}

# --- Single series palette (no scope: one dict, series key is enough) ---
SERIES = {
    "sample": {"color": "C0", "linestyle": "-", "label": "Sample (no shrinkage)"},
    "bs_mean": {"color": "C1", "linestyle": "--", "label": "Bayes-Stein mean shrinkage"},
    "bs_mean_cov": {"color": "C2", "linestyle": "-.", "label": "Bayes-Stein mean+cov shrinkage"},
    "industries": {"color": "C0", "linestyle": "-", "label": "Industries"},
    "cml": {"color": "C1", "label": "CML"},
    "ff3": {"color": "C1", "linestyle": "-", "label": "FF3 factors"},
    "ff5": {"color": "C2", "linestyle": "-", "label": "FF5 factors"},
}

# --- Proxy variants: distinct colours (not hue-paired with FF bases) ---
PROXY_COLORS = {
    "ff3": "C3",
    "ff5": "C4",
}
PROXY_LABELS = {"ff3": "3-factor proxy basket", "ff5": "5-factor proxy basket"}

# --- Scope 3 (kept for notebook that still uses nested structure) ---
ESTIMATOR_STYLE = {k: v for k, v in SERIES.items() if k in ("sample", "bs_mean", "bs_mean_cov")}
SCOPE3_PLOT_STYLE = {
    "estimator": {
        "sample": {"linestyle": "-", "label": "Sample estimates"},
        "bs_mean": {"linestyle": "--", "label": "BS means"},
        "bs_mean_cov": {"linestyle": "-.", "label": "BS mean+cov"},
    },
    "universe": {
        "industry": {"color": "C0", "label": "30 industries"},
        "stock": {"color": "C3", "label": "30 stocks"},
    },
    "portfolio_marker": {"GMV": "o", "TAN": "*"},
}

# --- Legacy (scope4 used via style now; keep for any direct refs) ---
SCOPE4_FRONTIER_STYLE = {**FRONTIER_STYLE, **SERIES["industries"]}
SCOPE4_CML_STYLE = {**CML_STYLE, **SERIES["cml"]}

ROLE_BASES = {
    "frontier": FRONTIER_STYLE,
    "cml": CML_STYLE,
    "gmv": GMV_STYLE,
    "tan": TAN_STYLE,
}


def style(
    role: str,
    series: str,
    *,
    proxy: bool = False,
    label: str | None = None,
) -> dict:
    """
    Return a single style dict for ax.plot(...) or ax.scatter(...).

    - role: "frontier" | "cml" | "gmv" | "tan"
    - series: key into SERIES (e.g. "industries", "ff3", "sample")
    - proxy: if True and series in ("ff3", "ff5"), apply proxy colour variant
    - label: optional legend override (mainly for CML)

    Example: ax.plot(..., **style("frontier", "ff3"))
             ax.plot(..., **style("frontier", "ff3", proxy=True))   # Proxy-3
             ax.plot(..., **style("cml", "ff3", proxy=True))       # CML with proxy colour variant
    """
    base = ROLE_BASES.get(role)
    if base is None:
        raise ValueError(f"Unknown role: {role}")
    series_d = SERIES.get(series)
    if series_d is None:
        raise ValueError(f"Unknown series: {series!r}")
    out = {**base, **series_d}

    # Keep CML visual semantics consistent across scopes.
    if role == "cml":
        out["linestyle"] = CML_STYLE["linestyle"]
        out["alpha"] = CML_STYLE["alpha"]

    if proxy and series in ("ff3", "ff5"):
        out["color"] = PROXY_COLORS[series]
        out["label"] = PROXY_LABELS[series]
        if role in ("gmv", "tan"):
            # Keep same marker semantics as base; hollow so proxy is distinguishable.
            ec = out.get("color", series_d.get("color"))
            out["facecolors"] = "none"
            out["edgecolors"] = ec
            out.pop("color", None)  # avoid scatter using 'color' and overriding face/edge

    # Keep frontier/CML labels by default, suppress marker labels unless overridden.
    if role in ("gmv", "tan") and label is None:
        out["label"] = "_nolegend_"

    if label is not None:
        out["label"] = label
    return out


def scope6_legend_handles():
    """Return (line_handles, marker_handles) for Scope 6 single-panel."""
    from matplotlib.lines import Line2D

    line_handles = [
        Line2D([0], [0], color=SERIES["ff3"]["color"], ls="-", label=SERIES["ff3"]["label"]),
        Line2D([0], [0], color=PROXY_COLORS["ff3"], ls="-", label=PROXY_LABELS["ff3"]),
        Line2D([0], [0], color=SERIES["ff5"]["color"], ls="-", label=SERIES["ff5"]["label"]),
        Line2D([0], [0], color=PROXY_COLORS["ff5"], ls="-", label=PROXY_LABELS["ff5"]),
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", color="gray", ls="", markersize=8, label="GMV"),
        Line2D([0], [0], marker="*", color="gray", ls="", markersize=11, label="TAN"),
    ]
    return (line_handles, marker_handles)


def scope6_panel_legend_handles(series: str):
    """Return (line_handles, marker_handles) for one Scope 6 panel (ff3 or ff5): 2 lines + 3 markers."""
    from matplotlib.lines import Line2D

    line_handles = [
        Line2D([0], [0], color=SERIES[series]["color"], ls="-", label=SERIES[series]["label"]),
        Line2D([0], [0], color=PROXY_COLORS[series], ls="-", label=PROXY_LABELS[series]),
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", color="gray", ls="", markersize=8, label="GMV"),
        Line2D([0], [0], marker="*", color="gray", ls="", markersize=11, label="TAN"),
    ]
    return (line_handles, marker_handles)
