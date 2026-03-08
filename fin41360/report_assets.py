"""
Utilities for exporting notebook outputs into report-ready assets.

Design goals:
- Save figures into `report/figures/` with descriptive names.
- Save DataFrame tables into `report/tables/` as CSV and LaTeX.
- Avoid accidental overwrite by default (create versioned filenames).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

import pandas as pd

from .config import PROJECT_ROOT
from .report_tables import (
    build_scope2_table,
    build_scope3_tables,
    build_scope5_table,
    build_scope6_pairwise_table,
    build_scope7_decision_table,
)


def _report_dir() -> Path:
    return PROJECT_ROOT / "report"


def ensure_report_asset_dirs(report_dir: Path | None = None) -> dict[str, Path]:
    """
    Ensure report asset folders exist and return them.
    """
    base = report_dir if report_dir is not None else _report_dir()
    figures = base / "figures"
    tables = base / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return {"report": base, "figures": figures, "tables": tables}


def _versioned_path(path: Path) -> Path:
    """
    If `path` exists, return the next available `_vNN` sibling path.
    """
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_v{i:02d}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def save_figure(
    fig: Any,
    name: str,
    *,
    fmt: str = "png",
    dpi: int = 300,
    overwrite: bool = False,
    close: bool = False,
    figures_dir: Path | None = None,
) -> Path:
    """
    Save a Matplotlib figure into `report/figures`.
    """
    dirs = ensure_report_asset_dirs()
    out_dir = figures_dir if figures_dir is not None else dirs["figures"]
    target = out_dir / f"{name}.{fmt}"
    out_path = target if overwrite else _versioned_path(target)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return out_path


def save_figures(
    figures: Mapping[str, Any],
    *,
    fmt: str = "png",
    dpi: int = 300,
    overwrite: bool = False,
    close: bool = False,
    figures_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Save multiple figures with `{name: fig}` mapping.
    """
    out: dict[str, Path] = {}
    for name, fig in figures.items():
        out[name] = save_figure(
            fig,
            name,
            fmt=fmt,
            dpi=dpi,
            overwrite=overwrite,
            close=close,
            figures_dir=figures_dir,
        )
    return out


def save_table(
    df: pd.DataFrame,
    name: str,
    *,
    overwrite: bool = False,
    index: bool = False,
    float_format: str = "%.4f",
    tables_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Save a DataFrame as CSV and LaTeX in `report/tables`.
    """
    dirs = ensure_report_asset_dirs()
    out_dir = tables_dir if tables_dir is not None else dirs["tables"]
    csv_target = out_dir / f"{name}.csv"
    tex_target = out_dir / f"{name}.tex"
    csv_path = csv_target if overwrite else _versioned_path(csv_target)
    tex_path = tex_target if overwrite else _versioned_path(tex_target)

    df.to_csv(csv_path, index=index)
    latex = df.to_latex(index=index, float_format=float_format.__mod__, escape=True)
    tex_path.write_text(latex)
    return {"csv": csv_path, "tex": tex_path}


def save_scope_summary_tables(
    scope_result: Mapping[str, Any],
    scope_name: str,
    *,
    overwrite: bool = False,
    index: bool = False,
    float_format: str = "%.4f",
    tables_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Export all `summary_tables` DataFrames from a workflow scope output.
    """
    tables = scope_result.get("summary_tables", {})
    out: dict[str, dict[str, Path]] = {}
    for tbl_name, value in tables.items():
        if isinstance(value, pd.DataFrame):
            out[tbl_name] = save_table(
                value,
                f"{scope_name}_{tbl_name}",
                overwrite=overwrite,
                index=index,
                float_format=float_format,
                tables_dir=tables_dir,
            )
    return out


def write_scope3_mapping_note(
    scope3_result: Mapping[str, Any],
    *,
    out_path: Path | None = None,
) -> Path:
    """
    Write a short Scope 3 mapping note with key metadata for report reuse.
    """
    inputs = scope3_result.get("inputs", {})
    diagnostics = scope3_result.get("diagnostics", {})
    default_path = ensure_report_asset_dirs()["tables"] / "scope3_mapping_note.txt"
    target = out_path if out_path is not None else default_path

    lines = [
        "Scope 3 Mapping Note",
        "--------------------",
        "Selection is deterministic and SIC-based (ticker -> SIC -> FF30 industry).",
        f"Common window: {inputs.get('common_start', 'NA')} to {inputs.get('common_end', 'NA')}",
        f"Observations: {inputs.get('n_obs', 'NA')}",
        f"Industry assets: {inputs.get('n_assets_industry', 'NA')}",
        f"Stock assets: {inputs.get('n_assets_stock', 'NA')}",
        f"Stock source: {inputs.get('stock_source', diagnostics.get('stock_source', 'NA'))}",
        f"Stock source file: {inputs.get('stock_source_file', diagnostics.get('stock_source_file', 'NA'))}",
        "Tie-break sequence: coverage, history, liquidity, market cap, alphabetical fallback.",
    ]
    target.write_text("\n".join(lines) + "\n")
    return target


def save_scope_presentation_tables(
    *,
    scope2_result: Mapping[str, Any] | None = None,
    scope3_sensitivity_result: Mapping[str, Any] | None = None,
    scope5_result: Mapping[str, Any] | None = None,
    scope6_result: Mapping[str, Any] | None = None,
    scope7_result: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    index: bool = False,
    float_format: str = "%.4f",
    tables_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Build and save presentation-ready tables for report Scopes 2-7.
    """
    out: dict[str, dict[str, Path]] = {}

    if scope2_result is not None:
        out["scope2_presentation"] = save_table(
            build_scope2_table(dict(scope2_result)),
            "scope2_presentation",
            overwrite=overwrite,
            index=index,
            float_format=float_format,
            tables_dir=tables_dir,
        )
    if scope3_sensitivity_result is not None:
        scope3_tables = build_scope3_tables(dict(scope3_sensitivity_result))
        out["scope3_comparison_presentation"] = save_table(
            scope3_tables["comparison"],
            "scope3_comparison_presentation",
            overwrite=overwrite,
            index=index,
            float_format=float_format,
            tables_dir=tables_dir,
        )
        out["scope3_window_summary"] = save_table(
            scope3_tables["window_summary"],
            "scope3_window_summary",
            overwrite=overwrite,
            index=index,
            float_format=float_format,
            tables_dir=tables_dir,
        )
    if scope5_result is not None:
        out["scope5_presentation"] = save_table(
            build_scope5_table(dict(scope5_result)),
            "scope5_presentation",
            overwrite=overwrite,
            index=index,
            float_format=float_format,
            tables_dir=tables_dir,
        )
    if scope6_result is not None:
        out["scope6_pairwise_presentation"] = save_table(
            build_scope6_pairwise_table(dict(scope6_result)),
            "scope6_pairwise_presentation",
            overwrite=overwrite,
            index=index,
            float_format=float_format,
            tables_dir=tables_dir,
        )
    if scope7_result is not None:
        out["scope7_decision_presentation"] = save_table(
            build_scope7_decision_table(dict(scope7_result)),
            "scope7_decision_presentation",
            overwrite=overwrite,
            index=index,
            float_format=float_format,
            tables_dir=tables_dir,
        )

    return out
