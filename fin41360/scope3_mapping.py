"""
Scope 3 helper utilities for building an auditable stock -> FF30 industry map.

The key idea is to avoid manual ticker-to-industry assignment and instead:
1) map each candidate stock via SIC code using the official FF30 SIC ranges,
2) apply deterministic selection rules to choose one representative stock per
   FF30 industry for the assignment's one-stock-per-industry requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PROCESSED_DIR


FF30_INDUSTRIES = [
    "Food",
    "Beer",
    "Smoke",
    "Games",
    "Books",
    "Hshld",
    "Clths",
    "Hlth",
    "Chems",
    "Txtls",
    "Cnstr",
    "Steel",
    "FabPr",
    "ElcEq",
    "Autos",
    "Carry",
    "Mines",
    "Coal",
    "Oil",
    "Util",
    "Telcm",
    "Servs",
    "BusEq",
    "Paper",
    "Trans",
    "Whlsl",
    "Rtail",
    "Meals",
    "Fin",
    "Other",
]


@dataclass(frozen=True)
class Scope3SelectionConfig:
    sample_start: str = "1990-01"
    sample_end: str = "2025-12"
    min_sample_months: int = 360


def _period_yyyymm(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M")


def _month_count(start: pd.Period, end: pd.Period) -> int:
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    return int(end.ordinal - start.ordinal + 1)


def load_ff30_sic_ranges(csv_path: str | Path) -> pd.DataFrame:
    """
    Load FF30 SIC bucket ranges from CSV.

    Required columns:
    - ff30_industry
    - sic_start
    - sic_end
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    req = {"ff30_industry", "sic_start", "sic_end"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required SIC range columns in {path}: {sorted(missing)}")

    out = df.copy()
    out["ff30_industry"] = out["ff30_industry"].astype(str).str.strip()
    out["sic_start"] = pd.to_numeric(out["sic_start"], errors="coerce").astype("Int64")
    out["sic_end"] = pd.to_numeric(out["sic_end"], errors="coerce").astype("Int64")

    bad = out[out["sic_start"].isna() | out["sic_end"].isna() | (out["sic_end"] < out["sic_start"])]
    if not bad.empty:
        raise ValueError("SIC range file contains invalid sic_start/sic_end rows.")

    unknown = sorted(set(out["ff30_industry"]) - set(FF30_INDUSTRIES))
    if unknown:
        raise ValueError(f"SIC range file includes unknown FF30 labels: {unknown}")

    return out


def map_candidates_to_ff30(
    candidates: pd.DataFrame,
    ff30_sic_ranges: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map candidate stocks to FF30 industries using SIC ranges.

    Required candidate columns:
    - ticker
    - sic
    - market_cap_usd
    - median_dollar_volume_usd
    - first_return_month
    - last_return_month
    """
    req = {
        "ticker",
        "sic",
        "market_cap_usd",
        "median_dollar_volume_usd",
        "first_return_month",
        "last_return_month",
    }
    missing = req.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing required candidate columns: {sorted(missing)}")

    out = candidates.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["sic"] = pd.to_numeric(out["sic"], errors="coerce").astype("Int64")
    out["market_cap_usd"] = pd.to_numeric(out["market_cap_usd"], errors="coerce")
    out["median_dollar_volume_usd"] = pd.to_numeric(out["median_dollar_volume_usd"], errors="coerce")
    out["first_return_month"] = _period_yyyymm(out["first_return_month"])
    out["last_return_month"] = _period_yyyymm(out["last_return_month"])

    mapped_industry: list[Optional[str]] = []
    mapped_status: list[str] = []
    mapped_matches: list[int] = []

    for sic in out["sic"]:
        if pd.isna(sic):
            mapped_industry.append(None)
            mapped_status.append("missing_sic")
            mapped_matches.append(0)
            continue

        matches = ff30_sic_ranges[
            (ff30_sic_ranges["sic_start"] <= int(sic))
            & (int(sic) <= ff30_sic_ranges["sic_end"])
        ]
        n = len(matches)
        mapped_matches.append(n)
        if n == 1:
            mapped_industry.append(matches.iloc[0]["ff30_industry"])
            mapped_status.append("mapped_unique")
        elif n == 0:
            mapped_industry.append(None)
            mapped_status.append("unmapped_sic")
        else:
            mapped_industry.append(None)
            mapped_status.append("ambiguous_sic")

    out["ff30_industry"] = mapped_industry
    out["mapping_status"] = mapped_status
    out["mapping_match_count"] = mapped_matches
    out["history_months_total"] = [
        _month_count(s, e) for s, e in zip(out["first_return_month"], out["last_return_month"])
    ]
    return out


def select_scope3_representatives(
    mapped_candidates: pd.DataFrame,
    config: Scope3SelectionConfig = Scope3SelectionConfig(),
) -> pd.DataFrame:
    """
    Choose one representative stock per FF30 industry using deterministic rules.
    """
    start = pd.Period(config.sample_start, freq="M")
    end = pd.Period(config.sample_end, freq="M")
    sample_months = _month_count(start, end)

    candidates = mapped_candidates.copy()
    candidates = candidates[candidates["mapping_status"] == "mapped_unique"].copy()
    if candidates.empty:
        candidates["history_months_in_sample"] = pd.Series(dtype=int)
        candidates["has_full_sample"] = pd.Series(dtype=bool)
        candidates["meets_min_sample"] = pd.Series(dtype=bool)
    else:
        in_sample_months = []
        for _, row in candidates.iterrows():
            overlap_start = max(row["first_return_month"], start)
            overlap_end = min(row["last_return_month"], end)
            in_sample_months.append(_month_count(overlap_start, overlap_end))
        candidates["history_months_in_sample"] = in_sample_months
        candidates["has_full_sample"] = candidates["history_months_in_sample"] >= sample_months
        candidates["meets_min_sample"] = candidates["history_months_in_sample"] >= config.min_sample_months

    selected_rows = []
    for ind in FF30_INDUSTRIES:
        group = candidates[candidates["ff30_industry"] == ind].copy()
        if group.empty:
            selected_rows.append(
                {
                    "ff30_industry": ind,
                    "ticker": None,
                    "selection_status": "missing_candidate",
                    "selection_rule": "no uniquely mapped stock candidates available",
                }
            )
            continue

        group = group.sort_values(
            by=[
                "has_full_sample",
                "meets_min_sample",
                "history_months_in_sample",
                "median_dollar_volume_usd",
                "market_cap_usd",
                "ticker",
            ],
            ascending=[False, False, False, False, False, True],
        )
        pick = group.iloc[0]
        status = "selected"
        if not bool(pick["meets_min_sample"]):
            status = "selected_below_min_months"

        selected_rows.append(
            {
                "ff30_industry": ind,
                "ticker": pick["ticker"],
                "sic": int(pick["sic"]) if not pd.isna(pick["sic"]) else None,
                "market_cap_usd": float(pick["market_cap_usd"]) if not pd.isna(pick["market_cap_usd"]) else None,
                "median_dollar_volume_usd": (
                    float(pick["median_dollar_volume_usd"])
                    if not pd.isna(pick["median_dollar_volume_usd"])
                    else None
                ),
                "first_return_month": str(pick["first_return_month"]),
                "last_return_month": str(pick["last_return_month"]),
                "history_months_in_sample": int(pick["history_months_in_sample"]),
                "has_full_sample": bool(pick["has_full_sample"]),
                "meets_min_sample": bool(pick["meets_min_sample"]),
                "selection_status": status,
                "selection_rule": (
                    "ranked by full-sample coverage, min sample coverage, "
                    "in-sample history, median dollar volume, market cap"
                ),
            }
        )

    return pd.DataFrame(selected_rows)


def build_scope3_mapping_outputs(
    candidates_csv: str | Path,
    ff30_sic_ranges_csv: str | Path,
    out_dir: str | Path | None = None,
    config: Scope3SelectionConfig = Scope3SelectionConfig(),
) -> dict[str, Path]:
    """
    Run full scope-3 mapping workflow and write output CSV artifacts.
    """
    print("[Scope3] Stage 5/7: Loading inputs for SIC->FF30 mapping", flush=True)
    target = Path(out_dir) if out_dir is not None else PROCESSED_DIR
    target.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(candidates_csv)
    ff30_ranges = load_ff30_sic_ranges(ff30_sic_ranges_csv)
    print(
        "[Scope3] Inputs loaded: "
        f"candidates={len(candidates)}, sic_ranges={len(ff30_ranges)}",
        flush=True,
    )
    print("[Scope3] Stage 6/7: Mapping candidates and selecting one per FF30 industry", flush=True)
    mapped = map_candidates_to_ff30(candidates, ff30_ranges)
    selected = select_scope3_representatives(mapped, config=config)

    print("[Scope3] Stage 7/7: Writing mapping outputs", flush=True)
    summary = pd.DataFrame(
        [
            {
                "n_candidates": len(mapped),
                "n_mapped_unique": int((mapped["mapping_status"] == "mapped_unique").sum()),
                "n_unmapped_sic": int((mapped["mapping_status"] == "unmapped_sic").sum()),
                "n_ambiguous_sic": int((mapped["mapping_status"] == "ambiguous_sic").sum()),
                "n_missing_sic": int((mapped["mapping_status"] == "missing_sic").sum()),
                "n_selected": int((selected["selection_status"].str.startswith("selected")).sum()),
                "n_missing_industries": int((selected["selection_status"] == "missing_candidate").sum()),
                "sample_start": config.sample_start,
                "sample_end": config.sample_end,
                "min_sample_months": config.min_sample_months,
            }
        ]
    )

    mapped_path = target / "scope3_candidates_mapped_ff30.csv"
    selected_path = target / "scope3_selected_30_stocks_ff30.csv"
    summary_path = target / "scope3_mapping_audit_summary.csv"

    mapped_out = mapped.copy()
    mapped_out["first_return_month"] = mapped_out["first_return_month"].astype(str)
    mapped_out["last_return_month"] = mapped_out["last_return_month"].astype(str)
    mapped_out.to_csv(mapped_path, index=False)
    selected.to_csv(selected_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(
        "[Scope3] Mapping outputs written: "
        f"mapped={mapped_path.name}, selected={selected_path.name}, summary={summary_path.name}",
        flush=True,
    )

    return {
        "mapped_candidates": mapped_path,
        "selected_30": selected_path,
        "audit_summary": summary_path,
    }
