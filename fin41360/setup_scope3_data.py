"""
One-shot setup for Scope 3 stock universe + SIC-based FF30 mapping.

This script is designed for reproducibility on a fresh machine:
- it reuses existing files by default (no unnecessary downloads),
- it caches network downloads under fin41360_data/raw/scope3/,
- it writes deterministic processed outputs under fin41360_data/processed/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROCESSED_DIR
from .scope3_mapping import Scope3SelectionConfig, build_scope3_mapping_outputs
from .scope3_universe import build_scope3_candidates_csv, ensure_ff30_sic_ranges_csv
from .stock_data import build_scope3_selected_stock_returns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup Scope 3 stock candidates and SIC-based FF30 mapping."
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh cached downloads and rebuild outputs.")
    parser.add_argument(
        "--no-sp500",
        action="store_true",
        help="Do not augment candidate universe with S&P 500 symbols.",
    )
    parser.add_argument(
        "--sample-start",
        type=str,
        default="1990-01",
        help="Sample start month for metadata and selection (YYYY-MM).",
    )
    parser.add_argument(
        "--sample-end",
        type=str,
        default="2025-12",
        help="Sample end month for metadata and selection (YYYY-MM).",
    )
    parser.add_argument(
        "--min-sample-months",
        type=int,
        default=360,
        help="Minimum preferred sample months for representative selection.",
    )
    parser.add_argument(
        "--sec-user-agent",
        type=str,
        default=None,
        help="SEC-compliant user agent string, e.g. 'Name email@domain.com'.",
    )
    parser.add_argument(
        "--candidates-out",
        type=Path,
        default=PROCESSED_DIR / "scope3_stock_candidates.csv",
        help="Path to candidate universe CSV output.",
    )
    parser.add_argument(
        "--sic-ranges-out",
        type=Path,
        default=PROCESSED_DIR / "ff30_sic_ranges.csv",
        help="Path to FF30 SIC ranges CSV output.",
    )
    parser.add_argument(
        "--skip-selected-returns",
        action="store_true",
        help="Skip building scope3_selected_30_stock_monthly_gross.csv from selected mapping.",
    )
    args = parser.parse_args()
    print("[Scope3] Starting setup_scope3_data", flush=True)
    print(
        "[Scope3] Config: "
        f"refresh={args.refresh}, include_sp500={not args.no_sp500}, "
        f"sample={args.sample_start}..{args.sample_end}, min_months={args.min_sample_months}",
        flush=True,
    )

    candidates_csv = build_scope3_candidates_csv(
        out_csv=args.candidates_out,
        start=args.sample_start,
        end=args.sample_end,
        include_sp500=not args.no_sp500,
        refresh=args.refresh,
        sec_user_agent=args.sec_user_agent,
    )
    sic_csv = ensure_ff30_sic_ranges_csv(
        out_csv=args.sic_ranges_out,
        refresh=args.refresh,
    )

    config = Scope3SelectionConfig(
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        min_sample_months=args.min_sample_months,
    )
    outputs = build_scope3_mapping_outputs(
        candidates_csv=candidates_csv,
        ff30_sic_ranges_csv=sic_csv,
        out_dir=PROCESSED_DIR,
        config=config,
    )

    selected_returns_path = None
    if not args.skip_selected_returns:
        print("[Scope3] Stage 8/8: Building selected-30 stock return matrix", flush=True)
        selected_returns_path = build_scope3_selected_stock_returns(
            start=args.sample_start,
            end=args.sample_end,
            mapping_file=outputs["selected_30"],
            out_file=PROCESSED_DIR / "scope3_selected_30_stock_monthly_gross.csv",
        )

    print("[Scope3] Setup complete.")
    print(f"  candidates_csv: {candidates_csv}")
    print(f"  sic_ranges_csv: {sic_csv}")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
    if selected_returns_path is not None:
        print(f"  selected_30_stock_returns: {selected_returns_path}")


if __name__ == "__main__":
    main()
