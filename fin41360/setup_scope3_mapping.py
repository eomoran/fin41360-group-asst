"""
CLI entrypoint to build a validated Scope 3 stock -> FF30 mapping.

Example:
    python -m fin41360.setup_scope3_mapping \
      --candidates fin41360_data/processed/scope3_stock_candidates.csv \
      --sic-ranges fin41360_data/processed/ff30_sic_ranges.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .scope3_mapping import Scope3SelectionConfig, build_scope3_mapping_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated Scope 3 FF30 stock mapping.")
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="CSV with candidate stocks and metadata.",
    )
    parser.add_argument(
        "--sic-ranges",
        type=Path,
        required=True,
        help="CSV with official FF30 SIC ranges (from Siccodes30 source).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: fin41360_data/processed).",
    )
    parser.add_argument(
        "--sample-start",
        type=str,
        default="1990-01",
        help="Sample start month (YYYY-MM).",
    )
    parser.add_argument(
        "--sample-end",
        type=str,
        default="2025-12",
        help="Sample end month (YYYY-MM).",
    )
    parser.add_argument(
        "--min-sample-months",
        type=int,
        default=360,
        help="Minimum in-sample months preferred when selecting representatives.",
    )
    args = parser.parse_args()

    config = Scope3SelectionConfig(
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        min_sample_months=args.min_sample_months,
    )
    outputs = build_scope3_mapping_outputs(
        candidates_csv=args.candidates,
        ff30_sic_ranges_csv=args.sic_ranges,
        out_dir=args.out_dir,
        config=config,
    )

    print("Scope 3 mapping outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
