"""
One-shot setup script for FIN41360 Assignment 1 Fama–French data.

Usage (from the project root):

    python -m fin41360.setup_french_data

This will:
1. Discover and download the required ZIP files for:
   - 30 Industry Portfolios (value-weighted)
   - Fama–French 3-Factor model (monthly)
   - Fama–French 5-Factor (2x3) model (monthly)
2. Process all downloaded ZIPs into tidy CSVs under `fin41360_data/processed/`.
3. Print basic diagnostics so you can verify the date ranges and shapes.

EXPLAIN: Keeping this as a small orchestrator script lets you re-run the
entire data-setup step at any time without touching analysis code.
"""

from __future__ import annotations

from .config import DATA_ROOT, RAW_DIR, PROCESSED_DIR
from .download_french import download_all_core_french_zips
from .process_french import (
    process_all_raw_zips,
    load_industry_30_monthly,
    load_ff3_monthly,
    load_ff5_monthly,
)


def main() -> None:
    """Run the full data-setup pipeline."""
    print(f"FIN41360 data root: {DATA_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStep 1: Downloading core Fama–French ZIPs ...")
    ind_zip, ff3_zip, ff5_zip = download_all_core_french_zips()
    print(f"  Industry ZIP: {ind_zip}")
    print(f"  FF3 ZIP:      {ff3_zip}")
    print(f"  FF5 ZIP:      {ff5_zip}")

    print("\nStep 2: Processing all raw ZIPs into tidy CSVs ...")
    _ = process_all_raw_zips()

    print("\nStep 3: Quick sanity checks on processed datasets ...")
    ind = load_industry_30_monthly()
    ff3, rf3 = load_ff3_monthly()
    ff5, rf5 = load_ff5_monthly()

    print("\n30-Industry monthly (gross returns):")
    print(f"  Shape: {ind.shape}")
    print(f"  Date range: {ind.index.min().date()} -> {ind.index.max().date()}")

    print("\nFF3 monthly factors (excess) + RF (gross):")
    print(f"  Factors shape: {ff3.shape}, RF length: {len(rf3)}")
    print(f"  Date range: {ff3.index.min().date()} -> {ff3.index.max().date()}")

    print("\nFF5 monthly factors (excess) + RF (gross):")
    print(f"  Factors shape: {ff5.shape}, RF length: {len(rf5)}")
    print(f"  Date range: {ff5.index.min().date()} -> {ff5.index.max().date()}")

    print("\n✓ FIN41360 Fama–French data setup complete.")


if __name__ == "__main__":
    main()

