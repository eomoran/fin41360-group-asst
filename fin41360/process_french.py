"""
Processing utilities for Fama–French ZIP files for FIN41360.

This module:
- Extracts downloaded ZIP files under `fin41360_data/raw/`.
- Parses the Ken French CSV/TXT formats, which may contain multiple sections
  (e.g. monthly and annual tables, value-weighted vs equal-weighted).
- Saves tidy CSVs to `fin41360_data/processed/`.
- Provides high-level loaders that return:
  * 30-industry monthly value-weighted returns (in gross form R = 1 + r),
  * Fama–French 3-factor and 5-factor monthly tables (RF in gross form,
    factors as excess returns).

The parsing logic is adapted from the FIN50040 utilities, trimmed down
to what we need here.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import zipfile

from .config import RAW_DIR, PROCESSED_DIR


def _extract_zip_file(zip_path: Path, output_dir: Path | None = None) -> Path:
    """
    Extract a ZIP file to `output_dir` (default: folder named after the ZIP).
    """
    if output_dir is None:
        output_dir = zip_path.parent / zip_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    print(f"  Extracted {zip_path.name} to {output_dir}")
    return output_dir


def _parse_section(lines, section_label: str) -> pd.DataFrame | None:
    """
    Parse a single section of a Fama–French CSV file into a DataFrame.
    """
    if not lines:
        return None

    header_idx = None
    for i, line in enumerate(lines[:50]):
        stripped = line.strip()
        if stripped.startswith(",") and stripped.count(",") > 2:
            header_idx = i
            break

    if header_idx is None:
        return None

    section_text = "".join(lines[header_idx:])
    try:
        df = pd.read_csv(
            StringIO(section_text),
            header=0,
            skipinitialspace=True,
            na_values=["-99.99", "-999", "NA", "", "NaN"],
            on_bad_lines="skip",
        )

        # Detect date column (YYYY or YYYYMM) and set as index.
        date_col = None
        for col in df.columns:
            sample_val = str(df[col].iloc[0]) if len(df) > 0 else ""
            if (len(sample_val) == 6 or len(sample_val) == 4) and sample_val.isdigit():
                date_col = col
                break

        if date_col is None:
            date_col = df.columns[0]

        sample_val = str(df[date_col].iloc[0]) if len(df) > 0 else ""
        if len(sample_val) == 6 and sample_val.isdigit():
            df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m", errors="coerce")
        elif len(sample_val) == 4 and sample_val.isdigit():
            df[date_col] = pd.to_datetime(df[date_col].astype(str) + "-01-01", format="%Y-%m-%d", errors="coerce")
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        df = df.rename(columns={date_col: "Date"}).set_index("Date")
        df = df[df.index.notna()]

        # Convert numeric columns and, if needed, scale from percent to decimal.
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if df.abs().max().max() > 1:
            df = df / 100.0

        return df
    except Exception as exc:  # pragma: no cover - defensive
        print(f"        Error parsing section '{section_label}': {exc}")
        return None


def _parse_famafrench_csv_sections(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Parse a Fama–French CSV file that may contain multiple labelled sections.

    Returns
    -------
    dict
        Mapping from section_label -> DataFrame.
    """
    sections: Dict[str, pd.DataFrame] = {}

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_label = None
    section_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if any(
            keyword in stripped
            for keyword in ["Monthly", "Annual", "Value Weight", "Equal Weight", "Average"]
        ):
            # Close previous section if any.
            if current_label is not None and section_start is not None:
                df = _parse_section(lines[section_start:i], current_label)
                if df is not None and not df.empty:
                    sections[current_label] = df
                    print(f"      Found section: {current_label} ({df.shape[0]} rows)")

            current_label = stripped.replace(",", " ").strip()
            section_start = i

        if current_label is None and stripped.startswith(",") and stripped.count(",") > 2:
            current_label = "Value Weighted Returns -- Monthly"
            section_start = 0

    if current_label is not None and section_start is not None:
        df = _parse_section(lines[section_start:], current_label)
        if df is not None and not df.empty:
            sections[current_label] = df
            print(f"      Found section: {current_label} ({df.shape[0]} rows)")

    return sections


def _process_zip_file(zip_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Process a single Fama–French ZIP file and write processed CSVs to PROCESSED_DIR.

    Returns
    -------
    dict
        Mapping from "inner_filename_section_label" -> DataFrame.
    """
    print(f"\nProcessing: {zip_path.name}")
    extract_dir = _extract_zip_file(zip_path)

    csv_files = list(extract_dir.glob("*.csv"))
    txt_files = list(extract_dir.glob("*.txt")) if not csv_files else []
    data_files = csv_files or txt_files

    if not data_files:
        print(f"  ⚠ No CSV or TXT files found in {extract_dir}")
        return {}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}
    for data_file in data_files:
        print(f"  Parsing {data_file.name} ...")
        sections = _parse_famafrench_csv_sections(data_file)

        if not sections:
            continue

        zip_stem = zip_path.stem.replace("_CSV", "").replace("_TXT", "")
        for label, df in sections.items():
            safe_label = (
                label.replace(" ", "_")
                .replace("--", "_")
                .replace(",", "")
            )
            safe_label = "".join(c for c in safe_label if c.isalnum() or c in ("_", "-"))[:50]
            output_name = f"{zip_stem}_{data_file.stem}_{safe_label}.csv"
            output_path = PROCESSED_DIR / output_name

            # Portfolio vs factor handling:
            if "Portfolio" in zip_stem or "Portfolios" in zip_stem:
                # Convert net returns to gross (R = 1 + r)
                df = 1.0 + df
                print("    ✓ Converted portfolio returns to gross (R = 1 + r)")
            elif "Factor" in zip_stem or "Factors" in zip_stem:
                if "RF" in df.columns:
                    df["RF"] = 1.0 + df["RF"]
                    print("    ✓ Converted RF to gross return (R_f = 1 + r_f)")

            df.to_csv(output_path)
            print(f"    ✓ Saved section '{label}' to {output_path}")
            results[f"{data_file.name}_{label}"] = df

    return results


def process_all_raw_zips() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Scan RAW_DIR for all .zip files and process them.

    Returns
    -------
    dict
        Mapping from zip_name -> {section_key -> DataFrame}
    """
    if not RAW_DIR.exists():
        print(f"No RAW_DIR at {RAW_DIR}, nothing to process.")
        return {}

    zip_files = sorted(RAW_DIR.glob("*.zip"))
    if not zip_files:
        print(f"No ZIP files found in {RAW_DIR}.")
        return {}

    print("=" * 70)
    print("Processing Fama–French ZIPs for FIN41360")
    print("=" * 70)

    all_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for zf in zip_files:
        results = _process_zip_file(zf)
        if results:
            all_results[zf.name] = results

    return all_results


def load_industry_30_monthly(start: str = "1980-01", end: str = "2025-12") -> pd.DataFrame:
    """
    Load 30-industry value-weighted monthly returns from processed CSVs.

    Returns
    -------
    DataFrame
        Gross returns (R = 1 + r) for the 30 industries, indexed by month.
    """
    # EXPLAIN: The processed file name for the value-weighted monthly
    # 30-industry series currently looks like:
    #   30_Industry_Portfolios_30_Industry_Portfolios_Average_Value_Weighted_Returns___Monthly.csv
    # so we match on "30_Industry_Portfolios" + "Average_Value_Weighted" + "Monthly".
    pattern = "*30_Industry_Portfolios*Average_Value_Weighted*Monthly*.csv"
    matches = list(PROCESSED_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No processed 30-industry monthly value-weighted file found in {PROCESSED_DIR} "
            f"matching pattern {pattern}. Have you run the download and processing steps?"
        )

    # If multiple matches exist, take the first; users can refine the pattern later if needed.
    csv_path = matches[0]
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Filter date range and ensure monthly frequency.
    df = df.sort_index()
    df = df.loc[start:end]
    df.index = df.index.to_period("M").to_timestamp("M")
    return df


def load_ff3_monthly(start: str = "1980-01", end: str = "2025-12") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Fama–French 3-factor monthly data (factors in excess returns, RF in gross form).

    Note: We source these from the 5-factor (2x3) monthly file, which contains
    the 3-factor set as a subset of columns. This avoids needing a separate
    download for the 3-factor file and is consistent with the assignment needs.

    Returns
    -------
    (factors, rf)
        factors : DataFrame with columns ['Mkt-RF', 'SMB', 'HML']
        rf : Series with gross risk-free returns (R_f = 1 + r_f)
    """
    # EXPLAIN: We use the same processed FF5 monthly file as in `load_ff5_monthly`
    # and simply select the 3-factor subset (Mkt-RF, SMB, HML).
    pattern = "*5_Factors_2x3*Value_Weighted*Monthly*.csv"
    matches = list(PROCESSED_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No processed FF3 monthly factors file found in {PROCESSED_DIR} "
            f"matching pattern {pattern}. Have you run the download and processing steps?"
        )

    csv_path = matches[0]
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.loc[start:end]
    df.index = df.index.to_period("M").to_timestamp("M")

    rf = df["RF"].copy()
    factors = df[["Mkt-RF", "SMB", "HML"]].copy()
    return factors, rf


def load_ff5_monthly(start: str = "1980-01", end: str = "2025-12") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Fama–French 5-factor (2x3) monthly data (factors in excess returns, RF in gross form).

    Returns
    -------
    (factors, rf)
        factors : DataFrame with columns ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        rf : Series with gross risk-free returns (R_f = 1 + r_f)
    """
    pattern = "*5_Factors_2x3*Monthly*.csv"
    matches = list(PROCESSED_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No processed FF5 monthly factors file found in {PROCESSED_DIR} "
            f"matching pattern {pattern}. Have you run the download and processing steps?"
        )

    csv_path = matches[0]
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.loc[start:end]
    df.index = df.index.to_period("M").to_timestamp("M")

    rf = df["RF"].copy()
    factors = df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].copy()
    return factors, rf

