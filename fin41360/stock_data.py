"""
Load and align monthly returns for 30 individual stocks (one per Fama–French industry).

Used for Scope 3 / Question 3: repeat the MV frontier analysis on 30 stocks and
compare to the 30-industry frontiers. Requires a common sample period where
both industry and stock data are available.

Dependencies
-----------
- yfinance : pip install yfinance
  Used to fetch adjusted close prices; we then compute monthly gross returns.

Design choices
--------------
- We return gross returns (R = 1 + r) so that the same pipeline as industry
  data (compute_moments_from_gross, etc.) can be used without branching.
- One representative ticker per industry: chosen for liquidity and long history
  where possible. You can override INDUSTRY_TICKERS if you prefer different
  names (e.g. to match a specific screen or data vendor).
- Optional cache: save/load from PROCESSED_DIR to avoid repeated API calls and
  to make the notebook reproducible. Cache key is based on start/end and
  ticker list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .config import PROCESSED_DIR

# One representative stock per Fama–French 30 industry (Yahoo Finance tickers).
# Chosen for large cap, liquid, and long history where possible. Alternatives
# (e.g. different sector leaders) can be substituted; the assignment only
# requires one stock per industry and a consistent sample period.
#
# TODO: find tickers which provide a longer history so the common sample period
# with industry data (e.g. from 1980) is longer; current choice can yield only
# ~2019–2025 overlap, which is short for stable frontier/Sharpe estimation.
INDUSTRY_TICKERS = {
    "Food": "KO",
    "Beer": "TAP",
    "Smoke": "MO",
    "Games": "EA",
    "Books": "SCHL",
    "Hshld": "PG",
    "Clths": "NKE",
    "Hlth": "JNJ",
    "Chems": "DOW",
    "Txtls": "CRI",
    "Cnstr": "CAT",
    "Steel": "NUE",
    "FabPr": "ETN",
    "ElcEq": "AAPL",
    "Autos": "GM",
    "Carry": "DAL",
    "Mines": "FCX",
    "Coal": "BTU",
    "Oil": "XOM",
    "Util": "NEE",
    "Telcm": "VZ",
    "Servs": "ADP",
    "BusEq": "HPQ",
    "Paper": "IP",
    "Trans": "UPS",
    "Whlsl": "WMT",
    "Rtail": "HD",
    "Meals": "MCD",
    "Fin": "JPM",
    "Other": "MMM",
}

LEGACY_STOCK_GROSS_FILE = "stock_30_monthly_gross.csv"
BALANCED_STOCK_NET_FILE = "clean_30_stocks_monthly_returns_balanced_1990_2025.csv"
SCOPE3_SELECTED_GROSS_FILE = "scope3_selected_30_stock_monthly_gross.csv"
SCOPE3_SELECTED_MAPPING_FILE = "scope3_selected_30_stocks_ff30.csv"

# Proposed/manual crosswalk for the balanced long-history stock set.
# IMPORTANT: this is not an official Ken French / CRSP mapping for the FF30
# industry portfolios. It is a project crosswalk for "one stock per industry"
# style analysis and should be reviewed before final submission.
BALANCED_STOCK_FF30_CROSSWALK = {
    "KO": "Food",
    "MO": "Smoke",
    "DIS": "Games",
    "PFE": "Hlth",
    "DD": "Chems",
    "NKE": "Clths",
    "LEN": "Cnstr",
    "NUE": "Steel",
    "HON": "FabPr",
    "GE": "ElcEq",
    "F": "Autos",
    "BA": "Carry",
    "NEM": "Mines",
    "XOM": "Oil",
    "DUK": "Util",
    "T": "Telcm",
    "ADP": "Servs",
    "IBM": "BusEq",
    "FRT": "Paper",
    "FDX": "Trans",
    "COST": "Whlsl",
    "WMT": "Rtail",
    "MMM": "Other",
    "JPM": "Fin",
    # Lower-confidence / no direct obvious FF30 analogue in the chosen balanced set:
    "AIG": "Beer",
    "AXP": "Books",
    "CAT": "Txtls",
    "INTC": "Coal",
    "UNH": "Hshld",
    "MSFT": "Meals",
}


def _ff30_industry_order() -> list[str]:
    return list(INDUSTRY_TICKERS.keys())


def _load_legacy_cached_stock_gross(start: str, end: str) -> pd.DataFrame:
    cache_path = PROCESSED_DIR / LEGACY_STOCK_GROSS_FILE
    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    df = df.loc[start:end]
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.reindex(columns=_ff30_industry_order())
    df.attrs["stock_source"] = "legacy_cached_gross"
    df.attrs["stock_source_file"] = str(cache_path)
    return df


def _load_balanced_stock_net_as_gross(start: str, end: str) -> pd.DataFrame:
    path = PROCESSED_DIR / BALANCED_STOCK_NET_FILE
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # File stores monthly net returns with month-start-like stamps (e.g., 1990-01-01)
    # Convert to month-end and gross returns for compatibility with the frontier pipeline.
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp("M")
    df = 1.0 + df
    df = df.sort_index().loc[start:end]
    df.attrs["stock_source"] = "balanced_clean_net_to_gross"
    df.attrs["stock_source_file"] = str(path)
    return df


def _load_scope3_selected_cached_gross(start: str, end: str) -> pd.DataFrame:
    path = PROCESSED_DIR / SCOPE3_SELECTED_GROSS_FILE
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index().loc[start:end]
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.reindex(columns=_ff30_industry_order())
    df.attrs["stock_source"] = "scope3_selected_cached_gross"
    df.attrs["stock_source_file"] = str(path)
    return df


def build_scope3_selected_stock_returns(
    start: str = "1980-01",
    end: str = "2025-12",
    mapping_file: str | Path | None = None,
    out_file: str | Path | None = None,
) -> Path:
    """
    Build gross monthly returns for the SIC-selected Scope 3 30-stock set.

    Uses `scope3_selected_30_stocks_ff30.csv` (or provided mapping file) and
    downloads monthly prices from yfinance. Output columns are FF30 industry
    labels in canonical order so industry-vs-stock comparisons are aligned.
    """
    mpath = Path(mapping_file) if mapping_file is not None else (PROCESSED_DIR / SCOPE3_SELECTED_MAPPING_FILE)
    if not mpath.exists():
        raise FileNotFoundError(f"Selected mapping file not found: {mpath}")

    m = pd.read_csv(mpath)
    req = {"ff30_industry", "ticker", "selection_status"}
    if not req.issubset(m.columns):
        raise ValueError(f"Mapping file must contain columns: {sorted(req)}")

    m = m[m["selection_status"].astype(str).str.startswith("selected")].copy()
    m = m[m["ticker"].notna()].copy()
    m["ticker"] = m["ticker"].astype(str).str.upper().str.strip()
    m["ff30_industry"] = m["ff30_industry"].astype(str).str.strip()

    if len(m) < 1:
        raise ValueError("No selected rows found in mapping file.")

    out_path = Path(out_file) if out_file is not None else (PROCESSED_DIR / SCOPE3_SELECTED_GROSS_FILE)

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required for stock data. Install with: pip install yfinance")

    start_d = pd.Timestamp(start) - pd.offsets.MonthBegin(1)
    end_d = pd.Timestamp(end)

    out = {}
    for _, row in m.iterrows():
        ind = row["ff30_industry"]
        ticker = row["ticker"]
        try:
            hist = yf.download(
                ticker,
                start=start_d,
                end=end_d,
                progress=False,
                auto_adjust=True,
            )
            if hist.empty:
                out[ind] = pd.Series(dtype=float)
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                close = hist["Close"].iloc[:, 0]
            else:
                close = hist["Close"] if "Close" in hist.columns else hist["Adj Close"]
            close = close.sort_index()
            monthly = close.resample("ME").last().dropna()
            ret = monthly.pct_change().dropna()
            out[ind] = 1.0 + ret
        except Exception as e:
            out[ind] = pd.Series(dtype=float)
            print(f"  Warning: failed to load {ticker} ({ind}): {e}")

    df = pd.DataFrame(out)
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.sort_index().loc[start:end]
    df = df.reindex(columns=_ff30_industry_order())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return out_path


def stock_mapping_tables() -> dict[str, pd.DataFrame]:
    """
    Return explicit stock->industry mapping tables for the stock CSV datasets.

    `legacy_cached_gross` is the exact crosswalk used by `INDUSTRY_TICKERS`.
    `balanced_clean_proposed` is a project/manual proposed crosswalk and should
    be reviewed for final-report accuracy (especially "Other" and weak matches).
    """
    legacy_rows = []
    for industry, ticker in INDUSTRY_TICKERS.items():
        legacy_rows.append(
            {
                "dataset": "legacy_cached_gross",
                "dataset_file": LEGACY_STOCK_GROSS_FILE,
                "ticker": ticker,
                "ff30_industry": industry,
                "mapping_status": "manual_project_selection",
                "confidence": "high",
                "official_ff_crsp": False,
                "notes": "Original project one-stock-per-industry ticker list (manual).",
            }
        )

    balanced_rows = []
    low_conf = {"AIG", "AXP", "CAT", "INTC", "UNH", "MSFT"}
    for ticker in sorted(BALANCED_STOCK_FF30_CROSSWALK):
        industry = BALANCED_STOCK_FF30_CROSSWALK[ticker]
        confidence = "low" if ticker in low_conf else "medium"
        note = (
            "Proposed approximate FF30 crosswalk; no obvious direct match in balanced set."
            if ticker in low_conf
            else "Proposed project crosswalk for balanced long-history stock set."
        )
        balanced_rows.append(
            {
                "dataset": "balanced_clean_proposed",
                "dataset_file": BALANCED_STOCK_NET_FILE,
                "ticker": ticker,
                "ff30_industry": industry,
                "mapping_status": "proposed_manual_crosswalk",
                "confidence": confidence,
                "official_ff_crsp": False,
                "notes": note,
            }
        )

    return {
        "legacy_cached_gross": pd.DataFrame(legacy_rows).sort_values(["ff30_industry", "ticker"]).reset_index(drop=True),
        "balanced_clean_proposed": pd.DataFrame(balanced_rows).sort_values(["ff30_industry", "ticker"]).reset_index(drop=True),
    }


def write_stock_mapping_tables(out_dir: Optional[str | Path] = None) -> dict[str, Path]:
    """
    Write stock mapping tables to CSV files (default: `PROCESSED_DIR`).
    """
    target = Path(out_dir) if out_dir is not None else PROCESSED_DIR
    target.mkdir(parents=True, exist_ok=True)
    tables = stock_mapping_tables()
    out = {}
    for name, df in tables.items():
        path = target / f"{name}_stock_to_ff30_mapping.csv"
        df.to_csv(path, index=False)
        out[name] = path
    return out


def load_stock_returns_monthly(
    start: str = "1980-01",
    end: str = "2025-12",
    use_cache: bool = True,
    source: str = "auto",
) -> pd.DataFrame:
    """
    Load monthly gross returns for the 30 stocks (one per industry).

    Parameters
    ----------
    start, end : str
        Date range in 'YYYY-MM' or 'YYYY-MM-DD'. Requested range may be
        truncated if some tickers have shorter history.
    use_cache : bool, default True
        If True, read from or write to a CSV in PROCESSED_DIR to avoid
        repeated API calls.
    source : {'auto', 'scope3_selected', 'balanced', 'legacy', 'yfinance'}
        - 'auto' (default): prefer balanced long-history processed file if present,
          else legacy cached gross file if present, else fetch via yfinance.
          If a scope3-selected cached gross file exists, it is preferred first.
        - 'scope3_selected': load cached SIC-selected Scope 3 gross returns.
        - 'balanced': load balanced processed CSV (net returns) and convert to gross.
        - 'legacy': load legacy cached gross CSV (industry-labeled columns).
        - 'yfinance': always fetch and optionally refresh legacy cache.

    Returns
    -------
    DataFrame
        Index: month-end dates (DatetimeIndex).
        Columns: industry names (same order as French 30-industry data).
        Values: gross returns (R = 1 + r). May contain NaN for months
        where a ticker was not yet listed or data is missing.
    """
    cache_path = PROCESSED_DIR / LEGACY_STOCK_GROSS_FILE
    balanced_path = PROCESSED_DIR / BALANCED_STOCK_NET_FILE
    selected_path = PROCESSED_DIR / SCOPE3_SELECTED_GROSS_FILE

    source = source.lower()
    if source not in {"auto", "scope3_selected", "balanced", "legacy", "yfinance"}:
        raise ValueError("source must be one of {'auto', 'scope3_selected', 'balanced', 'legacy', 'yfinance'}")

    if source == "scope3_selected":
        if not selected_path.exists():
            raise FileNotFoundError(f"Scope3-selected stock gross cache not found: {selected_path}")
        return _load_scope3_selected_cached_gross(start=start, end=end)

    if source == "balanced":
        if not balanced_path.exists():
            raise FileNotFoundError(f"Balanced stock file not found: {balanced_path}")
        return _load_balanced_stock_net_as_gross(start=start, end=end)

    if source == "legacy":
        if not cache_path.exists():
            raise FileNotFoundError(f"Legacy stock gross cache not found: {cache_path}")
        return _load_legacy_cached_stock_gross(start=start, end=end)

    if source == "auto":
        if use_cache and selected_path.exists():
            return _load_scope3_selected_cached_gross(start=start, end=end)
        if use_cache and balanced_path.exists():
            return _load_balanced_stock_net_as_gross(start=start, end=end)
        if use_cache and cache_path.exists():
            return _load_legacy_cached_stock_gross(start=start, end=end)

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for stock data. Install with: pip install yfinance"
        )

    industries = list(INDUSTRY_TICKERS.keys())
    tickers = list(INDUSTRY_TICKERS.values())

    # yfinance expects start/end as dates; request a bit before start for return calc
    start_d = pd.Timestamp(start) - pd.offsets.MonthBegin(1)
    end_d = pd.Timestamp(end)

    out = {}
    for ind, ticker in zip(industries, tickers):
        try:
            hist = yf.download(
                ticker,
                start=start_d,
                end=end_d,
                progress=False,
                auto_adjust=True,
            )
            if hist.empty:
                out[ind] = pd.Series(dtype=float)
                continue
            # yfinance can return MultiIndex columns for single ticker in some versions
            if isinstance(hist.columns, pd.MultiIndex):
                close = hist["Close"].iloc[:, 0]
            else:
                close = hist["Close"] if "Close" in hist.columns else hist["Adj Close"]
            close = close.sort_index()
            # Monthly gross return: R_t = close_t / close_{t-1}
            monthly = close.resample("ME").last().dropna()
            ret = monthly.pct_change().dropna()
            gross = 1.0 + ret
            out[ind] = gross
        except Exception as e:
            # If one ticker fails, store empty series so we still get a column
            out[ind] = pd.Series(dtype=float)
            print(f"  Warning: failed to load {ticker} ({ind}): {e}")

    df = pd.DataFrame(out)
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.sort_index().loc[start:end]

    if use_cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
    df.attrs["stock_source"] = "yfinance_fetched_gross"
    df.attrs["stock_source_file"] = str(cache_path)

    return df


def get_common_sample_period(
    df_industries: pd.DataFrame,
    df_stocks: pd.DataFrame,
) -> Tuple[str, str]:
    """
    Get the start and end (as 'YYYY-MM') of the overlapping period where
    both industry and stock data have no missing values.

    Parameters
    ----------
    df_industries : DataFrame
        Industry returns, index = month-end dates.
    df_stocks : DataFrame
        Stock returns, index = month-end dates.

    Returns
    -------
    (start, end) : tuple of str
        Inclusive range 'YYYY-MM' for use in load_industry_30_monthly( start=..., end=... )
        and when slicing stock data so that both datasets are aligned.
    """
    # Align indices: keep only months present in both
    common_idx = df_industries.index.intersection(df_stocks.index).sort_values()
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between industry and stock data.")

    # Drop months with any NaN in either dataset
    ind_ok = df_industries.loc[common_idx].notna().all(axis=1)
    stk_ok = df_stocks.loc[common_idx].notna().all(axis=1)
    valid = common_idx[ind_ok & stk_ok]
    if len(valid) == 0:
        raise ValueError(
            "No month in the overlap has full data for both industries and stocks."
        )

    start = valid.min().strftime("%Y-%m")
    end = valid.max().strftime("%Y-%m")
    return start, end
