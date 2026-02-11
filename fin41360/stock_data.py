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


def load_stock_returns_monthly(
    start: str = "1980-01",
    end: str = "2025-12",
    use_cache: bool = True,
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

    Returns
    -------
    DataFrame
        Index: month-end dates (DatetimeIndex).
        Columns: industry names (same order as French 30-industry data).
        Values: gross returns (R = 1 + r). May contain NaN for months
        where a ticker was not yet listed or data is missing.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for stock data. Install with: pip install yfinance"
        )

    cache_name = "stock_30_monthly_gross.csv"
    cache_path = PROCESSED_DIR / cache_name

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df = df.loc[start:end]
        df.index = df.index.to_period("M").to_timestamp("M")
        return df

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
