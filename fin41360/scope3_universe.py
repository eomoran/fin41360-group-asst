"""
Build a cached, reproducible Scope 3 stock candidate universe.

This module is intentionally separate from analysis code so that:
- data acquisition can be re-run independently,
- downloads are cached to avoid repeated API hits/rate limits,
- candidate construction is deterministic and auditable.
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from .config import PROCESSED_DIR, RAW_DIR
from .scope3_mapping import FF30_INDUSTRIES
from .stock_data import INDUSTRY_TICKERS


SICCODES30_URL = "https://mba.tuck.dartmouth.edu/pages/Faculty/ken.french/ftp/Siccodes30.zip"
SP500_CSV_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_FMT = "https://data.sec.gov/submissions/CIK{cik}.json"


def _stage(msg: str) -> None:
    print(f"[Scope3] {msg}", flush=True)


def _clean_ticker(t: str) -> str:
    return str(t).upper().strip().replace(".", "-")


def _requests_session(user_agent: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    ua = user_agent or "fin41360-scope3-assignment/1.0 (academic use; contact: team@example.com)"
    s.headers.update({"User-Agent": ua, "Accept": "application/json,text/plain,*/*"})
    return s


def ensure_siccodes30_zip(raw_dir: Path = RAW_DIR, refresh: bool = False) -> Path:
    target = raw_dir / "Siccodes30.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not refresh:
        _stage(f"SIC definitions cache hit: {target}")
        return target

    _stage(f"Downloading SIC definitions from Ken French: {SICCODES30_URL}")
    resp = requests.get(SICCODES30_URL, timeout=60)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    _stage(f"Saved SIC definitions zip: {target}")
    return target


def parse_siccodes30_zip_to_ranges_csv(
    zip_path: str | Path,
    out_csv: str | Path,
) -> Path:
    """
    Parse Ken French Siccodes30 zip text into normalized FF30 SIC ranges CSV.

    The source format is semi-structured text. We parse it defensively:
    - detect FF30 section headers by known FF30 labels,
    - extract 3/4-digit SIC ranges and singleton SIC codes.
    """
    zpath = Path(zip_path)
    out = Path(out_csv)
    _stage(f"Parsing SIC definitions from zip: {zpath}")
    with zipfile.ZipFile(zpath, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"No files found inside zip: {zpath}")
        raw = zf.read(names[0]).decode("latin-1", errors="ignore")

    header_pattern = re.compile(
        r"^(?:\d+\s+)?(" + "|".join(re.escape(x) for x in FF30_INDUSTRIES) + r")\b",
        re.IGNORECASE,
    )
    range_pattern = re.compile(r"(?<!\d)(\d{3,4})\s*-\s*(\d{3,4})(?!\d)")
    singleton_pattern = re.compile(r"(?<![\d-])(\d{3,4})(?![\d-])")

    rows: list[dict[str, object]] = []
    current_ind: Optional[str] = None
    for raw_line in raw.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue

        m = header_pattern.match(line)
        if m:
            token = m.group(1)
            # Keep canonical FF30 capitalization from constant list.
            current_ind = next(x for x in FF30_INDUSTRIES if x.lower() == token.lower())

        if current_ind is None:
            continue

        consumed_spans = []
        for rm in range_pattern.finditer(line):
            a, b = int(rm.group(1)), int(rm.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            rows.append(
                {
                    "ff30_industry": current_ind,
                    "sic_start": lo,
                    "sic_end": hi,
                    "source": "Ken French Siccodes30",
                }
            )
            consumed_spans.append((rm.start(), rm.end()))

        # Some specs include singleton SIC codes. Capture them too.
        # Skip if the singleton token is part of an already-captured range span.
        for sm in singleton_pattern.finditer(line):
            pos = (sm.start(), sm.end())
            if any(a <= pos[0] and pos[1] <= b for a, b in consumed_spans):
                continue
            code = int(sm.group(1))
            rows.append(
                {
                    "ff30_industry": current_ind,
                    "sic_start": code,
                    "sic_end": code,
                    "source": "Ken French Siccodes30",
                }
            )

    if not rows:
        raise ValueError("Failed to parse any SIC rows from Siccodes30 zip.")

    df = pd.DataFrame(rows).drop_duplicates()
    parsed_inds = set(df["ff30_industry"])
    missing_inds = [x for x in FF30_INDUSTRIES if x not in parsed_inds]
    if missing_inds:
        raise ValueError(
            "Siccodes30 parse incomplete; missing FF30 industries: "
            + ", ".join(missing_inds)
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["ff30_industry", "sic_start", "sic_end"]).to_csv(out, index=False)
    _stage(f"Wrote parsed SIC ranges CSV ({len(df)} rows): {out}")
    return out


def ensure_ff30_sic_ranges_csv(
    out_csv: Path = PROCESSED_DIR / "ff30_sic_ranges.csv",
    refresh: bool = False,
) -> Path:
    if out_csv.exists() and not refresh:
        _stage(f"FF30 SIC ranges cache hit: {out_csv}")
        return out_csv
    zip_path = ensure_siccodes30_zip(refresh=refresh)
    return parse_siccodes30_zip_to_ranges_csv(zip_path=zip_path, out_csv=out_csv)


def _download_sp500_tickers(
    cache_csv: Path,
    refresh: bool = False,
) -> list[str]:
    if cache_csv.exists() and not refresh:
        _stage(f"S&P500 ticker cache hit: {cache_csv}")
        cached = pd.read_csv(cache_csv)
        if "ticker" in cached.columns:
            return [_clean_ticker(x) for x in cached["ticker"].dropna().astype(str)]

    _stage(f"Downloading S&P500 constituents: {SP500_CSV_URL}")
    df = pd.read_csv(SP500_CSV_URL)
    if "Symbol" not in df.columns:
        raise RuntimeError("S&P 500 source CSV missing Symbol column.")
    tickers = sorted({_clean_ticker(t) for t in df["Symbol"].dropna().astype(str)})
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers, "source": "datahub_sp500"}).to_csv(cache_csv, index=False)
    _stage(f"Saved S&P500 ticker cache ({len(tickers)} symbols): {cache_csv}")
    return tickers


def _load_sec_ticker_to_cik(
    session: requests.Session,
    cache_json: Path,
    refresh: bool = False,
) -> dict[str, str]:
    if cache_json.exists() and not refresh:
        _stage(f"SEC ticker->CIK cache hit: {cache_json}")
        payload = json.loads(cache_json.read_text())
    else:
        _stage(f"Downloading SEC ticker->CIK map: {SEC_TICKER_CIK_URL}")
        resp = session.get(SEC_TICKER_CIK_URL, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        cache_json.write_text(json.dumps(payload))
        _stage(f"Saved SEC ticker->CIK cache: {cache_json}")

    # SEC JSON has integer keys: {"0": {...}, "1": {...}, ...}
    out: dict[str, str] = {}
    for _, row in payload.items():
        ticker = _clean_ticker(row.get("ticker", ""))
        cik = str(int(row.get("cik_str"))).zfill(10) if row.get("cik_str") is not None else ""
        if ticker and cik:
            out[ticker] = cik
    _stage(f"Loaded SEC ticker->CIK entries: {len(out)}")
    return out


def _load_sec_sic_for_cik(
    cik_10: str,
    session: requests.Session,
    cache_dir: Path,
    refresh: bool = False,
    sec_sleep_s: float = 0.2,
) -> tuple[Optional[int], Optional[str]]:
    path = cache_dir / f"CIK{cik_10}.json"
    if path.exists() and not refresh:
        payload = json.loads(path.read_text())
    else:
        url = SEC_SUBMISSIONS_URL_FMT.format(cik=cik_10)
        resp = session.get(url, timeout=60)
        if resp.status_code == 404:
            return None, None
        resp.raise_for_status()
        payload = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        time.sleep(sec_sleep_s)

    sic = payload.get("sic")
    sic_desc = payload.get("sicDescription")
    try:
        sic_int = int(sic) if sic is not None else None
    except Exception:
        sic_int = None
    return sic_int, sic_desc


def _compute_yf_metadata(
    tickers: Iterable[str],
    start: str,
    end: str,
    yf_cache_csv: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Build price/volume-derived metadata needed by the selection rules.
    """
    if yf_cache_csv.exists() and not refresh:
        _stage(f"yfinance metadata cache hit: {yf_cache_csv}")
        return pd.read_csv(yf_cache_csv)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance required. Install with: pip install yfinance") from exc

    ticker_list = sorted({_clean_ticker(x) for x in tickers})
    total = len(ticker_list)
    _stage(f"Fetching yfinance metadata for {total} tickers...")

    rows = []
    # One call per ticker is slower but easier to make robust and cacheable.
    for i, t in enumerate(ticker_list, start=1):
        if i == 1 or i % 25 == 0 or i == total:
            _stage(f"yfinance progress: {i}/{total} tickers")
        try:
            hist = yf.download(
                t,
                start=(pd.Timestamp(start) - pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
                end=pd.Timestamp(end).strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
            )
            if hist.empty:
                rows.append(
                    {
                        "ticker": t,
                        "market_cap_usd": None,
                        "median_dollar_volume_usd": None,
                        "first_return_month": None,
                        "last_return_month": None,
                    }
                )
                continue

            close_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
            vol_col = "Volume"
            close = hist[close_col].astype(float)
            vol = hist[vol_col].astype(float) if vol_col in hist.columns else pd.Series(index=close.index, dtype=float)
            dollar_volume = (close * vol).dropna()
            median_dollar_vol = float(dollar_volume.median()) if not dollar_volume.empty else None

            monthly = close.resample("ME").last().dropna()
            ret = monthly.pct_change().dropna()
            first_month = ret.index.min().to_period("M").strftime("%Y-%m") if not ret.empty else None
            last_month = ret.index.max().to_period("M").strftime("%Y-%m") if not ret.empty else None

            market_cap = None
            try:
                info = yf.Ticker(t).fast_info
                mc = getattr(info, "market_cap", None)
                market_cap = float(mc) if mc is not None else None
            except Exception:
                market_cap = None

            rows.append(
                {
                    "ticker": t,
                    "market_cap_usd": market_cap,
                    "median_dollar_volume_usd": median_dollar_vol,
                    "first_return_month": first_month,
                    "last_return_month": last_month,
                }
            )
        except Exception:
            rows.append(
                {
                    "ticker": t,
                    "market_cap_usd": None,
                    "median_dollar_volume_usd": None,
                    "first_return_month": None,
                    "last_return_month": None,
                }
            )

    out = pd.DataFrame(rows)
    yf_cache_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(yf_cache_csv, index=False)
    _stage(f"Saved yfinance candidate metadata ({len(out)} rows): {yf_cache_csv}")
    return out


def build_scope3_candidates_csv(
    out_csv: Path = PROCESSED_DIR / "scope3_stock_candidates.csv",
    start: str = "1990-01",
    end: str = "2025-12",
    include_sp500: bool = True,
    refresh: bool = False,
    sec_user_agent: Optional[str] = None,
) -> Path:
    """
    Build candidate stock universe with SIC + liquidity/history metadata.
    """
    if out_csv.exists() and not refresh:
        _stage(f"Candidate universe cache hit: {out_csv}")
        return out_csv

    _stage("Stage 1/4: Building candidate ticker universe")
    candidate_tickers = set(INDUSTRY_TICKERS.values())
    _stage(f"Baseline ticker count: {len(candidate_tickers)}")

    raw_sp500_cache = RAW_DIR / "scope3" / "sp500_tickers.csv"
    if include_sp500:
        try:
            sp500 = _download_sp500_tickers(raw_sp500_cache, refresh=refresh)
            candidate_tickers.update(sp500)
            _stage(f"Added S&P500 symbols: {len(sp500)}")
        except Exception:
            # If S&P 500 scrape fails, still continue with local baseline list.
            _stage("Warning: failed to load S&P500 list; continuing with baseline list only.")
    else:
        _stage("S&P500 augmentation disabled (--no-sp500).")

    candidate_tickers = {_clean_ticker(x) for x in candidate_tickers if str(x).strip()}
    _stage(f"Total unique candidate tickers: {len(candidate_tickers)}")

    _stage("Stage 2/4: Loading SEC ticker->CIK mapping")
    session = _requests_session(user_agent=sec_user_agent)
    ticker_map = _load_sec_ticker_to_cik(
        session=session,
        cache_json=RAW_DIR / "scope3" / "sec_company_tickers.json",
        refresh=refresh,
    )
    sec_cache_dir = RAW_DIR / "scope3" / "sec_submissions"

    _stage("Stage 3/4: Resolving SEC SIC per candidate")
    sic_rows = []
    sorted_tickers = sorted(candidate_tickers)
    total = len(sorted_tickers)
    cache_hits = 0
    missing_cik = 0
    downloaded = 0
    for i, t in enumerate(sorted_tickers, start=1):
        if i == 1 or i % 25 == 0 or i == total:
            _stage(
                "SEC SIC progress: "
                f"{i}/{total} (cache_hits={cache_hits}, downloaded={downloaded}, missing_cik={missing_cik})"
            )
        cik = ticker_map.get(t)
        if cik is None:
            missing_cik += 1
            sic_rows.append({"ticker": t, "sic": None, "sic_description": None, "sec_cik": None})
            continue
        sub_path = sec_cache_dir / f"CIK{cik}.json"
        was_cached = sub_path.exists() and not refresh
        sic, sic_desc = _load_sec_sic_for_cik(
            cik_10=cik,
            session=session,
            cache_dir=sec_cache_dir,
            refresh=refresh,
        )
        if was_cached:
            cache_hits += 1
        else:
            downloaded += 1
        sic_rows.append({"ticker": t, "sic": sic, "sic_description": sic_desc, "sec_cik": cik})
    sec_df = pd.DataFrame(sic_rows)
    _stage(
        "SEC SIC stage done: "
        f"rows={len(sec_df)}, with_sic={int(sec_df['sic'].notna().sum())}, "
        f"missing_sic={int(sec_df['sic'].isna().sum())}"
    )

    _stage("Stage 4/4: Building market/liquidity/history metadata (yfinance)")
    yf_df = _compute_yf_metadata(
        tickers=sorted(candidate_tickers),
        start=start,
        end=end,
        yf_cache_csv=RAW_DIR / "scope3" / "yfinance_candidate_metadata.csv",
        refresh=refresh,
    )

    out = sec_df.merge(yf_df, on="ticker", how="outer")
    out["notes"] = (
        "Baseline local list + optional S&P500 augmentation; SIC from SEC submissions; "
        "market/liquidity/history from yfinance."
    )
    out = out.sort_values("ticker").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    _stage(f"Saved candidate universe ({len(out)} rows): {out_csv}")
    return out_csv
