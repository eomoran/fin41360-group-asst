"""
Tools for discovering and downloading the Fama–French datasets needed
for FIN41360 Assignment 1.

This module:
- Scrapes the Kenneth French data library page to discover available ZIP files.
- Selects the appropriate ZIPs for:
  * 30 Industry Portfolios (value-weighted, monthly)
  * Fama–French 3-Factor model (monthly, includes RF)
  * Fama–French 5-Factor model (monthly, includes RF)
- Downloads those ZIPs into `fin41360_data/raw/`.

The logic is adapted from the FIN50040 assignment utilities but kept
focused on the specific files we need here.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .config import RAW_DIR


BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
DATA_LIBRARY_URL = BASE_URL + "data_library.html"


class _LinkParser(HTMLParser):
    """Internal helper: parse HTML to extract links to .zip files."""

    def __init__(self) -> None:
        super().__init__()
        self.file_links: List[str] = []

    def handle_starttag(self, tag, attrs) -> None:  # type: ignore[override]
        if tag != "a":
            return
        for attr, value in attrs:
            if attr == "href" and value.endswith(".zip"):
                self.file_links.append(value)


@dataclass
class AvailableFiles:
    """Container for discovered Fama–French ZIP files."""

    filenames: List[str]
    url_map: Dict[str, str]


def discover_available_files() -> AvailableFiles:
    """
    Scrape the Fama–French data library page and return the list of
    available ZIP filenames and their fully-qualified URLs.

    Returns
    -------
    AvailableFiles
        Filenames and a mapping from filename -> absolute URL.
    """
    print(f"Discovering Fama–French ZIP files from {DATA_LIBRARY_URL} ...")

    response = requests.get(DATA_LIBRARY_URL, timeout=30)
    response.raise_for_status()

    parser = _LinkParser()
    parser.feed(response.text)

    filenames: List[str] = []
    url_map: Dict[str, str] = {}

    for link in parser.file_links:
        if link.startswith("http"):
            full_url = link
        elif link.startswith("/"):
            full_url = "https://mba.tuck.dartmouth.edu" + link
        elif link.startswith("ftp/"):
            full_url = BASE_URL + link
        else:
            full_url = BASE_URL + "ftp/" + link

        filename = full_url.split("/")[-1]
        filenames.append(filename)
        url_map[filename] = full_url

    filenames = sorted(set(filenames))
    print(f"  Found {len(filenames)} ZIP files.")
    return AvailableFiles(filenames=filenames, url_map=url_map)


def _find_file_by_keywords(
    available: AvailableFiles,
    keywords: List[str],
    exclude_keywords: Optional[List[str]] = None,
    prefer_csv: bool = True,
) -> Optional[str]:
    """
    Find a filename that contains all provided keywords (case-insensitive).

    Parameters
    ----------
    available : AvailableFiles
        Discovered filenames and URL map.
    keywords : list of str
        All keywords must appear in the filename.
    exclude_keywords : list of str, optional
        If provided, none of these keywords may appear in the filename.
    prefer_csv : bool, default True
        If True, prefer filenames that contain '_CSV'.
    """
    exclude_keywords = exclude_keywords or []
    matches: List[str] = []
    for filename in available.filenames:
        name_upper = filename.upper()
        if all(kw.upper() in name_upper for kw in keywords):
            if any(kw.upper() in name_upper for kw in exclude_keywords):
                continue
            matches.append(filename)

    if not matches:
        return None

    if prefer_csv:
        csv_matches = [m for m in matches if "_CSV" in m.upper()]
        if csv_matches:
            return csv_matches[0]

    return matches[0]


def _download_zip(url: str, dest: Path, description: str) -> Path:
    """
    Download a ZIP file from `url` to `dest`.

    Parameters
    ----------
    url : str
        Fully-qualified URL of the ZIP file.
    dest : Path
        Destination path (including filename).
    description : str
        Short description for logging.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  ✓ {description} already downloaded at {dest}")
        return dest

    print(f"  Downloading {description} ...")
    print(f"    URL: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    if not resp.content:
        raise RuntimeError(f"Empty response when downloading {description} from {url}")

    dest.write_bytes(resp.content)
    size = dest.stat().st_size
    print(f"  ✓ Saved {description} ({size:,} bytes) to {dest}")
    return dest


def download_industry_30_zip(available: Optional[AvailableFiles] = None) -> Path:
    """
    Download the 30 Industry Portfolios (value-weighted) ZIP.

    Returns
    -------
    Path
        Path to the downloaded ZIP file.
    """
    if available is None:
        available = discover_available_files()

    # Keywords chosen to match the 30-industry value-weighted monthly file.
    # EXPLAIN: In practice the file is typically named something like
    # `30_Industry_Portfolios_CSV.zip` or
    # `30_Industry_Portfolios_Value_Weighted_CSV.zip`. We do NOT insist on
    # "Value" in the name to avoid missing the CSV variant if the naming
    # convention changes slightly.
    filename = _find_file_by_keywords(
        available,
        keywords=["30", "Industry", "Portfolios"],
        prefer_csv=True,
    )
    if filename is None:
        raise RuntimeError("Could not find 30 Industry Portfolios ZIP on the data library page.")

    url = available.url_map[filename]
    dest = RAW_DIR / filename
    return _download_zip(url, dest, description="30 Industry Portfolios (value-weighted)")


def download_ff_factor_zips(available: Optional[AvailableFiles] = None) -> Tuple[Path, Path]:
    """
    Download the Fama–French 3-factor and 5-factor (2x3) monthly ZIPs.

    Returns
    -------
    (Path, Path)
        Paths to the 3-factor ZIP and 5-factor ZIP, respectively.
    """
    if available is None:
        available = discover_available_files()

    # 3-factor model: "F-F_Research_Data_Factors" (exclude 5-factor variants).
    ff3_name = _find_file_by_keywords(
        available,
        keywords=["F-F", "Research", "Data", "Factors"],
        exclude_keywords=["5_Factors", "2x3"],
        prefer_csv=True,
    )
    if ff3_name is None:
        raise RuntimeError("Could not find Fama–French 3-factor ZIP.")

    # 5-factor model: look for "5_Factors_2x3".
    ff5_name = _find_file_by_keywords(
        available,
        keywords=["5", "Factors", "2x3"],
        prefer_csv=True,
    )
    if ff5_name is None:
        raise RuntimeError("Could not find Fama–French 5-factor (2x3) ZIP.")
    if ff5_name == ff3_name:
        raise RuntimeError(
            f"FF3 and FF5 resolved to the same file ({ff3_name}). "
            "Check keyword filters in download_ff_factor_zips()."
        )

    ff3_zip = _download_zip(available.url_map[ff3_name], RAW_DIR / ff3_name, "Fama–French 3-Factor model")
    ff5_zip = _download_zip(available.url_map[ff5_name], RAW_DIR / ff5_name, "Fama–French 5-Factor model")

    return ff3_zip, ff5_zip


def download_all_core_french_zips() -> Tuple[Path, Path, Path]:
    """
    Convenience function: download all core Fama–French ZIPs required for
    Assignment 1 question 1 (industries + FF3 + FF5).

    Returns
    -------
    (Path, Path, Path)
        (industry_30_zip, ff3_zip, ff5_zip)
    """
    available = discover_available_files()
    ind_zip = download_industry_30_zip(available)
    ff3_zip, ff5_zip = download_ff_factor_zips(available)
    return ind_zip, ff3_zip, ff5_zip
