"""
Configuration for FIN41360 Assignment 1 utilities.

Centralises paths so that all data for this assignment lives under a single
`fin41360_data/` directory at the project root, separate from other modules.
"""

from pathlib import Path

# Project root = parent of this `fin41360` package
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Root directory for all FIN41360-specific data
DATA_ROOT = PROJECT_ROOT / "fin41360_data"
RAW_DIR = DATA_ROOT / "raw"          # Downloaded ZIP files
PROCESSED_DIR = DATA_ROOT / "processed"  # Parsed CSVs with clean structure

# EXPLAIN: Actual directory creation is done by the download/processing code,
# not at import time, to keep module side effects minimal.

