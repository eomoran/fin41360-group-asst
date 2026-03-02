# FIN41360 Group Assignment 1 (Portfolio Choice & Performance Evaluation)

This repository contains the code and notebooks for the FIN41360 group assignment.

Current status:
- Scopes / Questions 2-7 are refactored into reusable workflow + plotting functions and wired into a thin report notebook.
- Scope 8 and Scope 9 are still pending (not yet integrated into the refactored report flow).

## Repo entrypoints

- Main notebook (refactored): `fin41360_report.ipynb`
- Scratch / legacy notebook: `fin41360_data_check.ipynb`
- Scope 8 exploratory notebook: `fin41360_scope8.ipynb`

## Clone the repo

SSH (recommended if you already have GitHub keys set up):

```bash
git clone git@github.com:eomoran/fin41360-group-asst.git
```

HTTPS also works, but GitHub password auth does not. Use a Personal Access Token (PAT) or `gh auth login`.

```bash
git clone https://github.com/eomoran/fin41360-group-asst.git
```

## Setup

From the repo root:

```bash
pip install -r requirements.txt
source ./env.workspace.sh
```

`env.workspace.sh` sets project-local cache directories (e.g., Matplotlib cache) to avoid sandbox/home-directory cache permission issues.

## Data setup

Fama-French data setup is now automatic on first notebook run:
- if required processed files are missing, loaders auto-download and process
  core datasets (30 industries, FF3, FF5),
- if files already exist, cached processed files are reused.

Manual pre-build is still available (optional):

```bash
python -m fin41360.setup_french_data
```

## Run the analysis notebook

Open and run:

- `fin41360_report.ipynb`

The notebook is designed to call descriptively named workflow functions (in `fin41360/workflows.py`) and plotting helpers (in `fin41360/plot_frontiers.py`), keeping notebook logic thin.
On a fresh machine, the first run may take longer because missing Fama-French
data is bootstrapped automatically.

## Scope 3 stock data note (important)

Scope 3 currently uses a custom balanced stock file when available:

- `fin41360_data/processed/clean_30_stocks_monthly_returns_balanced_1990_2025.csv`

This file is loaded automatically by `load_stock_returns_monthly(..., source="auto")`, converted from monthly net returns to gross returns, and then used in the frontier pipeline.

The stock-to-FF30-industry mapping for that file is a project crosswalk (not an official Ken French / CRSP one-stock mapping). Use the Scope 3 validation workflow below to produce an auditable SIC-based mapping before final submission.

After running `python -m fin41360.setup_scope3_data`, the file
`fin41360_data/processed/scope3_selected_30_stock_monthly_gross.csv` is built
from the SIC-selected mapping and becomes the first preference for
`load_stock_returns_monthly(..., source="auto")`.

`load_stock_returns_monthly(..., source="auto")` now also attempts to build
that file automatically when missing (then reuses the local cached CSV on
subsequent runs). Use `scope3_refresh=True` to force a rebuild/download.

To build a validated, auditable mapping from SIC to FF30 with deterministic
stock selection rules, use:

```bash
python -m fin41360.setup_scope3_mapping \
  --candidates fin41360_data/processed/scope3_stock_candidates.csv \
  --sic-ranges fin41360_data/processed/ff30_sic_ranges.csv
```

For a fresh machine / reproducible setup (with caching and automatic download
of candidate metadata + Siccodes30 definitions), use:

```bash
python -m fin41360.setup_scope3_data
```

By default this reuses existing processed files and raw caches to avoid
unnecessary downloads/rate limits. Use `--refresh` only when you want to
force a rebuild.
The command prints stage-by-stage progress (cache hits, download steps, SEC/yfinance progress).
Pass `--skip-selected-returns` to skip building the selected 30 return matrix.

Method details are documented in:
- `docs/scope3_ff30_mapping_method.md`
- `docs/templates/ff30_sic_ranges.template.csv`
- `docs/templates/scope3_stock_candidates.template.csv`

For Scope 3 sensitivity (with Coal vs without Coal), use:
- `run_scope3_sensitivity_with_and_without_coal(...)` in `fin41360/workflows.py`
- `fin41360_report.ipynb` now includes both `with_coal_30` and `drop_coal_29`
  views, plus fixed-axis comparison plots (drop-coal limits and Scope 2 limits).

## Notes for contributors

- Refactored analysis helpers live in:
  - `fin41360/frontier_workflow.py`
  - `fin41360/workflows.py`
  - `fin41360/plot_frontiers.py`
  - `fin41360/plot_styles.py`
- Plotting defaults can be tuned from the top of `fin41360_report.ipynb`.
- Many plots default to `efficient_frontier_only=True` (toggleable for sanity checks).
