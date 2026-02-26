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

Download and process the core Fama-French datasets (industries, FF3, FF5):

```bash
python -m fin41360.setup_french_data
```

## Run the analysis notebook

Open and run:

- `fin41360_report.ipynb`

The notebook is designed to call descriptively named workflow functions (in `fin41360/workflows.py`) and plotting helpers (in `fin41360/plot_frontiers.py`), keeping notebook logic thin.

## Scope 3 stock data note (important)

Scope 3 currently uses a custom balanced stock file when available:

- `fin41360_data/processed/clean_30_stocks_monthly_returns_balanced_1990_2025.csv`

This file is loaded automatically by `load_stock_returns_monthly(..., source="auto")`, converted from monthly net returns to gross returns, and then used in the frontier pipeline.

The stock-to-FF30-industry mapping for that file is a project crosswalk (not an official Ken French / CRSP one-stock mapping). Some assignments, especially residual-style categories such as `Other`, remain a TODO for validation before final submission.

## Notes for contributors

- Refactored analysis helpers live in:
  - `fin41360/frontier_workflow.py`
  - `fin41360/workflows.py`
  - `fin41360/plot_frontiers.py`
  - `fin41360/plot_styles.py`
- Plotting defaults can be tuned from the top of `fin41360_report.ipynb`.
- Many plots default to `efficient_frontier_only=True` (toggleable for sanity checks).
