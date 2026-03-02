# Scope 3 FF30 Stock Mapping Method

## Goal
Create a defensible mapping for Scope 3 where each of the 30 Fama-French industries has one representative stock.

## Core rule
Use SIC-code classification as the primary mapping rule:
- `ticker -> SIC -> FF30 industry`
- FF30 industry assignment must come from the official FF30 SIC ranges (`Siccodes30` source).

This avoids manual "looks-like" mapping and keeps the process auditable.

## Selection policy (one stock per industry)
After candidate stocks are mapped to FF30 industries by SIC:
1. Prefer stocks with full sample coverage.
2. Prefer stocks with longer in-sample history.
3. Prefer higher median dollar volume (liquidity).
4. Prefer larger market cap.
5. Use ticker alphabetical order as a deterministic final tie-breaker.

This gives a consistent representative that is:
- actually in the intended industry by SIC rule,
- tradable enough for portfolio interpretation,
- stable enough over the analysis sample.

## Required input files
1. `ff30_sic_ranges.csv`
  - Source from the official FF30 SIC definition file.
  - Required columns:
    - `ff30_industry`
    - `sic_start`
    - `sic_end`
2. `scope3_stock_candidates.csv`
  - Candidate stocks and metadata.
  - Required columns:
    - `ticker`
    - `sic`
    - `market_cap_usd`
    - `median_dollar_volume_usd`
    - `first_return_month`
    - `last_return_month`

Template files are provided in `docs/templates/`.

Official source:
- Ken French data library detail page for 30 industries (includes SIC definition download link): `det_30_ind_port.html`
- Siccodes30 zip URL: `https://mba.tuck.dartmouth.edu/pages/Faculty/ken.french/ftp/Siccodes30.zip`

Automated option:
- `python -m fin41360.setup_scope3_data` will download/cache Siccodes30 and parse to
  `fin41360_data/processed/ff30_sic_ranges.csv` if that file does not already exist.

## Run
```bash
python -m fin41360.setup_scope3_mapping \
  --candidates fin41360_data/processed/scope3_stock_candidates.csv \
  --sic-ranges fin41360_data/processed/ff30_sic_ranges.csv \
  --sample-start 1990-01 \
  --sample-end 2025-12 \
  --min-sample-months 360
```

Or run the full bootstrap (recommended on a new machine):

```bash
python -m fin41360.setup_scope3_data
```

## Outputs
- `scope3_candidates_mapped_ff30.csv`
  - Candidate list with mapping status (`mapped_unique`, `unmapped_sic`, etc.)
- `scope3_selected_30_stocks_ff30.csv`
  - Final one-stock-per-industry selection.
- `scope3_mapping_audit_summary.csv`
  - Coverage and mapping diagnostics for QA / report appendix.

## Coal Sensitivity Workflow
To support report sensitivity analysis when Coal has short stock history:

1. Base case: keep Coal in both universes (`with_coal_30`).
2. Alternative: drop Coal from both universes (`drop_coal_29`).
3. Industry stability check: compare 30-industry frontier on:
  - full available sample
  - short common window implied by the with-Coal case.

Implemented in:
- `run_scope3_sensitivity_with_and_without_coal(...)` in `fin41360/workflows.py`.

## Reporting language (recommended)
In the report, state:
- the FF30 mapping is SIC-range-based (not subjective),
- the one-stock choice is a deterministic liquidity/history screen within each mapped industry,
- any missing/ambiguous mappings are explicitly disclosed in the audit output.
