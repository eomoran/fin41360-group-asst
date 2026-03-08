# Legacy Notebook Investigation

This memo compares the legacy notebooks in `/Users/eoinmoran/Downloads/PRM ASSIGNMENT` against the current repo. The repo is treated as the intended baseline. The goal is to identify which differences are just refactors versus differences that may change results or interpretation.

## Current Baseline

- Repo notebook entrypoint: `fin41360_report.ipynb`
- Main workflow entrypoints: `run_scope2_industries_sample_vs_bs`, `run_scope3_sensitivity_with_and_without_coal`, `run_scope4_industries_with_rf`, `run_scope5_industries_vs_ff`
- Supporting policy docs:
  - `README.md`
  - `docs/adr/0003-bayes-stein-and-covariance-shrinkage-policy.md`
  - `docs/adr/0007-scope7-split-and-inference-guardrails.md`
- Working tree note: the repo is not clean. At the time of investigation, `fin41360/plot_frontiers.py` and `report/main.tex` were modified, and multiple report artifacts were untracked.

## Notebook To Repo Mapping

| Legacy notebook | Current repo target | Notes |
| --- | --- | --- |
| `Q2 Main.ipynb` | `fin41360_report.ipynb` Scope 2 cells and `fin41360/workflows.py::run_scope2_industries_sample_vs_bs` | Scope 2 frontier work plus some CAL consistency checks that overlap with Scope 4 concepts |
| `Q3.ipynb` | `fin41360_report.ipynb` Scope 3 cells and `fin41360/workflows.py::run_scope3_sensitivity_with_and_without_coal` / `run_scope3_industries_vs_stocks` | Same broad problem, but current repo changed the stock source and comparison design materially |
| `Q4.ipynb` | `fin41360_report.ipynb` Scope 4 cells and `fin41360/workflows.py::run_scope4_industries_with_rf` | Legacy notebook is stateful and assumes Q2 has already defined variables |
| `Q5.ipynb` | `fin41360_report.ipynb` Scope 5 cells and `fin41360/workflows.py::run_scope5_industries_vs_ff` | Current repo absorbs the “rerun Q4 in excess-return form” logic directly into the workflow |

## Behavioral Spot Checks Run

These checks were run against the local repo data and current workflow code:

- Scope 2 current baseline:
  - sample: `1980-01` to `2025-12`
  - observations: `552`
  - covariance shrinkage: `ledoit_wolf`
  - effective covariance shrinkage intensity: `0.0156`
  - Bayes-Stein target mean: `0.01085`
- Scope 3 current baseline:
  - common sample: `2017-05` to `2025-11`
  - observations: `103`
  - stock source: `fin41360_data/processed/scope3_selected_30_stock_monthly_gross.csv`
  - covariance shrinkage: `ledoit_wolf`
- Scope 4 current baseline:
  - sample: `1980-01` to `2025-12`
- Scope 5 current baseline:
  - sample: `1980-01` to `2025-12`
  - estimation space: excess returns with `rf=0`

These spot checks confirm the repo is using current local processed datasets, not the absolute file paths embedded in the legacy notebooks.

## Material Differences

| Notebook | Difference | Class | Severity | Likely impact | Recommendation |
| --- | --- | --- | --- | --- | --- |
| `Q2 Main.ipynb` | Legacy notebook loads industry and RF data from hard-coded local Excel paths under `/Users/kevinoregan/...`; repo loads processed local French data via `process_french.py` | `data/input change` | Medium | Different source plumbing, but likely equivalent after conversion | `document` |
| `Q2 Main.ipynb` | Legacy notebook uses net-return decimals directly; repo stores industries and RF in gross form and converts internally | `data/input change` | Low | Mostly convention-level if conversion is correct | `ignore` |
| `Q2 Main.ipynb` | Legacy Bayes-Stein mean target is the sample GMV mean (`mu0 = mu_hat' w_gmv`); repo baseline uses cross-sectional grand mean per ADR-0003 | `methodology change` | High | Can change shrinkage intensity, frontier shape, and tangency statistics | `validate numerically` |
| `Q2 Main.ipynb` | Legacy covariance shrinkage is fixed identity shrinkage with `delta = (N+2)/(T+N+2)`; repo baseline uses Ledoit-Wolf, with fixed lambda retained only as an override | `methodology change` | High | Can change tangency weights and Sharpe materially | `validate numerically` |
| `Q2 Main.ipynb` | Legacy notebook contains extra diagnostic cells exploring alternate Bayes-Stein definitions on excess returns and grand-mean targets | `output/reporting change` | Medium | Shows the legacy notebook was still resolving methodology | `document` |
| `Q3.ipynb` | Legacy notebook builds a hand-picked 30-stock universe from explicit Yahoo tickers and saves a balanced `1988-2025` panel; repo currently prefers SIC-selected cached gross returns | `data/input change` | High | Changes the stock universe itself, not just implementation | `validate numerically` |
| `Q3.ipynb` | Legacy stock sample is balanced `1988-01` to `2025-12`; repo current Scope 3 common window is `2017-05` to `2025-11` because it uses the selected stock file | `data/input change` | High | Huge change in sample length and likely in estimation stability | `validate numerically` |
| `Q3.ipynb` | Legacy industry comparison manually matches industries to the stock window; repo wraps this in `run_scope3_sensitivity_with_and_without_coal` and also adds coal-drop sensitivity and industry stability diagnostics | `refactor-only` | Low | Current repo is more structured and broader in scope | `ignore` |
| `Q3.ipynb` | Legacy notebook uses the same GMV-target Bayes-Stein plus fixed identity covariance shrinkage pattern as Q2; repo uses grand-mean plus Ledoit-Wolf baseline | `methodology change` | High | Affects both industry and stock frontiers | `validate numerically` |
| `Q4.ipynb` | Legacy notebook depends on `%run "./Q2 main.ipynb"` and pre-existing variables; repo computes the full Scope 4 state inside `run_scope4_industries_with_rf` | `refactor-only` | Low | Improves reproducibility and removes notebook-order drift | `ignore` |
| `Q4.ipynb` | Legacy notebook adds an explicit proof cell that the CAL touches the frontier at one point; repo exposes the same construction but not the exact proof cell | `output/reporting change` | Low | Reporting difference only | `document` |
| `Q5.ipynb` | Legacy notebook loads FF3/FF5 directly from processed CSV file paths and reruns Q4 in excess-return form; repo uses typed loaders and computes industries in excess-return space directly in `run_scope5_industries_vs_ff` | `refactor-only` | Low | Mostly architectural cleanup | `ignore` |
| `Q5.ipynb` | Legacy notebook is stateful via `%run` of Q2-Q4; repo removes hidden state and centralizes formulas in workflows | `refactor-only` | Low | Reproducibility improvement | `ignore` |
| `Q5.ipynb` | No clear methodology divergence was found between legacy Q5 and current Scope 5 beyond data-loading architecture and notebook structure | `refactor-only` | Low | Scope 5 appears substantively preserved | `ignore` |

## Notebook Notes

### Q2 Main.ipynb

This is the clearest example of a substantive methodology divergence rather than a simple refactor. The current repo has intentionally moved away from the legacy Bayes-Stein baseline.

- Legacy mean shrinkage target: sample GMV mean
- Repo mean shrinkage target: cross-sectional grand mean
- Legacy covariance shrinkage: fixed identity-shrinkage formula
- Repo covariance shrinkage: Ledoit-Wolf by default, fixed shrinkage only as an explicit override

This difference is intentional and documented in ADR-0003, so it should not be treated as a regression by default. It does mean any mismatch in Scope 2 outputs versus the legacy notebook is expected unless the repo is run with a legacy-style override.

### Q3.ipynb

This is the highest-impact notebook from a data perspective. The repo is no longer comparing the same stock panel as the legacy notebook.

- Legacy stock universe: hand-picked Yahoo tickers, balanced panel beginning in 1988
- Repo stock universe: SIC-selected cached stock file, currently producing a much shorter common window
- Legacy notebook performs a direct 30 industries vs 30 stocks comparison
- Repo extends this into a sensitivity package with coal and no-coal views plus industry window diagnostics

The current repo design is more auditable, but the resulting sample window is much shorter. That is likely the single most important difference to validate numerically before relying on historical comparability with the older notebook.

### Q4.ipynb

Q4 appears substantively preserved. The main change is architectural:

- legacy: depends on prior notebook execution
- repo: self-contained workflow returning aligned inputs, curve data, CML, and summary points

The extra CAL-touch proof in the legacy notebook is useful as an appendix diagnostic but not a missing methodology feature.

### Q5.ipynb

Q5 mostly looks like a monolithic predecessor to current Scope 5.

- Both implementations work in excess-return space for the combined industries/FF3/FF5 comparison
- The repo eliminates duplicated logic by handling industry excess-return conversion inside the workflow
- No major formula-level divergence was identified from static inspection

This notebook does not currently present an obvious porting target.

## Recommended Follow-Ups

1. Run one legacy-mode numerical comparison for Scope 2 by overriding the repo to use:
   - Bayes-Stein target = GMV mean
   - covariance shrinkage = fixed identity formula
   This will confirm how much of the output drift is pure policy change versus implementation error.
2. Run a Scope 3 comparison using the legacy balanced stock file `clean_30_stocks_monthly_returns_balanced_1990_2025.csv` or a reconstructed equivalent to quantify how much of the repo-vs-legacy difference comes from the stock universe and shortened common sample.
3. If the report still needs examiner-facing appendix diagnostics, consider adding a compact CAL consistency check for Scope 4, since the legacy notebook already used it as a proof-of-correctness step.

## Bottom Line

- `Q2 Main.ipynb`: materially different methodology from the repo baseline, but the divergence is intentional and documented.
- `Q3.ipynb`: materially different data universe and sample window from the repo baseline; this is the most important difference to validate.
- `Q4.ipynb`: mostly a clean refactor into a reproducible workflow.
- `Q5.ipynb`: mostly a clean refactor into a reusable excess-return workflow.
