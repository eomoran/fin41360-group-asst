# ADR-0009: Scope 8 Staged Constraints, Persistence, And Shrinkage Policy

- Status: Accepted (current baseline)
- Date: 2026-03-06
- Scope: Scope 8 (`fin41360/workflows.py`, `fin41360/plot_frontiers.py`)

## Context

Scope 8 evolved from a single extension idea into multiple related analyses:
- full-sample constrained frontier comparison,
- constrained IS/OOS persistence checks,
- shrinkage sensitivity for persistence.

Without explicit structure, notebook outputs and interpretation can mix
full-sample and split-based claims.

## Decision (Implemented Now)

1. Scope 8 is staged into three parts:
- **8.1** Full common-sample constraints (Scope 6 style), focused on
  constrained vs unconstrained frontier/tangency geometry.
- **8.2** Constraints + IS/OOS persistence, with fixed IS-selected portfolios
  evaluated in OOS and plotted against IS/OOS frontiers.
- **8.3** BS shrinkage persistence sensitivity, where shrinkage is estimated on
  IS only and persistence is evaluated OOS.

2. Constraint set baseline:
- Primary bounds are `w_i >= 0` and `w_i >= -0.25`.

3. Shrinkage protocol for persistence:
- For 8.3, shrinkage parameters are estimated using **IS data only**.
- OOS is for evaluation only (no look-ahead).

4. 8.2 visualization policy:
- Plot full IS/OOS frontiers + key point markers.
- Add universe tag in each panel (`Industries`, `FF5`, `Proxy5`).
- Provide FF5-based axis-scaling alternative for cross-panel comparability.

5. Recombination diagnostic:
- For constrained cases, report how close the best convex combination of
  IS constrained GMV/TAN gets to OOS constrained TAN (`constrained_recombination` table).

## Rationale

- Staging separates “shape comparison” (8.1) from “persistence” (8.2/8.3).
- IS-only shrinkage estimation keeps persistence inference clean.
- Shared plotting/label conventions reduce interpretation ambiguity.
- Recombination diagnostics provide an actionable interpretation for cases where
  IS TAN remains near OOS efficient sets but is not OOS TAN.

## Revisit Triggers And Tests

1. Trigger: constrained persistence claims depend strongly on panel scaling.
- Test: compare per-panel vs FF5-based fixed limits.
- Revisit threshold: qualitative conclusions change with scaling policy.

2. Trigger: shrinkage sensitivity appears negligible.
- Test: track Sharpe-gap and replication changes between sample and BS variants.
- Revisit threshold: consistently immaterial deltas across universes/windows.

3. Trigger: constrained recombination fails frequently.
- Test: monitor `sr_gap_vs_oos_tan` and `weight_l2_distance`.
- Revisit threshold: persistent large gaps despite high IS/OOS frontier retention.

## Consequences

- Scope 8 outputs are decision-separable and easier to audit.
- Persistence interpretation is aligned with no-look-ahead guardrails.
- Appendix-level shrinkage diagnostics can be retained without diluting primary
  Scope 8 constraint findings.
