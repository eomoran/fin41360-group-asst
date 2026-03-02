# ADR-0008: Scope 3 Coal Sensitivity And Common-Window Policy

- Status: Accepted (report baseline + sensitivity appendix)
- Date: 2026-03-02
- Scope: Scope 3 (`fin41360/workflows.py`, `fin41360_report.ipynb`)

## Context

Scope 3 compares industry and stock frontiers on a common sample window. The
Coal representative stock (currently `BTU` in selected mapping) can materially
shorten the overlap window, reducing `T` and mechanically increasing estimated
shrinkage intensity in Bayes-Stein style workflows.

## Decision (Implemented Now)

1. Report two Scope 3 scenarios:
- `with_coal_30`: keep Coal in both universes (30 vs 30).
- `drop_coal_29`: drop Coal industry and the mapped Coal stock (BTU in current
  selected mapping), producing 29 vs 29.

2. Always report the resulting common windows (`common_start`, `common_end`,
   `n_obs`) for both scenarios.

3. Keep `with_coal_30` as the headline baseline and treat `drop_coal_29` as a
   sensitivity view for sample-length effects.

## Rationale

- Preserves full FF30 breadth for the main result.
- Explicitly shows the tradeoff between breadth and overlap length.
- Makes shrinkage-intensity differences interpretable as partly `T`-driven.

## Revisit Triggers And Tests

1. Trigger: Coal representative ticker changes in selected mapping.
- Test: verify dropped stock column in diagnostics and rerun both scenarios.
- Revisit threshold: dropped stock no longer corresponds to Coal mapping.

2. Trigger: with-Coal and no-Coal conclusions diverge materially.
- Test: compare ranking/sign of headline Sharpe/GMV/TAN conclusions by
  estimator.
- Revisit threshold: qualitative conclusions reverse only under one scenario.

3. Trigger: very short with-Coal overlap.
- Test: monitor `n_obs` and shrinkage-intensity jumps across scenarios.
- Revisit threshold: with-Coal sample too short for stable covariance estimates.

## Consequences

- Scope 3 outputs are more transparent about sample-window sensitivity.
- Report can justify why both 30-asset and 29-asset variants are shown.
