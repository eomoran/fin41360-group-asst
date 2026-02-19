# ADR-0001: Frontier Plot Encoding And Axis Policy

- Status: Accepted (current baseline), with deferred items recorded as TODO/LATER
- Date: 2026-02-19
- Scope: FIN41360 Assignment 1 charts (`fin41360/plot_styles.py`, `fin41360/plot_frontiers.py`)

## Context

Scope 2 plots became hard to read due to:

- mixed legend semantics (line series and marker entries duplicated in one legend)
- unstable visual framing from ad-hoc x-axis limits
- unclear naming of frontier variants

We also discussed a stronger global visual encoding strategy for later scopes.

## Decision (Implemented Now)

1. Keep separate legends:
- Frontier legend: line series only.
- Marker legend: portfolio type only (`GMV`, `Tangency`) in neutral grey.

2. Keep line style differentiation for transformation type in Scope 2:
- `Sample (no shrinkage)`: solid
- `Bayes-Stein mean shrinkage`: dashed
- `Bayes-Stein mean+cov shrinkage`: dash-dot

3. Use configurable axis policy via `x_mode` in plotting:
- `frontier`: scale from max frontier volatility
- `tangency`: scale from max tangency volatility
- `max`: max of both

4. Use assignment-oriented title text for Scope 2:
- `Question 2: 30-Industry MV Frontier (...)`

## Rationale

- Better readability with less legend clutter.
- Marker semantics are stable and portable across scopes.
- Configurable axis policy prevents repeated ad-hoc edits in notebook cells.

## Deferred Items (TODO/LATER)

1. `TODO(LATER)` Evaluate a `gmv_center` x-axis policy:
- Frame x-limits around GMV/tangency geometry (instead of frontier/tangency max only).

2. `TODO(LATER)` Evaluate cross-scope fixed axis ranges:
- Keep the same x-range for comparable figures in the report for faster visual comparison.

3. `TODO(LATER)` Evaluate global semantic encoding refactor:
- Candidate convention:
  - Color = asset universe (industries, stocks, FF3, FF5, proxies)
  - Line style = estimator/transformation (sample, BS mean, BS mean+cov)
  - Marker shape = portfolio type (GMV/Tangency)
  - Marker fill = proxy vs non-proxy (if needed)
- This is intentionally deferred until Scope 3-7 plotting functions are implemented,
  to avoid partial style churn.

## Consequences

- Scope 2 is cleaner and closer to report-ready now.
- Later scopes can adopt the same conventions with lower rework risk.
- A larger global style refactor remains possible but is explicitly tracked.
