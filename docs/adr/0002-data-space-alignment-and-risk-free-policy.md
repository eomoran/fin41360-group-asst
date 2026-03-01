# ADR-0002: Data Space, Alignment, And Risk-Free Handling

- Status: Accepted (current baseline)
- Date: 2026-03-01
- Scope: Scope 2-7 workflows (`fin41360/workflows.py`, `fin41360/mv_frontier.py`)

## Context

Team members implemented similar analysis with different assistants and made
different implicit choices around:

- gross vs net vs excess return space,
- sample alignment across assets and risk-free data,
- how risk-free is incorporated into statistics and tangency construction.

These choices affect reproducibility and comparability across scopes.

## Decision (Implemented Now)

1. Core estimation space:
- Use **net returns** for mean/covariance estimation.
- Gross inputs are converted to net via `compute_moments_from_gross(...)`.

2. Excess-return scopes:
- Scope 5 and Scope 6 run in **excess-return space** with `rf=0`.
- Industry/proxy gross/net series are converted to excess by subtracting aligned RF.

3. Risk-free handling in non-excess scopes:
- Scope 2-4 use scalar `rf_mean` computed from aligned monthly RF net returns.
- Tangency and excess means are computed using this aligned `rf_mean`.

4. Alignment rule:
- Use date **intersection** across required series in each scope.
- If intersection is empty, fail fast with explicit errors.

## Rationale

- Matches existing module contracts and avoids mixed-space mistakes.
- Keeps formulas transparent and consistent across scopes.
- Preserves comparability of means/vols/Sharpe within each scope.

## Revisit Triggers And Tests

Revisit this ADR if any trigger is true:

1. Trigger: materially different outputs from gross-space plotting requests.
- Test: recompute tables in net vs gross display form.
- Expectation: weights, vol, Sharpe identical; only mean level shifts by +1 in gross display.

2. Trigger: large sample loss from intersection alignment.
- Test: report retained observations as `% kept` vs each input source.
- Revisit threshold: any scope keeps < 80% of the intended period.

3. Trigger: sensitivity to scalar RF approximation.
- Test: compare scalar-`rf_mean` tangency vs period-varying RF treatment.
- Revisit threshold: tangency Sharpe changes by > 5% relative.

## Consequences

- Current outputs are internally consistent.
- Gross-return charts can be added as a display layer without changing core results.
- Scope metadata should always report alignment window and `n_obs`.
